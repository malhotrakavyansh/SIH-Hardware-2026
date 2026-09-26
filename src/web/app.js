/**
 * app.js -- Nakshatra 3D web UI.
 *
 * Pure state-machine renderer: every visible change is driven by a message
 * from /ws/ui (see PROTOCOL.md). No polling, no client-side guessing about
 * what the edge/server are doing.
 *
 * Perf budget: this laptop is simultaneously running 100ms wake-word
 * inference and Vosk ASR, so this scene must not steal CPU. Everything
 * animated (star drift + twinkle, planet clouds, rings) is GPU-driven --
 * per-frame JS only updates a handful of uniforms/transforms (camera orbit,
 * satellite marker, two CSS variables), never per-star or per-vertex work.
 *
 * Depth model (camera at the origin looking down -Z):
 *   far stars   z -120..-180   barely move with parallax, drift ~0
 *   mid stars   z  -55..-85
 *   planet      centre (0,-30,-34), r 24; limb ~z -38, horizon ~1/3 up the screen
 *   orbit ring  centre (0,-1,-34), r 15, tilted so its lower arc recedes
 *               behind the planet limb (hidden by the depth test)
 *   near stars  z  -18..-28   largest, fastest, most parallax
 *   glass UI    DOM layer, parallax-shifted opposite to the scene
 * Mouse parallax translates the camera (not rotates), so on-screen shift
 * falls off as 1/depth -- the depth ordering above is what the eye reads.
 */

(function () {
  "use strict";

  const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ACCENT = 0x3fe0ff;
  const CONTENT_HOLD_MS = 10000; // how long the mission card stays up before auto-returning to IDLE

  // ===========================================================================
  // Three.js scene
  // ===========================================================================

  const canvas = document.getElementById("scene-canvas");
  // antialias OFF: 4x MSAA multiplied the fill of the full-screen background
  // and the large planet disc and was the single biggest GPU cost on the
  // demo laptop's integrated GPU. Nothing here needs it -- the ring band's
  // alpha already falls to 0 at its edges, stars are soft sprites, and the
  // planet limb sits under the halo glow.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: "high-performance" });
  renderer.setClearColor(0x03050a, 1);
  renderer.autoClearColor = false; // the background pass covers every pixel
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 400);

  const PLANET_CENTER = new THREE.Vector3(0, -30, -34);
  const PLANET_R = 24;
  const LOOK_TARGET = new THREE.Vector3(0, 0, -34);
  // Sun low, behind and to the left: a lit crescent along the upper-left limb
  // with the terminator sweeping across the visible cap, so the planet reads
  // as a lit sphere and the side behind the mission card stays dark.
  const SUN_DIR = new THREE.Vector3(-0.8, 0.42, -0.5).normalize();

  // Point sizes are authored in px at 1080p; scaled with viewport height so
  // the scene looks the same at 1920x1080 (recording) and on a projector.
  const pointScale = { value: 1 };

  function resize() {
    const w = window.innerWidth, h = window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    bgMat.uniforms.uAspect.value = w / h; // resize() first runs after bgMat is built
    pointScale.value = renderer.getPixelRatio() * (h / 1080);
  }
  window.addEventListener("resize", resize);

  const shared = { uTime: { value: 0 } };

  // ---- background: nebula wash + vignette in ONE opaque full-screen pass ---
  // Drawn first with depth test/write off, in clip space, so it replaces the
  // clear instead of adding a blended layer on top. (A separate additive
  // nebula quad plus a DOM vignette over the canvas cost several ms/frame of fill
  // on the demo laptop's integrated GPU.) The washes are anchored to the view
  // direction so they swing gently with the camera orbit.

  const bgMat = new THREE.ShaderMaterial({
    uniforms: { uAspect: { value: 16 / 9 }, uPan: { value: new THREE.Vector2() } },
    vertexShader: `
      varying vec2 vP;
      void main() { vP = position.xy; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      uniform float uAspect;
      uniform vec2 uPan;
      varying vec2 vP;
      void main() {
        vec2 p = vec2(vP.x * uAspect, vP.y) * 0.5 + uPan;
        float a = exp(-dot(p - vec2(-0.55, 0.30), p - vec2(-0.55, 0.30)) * 1.6);
        float b = exp(-dot(p - vec2(0.70, 0.12), p - vec2(0.70, 0.12)) * 2.2);
        vec3 col = vec3(0.012, 0.020, 0.040) + vec3(0.05, 0.10, 0.22) * a + vec3(0.12, 0.05, 0.18) * b;
        // vignette: darken toward the corners, centre slightly above middle
        vec2 v = vec2(vP.x, vP.y - 0.1);
        col *= 1.0 - 0.55 * smoothstep(0.55, 1.35, length(v * vec2(0.85, 1.0)));
        gl_FragColor = vec4(col, 1.0);
      }
    `,
    depthTest: false,
    depthWrite: false,
  });
  const bgGeom = new THREE.BufferGeometry();
  bgGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
  const background = new THREE.Mesh(bgGeom, bgMat);
  resize();
  background.frustumCulled = false;
  background.renderOrder = -100;
  scene.add(background);

  // ---- starfield: three depth layers ----------------------------------------
  // Positions are generated once; drift (a wrapped x-offset) and twinkle run
  // in the vertex shader, so a layer costs one uniform write per frame.

  const STAR_VERT = `
    attribute float aSize;
    attribute float aPhase;
    attribute vec3 color;
    uniform float uTime;
    uniform float uSpeed;
    uniform float uHalfW;
    uniform float uScale;
    varying vec3 vColor;
    varying float vTwinkle;
    void main() {
      vec3 p = position;
      p.x = mod(p.x + uTime * uSpeed + uHalfW, 2.0 * uHalfW) - uHalfW;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = aSize * uScale;
      vColor = color;
      vTwinkle = 0.72 + 0.28 * sin(uTime * (0.7 + aPhase * 1.6) + aPhase * 6.2831);
    }
  `;
  const STAR_FRAG = `
    uniform float uOpacity;
    varying vec3 vColor;
    varying float vTwinkle;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      float core = smoothstep(0.5, 0.0, d);
      float a = core * core * vTwinkle * uOpacity;
      if (a < 0.01) discard;
      gl_FragColor = vec4(vColor, a);
    }
  `;

  function makeStarLayer({ count, zNear, zFar, sizeMin, sizeMax, speed, opacity, minYFrac }) {
    const pos = new Float32Array(count * 3);
    const size = new Float32Array(count);
    const phase = new Float32Array(count);
    const col = new Float32Array(count * 3);
    // Half-extent of the layer, generous enough to cover wide aspect ratios
    // plus the camera orbit and parallax without ever revealing an edge.
    const halfW = zFar * 1.3;
    for (let i = 0; i < count; i++) {
      const z = -(zNear + Math.random() * (zFar - zNear));
      const d = -z;
      const yMin = minYFrac != null ? minYFrac * d : -0.75 * d;
      pos[i * 3] = (Math.random() * 2 - 1) * halfW;
      pos[i * 3 + 1] = yMin + Math.random() * (0.75 * d - yMin);
      pos[i * 3 + 2] = z;
      size[i] = sizeMin + Math.pow(Math.random(), 2.2) * (sizeMax - sizeMin);
      phase[i] = Math.random();
      const b = 0.6 + Math.random() * 0.4;
      const tint = Math.random();
      if (tint < 0.16) { col[i * 3] = b * 0.72; col[i * 3 + 1] = b * 0.84; col[i * 3 + 2] = b; }
      else if (tint < 0.22) { col[i * 3] = b; col[i * 3 + 1] = b * 0.86; col[i * 3 + 2] = b * 0.7; }
      else { col[i * 3] = b; col[i * 3 + 1] = b; col[i * 3 + 2] = b * 0.96; }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("aSize", new THREE.BufferAttribute(size, 1));
    geom.setAttribute("aPhase", new THREE.BufferAttribute(phase, 1));
    geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: shared.uTime,
        uSpeed: { value: REDUCED_MOTION ? 0 : speed },
        uHalfW: { value: halfW },
        uScale: pointScale,
        uOpacity: { value: opacity },
      },
      vertexShader: STAR_VERT,
      fragmentShader: STAR_FRAG,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const pts = new THREE.Points(geom, mat);
    pts.frustumCulled = false; // positions are rewritten in the shader
    scene.add(pts);
    return pts;
  }

  // Counts chosen to keep the frame under budget on the demo laptop's
  // integrated GPU -- lower these (near last) before dropping a layer.
  makeStarLayer({ count: 2400, zNear: 120, zFar: 180, sizeMin: 0.9, sizeMax: 2.0, speed: 0.10, opacity: 0.80 });
  makeStarLayer({ count: 900,  zNear: 55,  zFar: 85,  sizeMin: 1.5, sizeMax: 3.0, speed: 0.22, opacity: 0.85 });
  // Near stars stay above the horizon: real stars never sit in front of a planet.
  makeStarLayer({ count: 170,  zNear: 18,  zFar: 28,  sizeMin: 2.6, sizeMax: 5.0, speed: 0.45, opacity: 0.95, minYFrac: -0.05 });

  // ---- planet: lit sphere with terminator, clouds and a fresnel rim --------

  const planetMat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: shared.uTime,
      uSun: { value: SUN_DIR },
      uAtmo: { value: new THREE.Color(0x4fb8ff) },
    },
    vertexShader: `
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vObjN;
      void main() {
        vObjN = normal;
        vN = normalize(mat3(modelMatrix) * normal);
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }
    `,
    fragmentShader: `
      uniform float uTime;
      uniform vec3 uSun;
      uniform vec3 uAtmo;
      varying vec3 vN;
      varying vec3 vW;
      varying vec3 vObjN;
      float h(vec3 n) {
        return sin(n.x * 11.0 + sin(n.z * 7.0) * 2.0) * sin(n.z * 13.0 + n.y * 5.0)
             + 0.5 * sin(n.x * 29.0 + n.z * 23.0) * sin(n.y * 31.0);
      }
      void main() {
        vec3 N = normalize(vN);
        vec3 V = normalize(cameraPosition - vW);
        float ndl = dot(N, uSun);
        float day = smoothstep(-0.08, 0.35, ndl);

        // surface: ocean with darker landmasses, low contrast
        float land = smoothstep(0.35, 0.75, h(vObjN) * 0.5 + 0.5);
        vec3 surf = mix(vec3(0.020, 0.075, 0.150), vec3(0.055, 0.085, 0.070), land);

        // slow cloud bands, rotating with time
        float ca = uTime * 0.004;
        vec3 cn = vec3(vObjN.x * cos(ca) - vObjN.z * sin(ca), vObjN.y, vObjN.x * sin(ca) + vObjN.z * cos(ca));
        float cl = smoothstep(0.74, 0.98, sin(cn.x * 26.0 + sin(cn.z * 31.0 + cn.y * 11.0) * 2.1) * 0.5 + 0.5);
        cl *= smoothstep(0.35, 0.85, sin(cn.z * 9.0 + cn.x * 6.0 + sin(cn.y * 14.0)) * 0.5 + 0.5);

        vec3 col = surf * (0.035 + 1.15 * day) + vec3(0.78, 0.86, 0.95) * cl * 0.15 * day;

        // warm band along the terminator
        col += vec3(0.95, 0.45, 0.18) * exp(-pow(ndl / 0.07, 2.0)) * 0.10;

        // atmosphere seen edge-on: fresnel rim, bright on the day side,
        // a faint blue line still visible across the night side
        float fr = pow(1.0 - max(dot(N, V), 0.0), 2.6);
        col += uAtmo * fr * (0.10 + 1.25 * smoothstep(-0.25, 0.45, ndl));

        gl_FragColor = vec4(col, 1.0);
      }
    `,
  });
  const planet = new THREE.Mesh(new THREE.SphereGeometry(PLANET_R, 96, 64), planetMat);
  planet.position.copy(PLANET_CENTER);
  scene.add(planet);

  // Atmospheric halo: a slightly larger back-faced shell. For each fragment
  // the view ray's closest approach to the planet centre gives the altitude
  // it grazes, so the glow falls off smoothly above the limb and is lit by
  // the same sun as the surface (bright day limb, fading across the night).
  const HALO_R = PLANET_R * 1.085;
  const haloMat = new THREE.ShaderMaterial({
    uniforms: {
      uCenter: { value: PLANET_CENTER },
      uR: { value: PLANET_R },
      uH: { value: HALO_R - PLANET_R },
      uSun: { value: SUN_DIR },
      uColor: { value: new THREE.Color(0x46b4ff) },
    },
    vertexShader: `
      varying vec3 vW;
      void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vW = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }
    `,
    fragmentShader: `
      uniform vec3 uCenter;
      uniform float uR;
      uniform float uH;
      uniform vec3 uSun;
      uniform vec3 uColor;
      varying vec3 vW;
      void main() {
        vec3 dir = normalize(vW - cameraPosition);
        vec3 oc = uCenter - cameraPosition;
        float t = dot(oc, dir);
        vec3 closest = cameraPosition + dir * t;
        // alt < 0 is inside the silhouette: the planet occludes it by depth,
        // except in the sliver where the tessellated edge falls short of the
        // true sphere -- full glow there, so no dark seam along the limb
        float alt = max(length(closest - uCenter) - uR, 0.0);
        float k = clamp(1.0 - alt / uH, 0.0, 1.0);
        float glow = pow(k, 2.4);
        float lit = 0.18 + 0.95 * smoothstep(-0.35, 0.5, dot(normalize(closest - uCenter), uSun));
        gl_FragColor = vec4(uColor * glow * lit, glow * lit);
      }
    `,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const halo = new THREE.Mesh(new THREE.SphereGeometry(HALO_R, 96, 64), haloMat);
  halo.position.copy(PLANET_CENTER);
  scene.add(halo);

  // ---- orbital / countdown ring ---------------------------------------------
  // A real ring in 3D: tilted about X so its lower arc recedes into the scene
  // and passes behind the planet limb (depth-tested against the planet).
  // Base ring: always present, soft band with faint ticks, dimmer where it's
  // further away. Progress ring: bright arc on top, driven entirely by a
  // single uniform (uProgress, 1 -> 0 over 8s in WAKE).

  const RING_CENTER = new THREE.Vector3(0, -1, -34);
  const RING_R = 15, RING_W = 0.075;
  const RING_TILT = THREE.MathUtils.degToRad(58);

  const RING_VERT = `
    varying vec3 vPos;
    varying float vDepth;
    void main() {
      vPos = position;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mv.z;
      gl_Position = projectionMatrix * mv;
    }
  `;

  const ringUniformsBase = {
    uColor: { value: new THREE.Color(ACCENT) },
    uOpacity: { value: 0.16 },
    uR: { value: RING_R },
    uW: { value: RING_W },
  };

  const baseRingMat = new THREE.ShaderMaterial({
    uniforms: ringUniformsBase,
    vertexShader: RING_VERT,
    fragmentShader: `
      #define PI 3.14159265359
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uR;
      uniform float uW;
      varying vec3 vPos;
      varying float vDepth;
      void main() {
        float r = length(vPos.xy);
        float across = clamp(1.0 - abs(r - uR) / uW, 0.0, 1.0);
        float frac = (atan(vPos.x, vPos.y) + PI) / (2.0 * PI);
        float tick = step(0.82, fract(frac * 96.0)) * 0.9;
        float depthFade = mix(0.35, 1.0, smoothstep(50.0, 22.0, vDepth));
        float a = uOpacity * (pow(across, 1.4) + tick * across) * depthFade;
        gl_FragColor = vec4(uColor, a);
      }
    `,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const progressRingMat = new THREE.ShaderMaterial({
    uniforms: {
      uProgress: { value: 0.0 }, // 0 = hidden, ramps 0->1 on wake flash then depletes 1->0
      uColor: { value: new THREE.Color(ACCENT) },
      uOpacity: { value: 0.0 },
      uR: { value: RING_R },
      uW: { value: RING_W * 1.6 },
    },
    vertexShader: RING_VERT,
    fragmentShader: `
      #define PI 3.14159265359
      uniform float uProgress;
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uR;
      uniform float uW;
      varying vec3 vPos;
      varying float vDepth;
      void main() {
        float angle = atan(vPos.x, vPos.y);
        float frac = (angle + PI) / (2.0 * PI);
        if (frac > uProgress) discard;
        float r = length(vPos.xy);
        float across = clamp(1.0 - abs(r - uR) / uW, 0.0, 1.0);
        float head = smoothstep(uProgress - 0.04, uProgress, frac);   // bright leading edge
        float depthFade = mix(0.45, 1.0, smoothstep(50.0, 22.0, vDepth));
        float a = uOpacity * pow(across, 1.2) * (0.75 + 0.6 * head) * depthFade;
        gl_FragColor = vec4(uColor * (1.0 + head * 0.6), a);
      }
    `,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });

  const ringGroup = new THREE.Group();
  ringGroup.position.copy(RING_CENTER);
  ringGroup.rotation.x = RING_TILT;
  scene.add(ringGroup);

  const baseRing = new THREE.Mesh(new THREE.RingGeometry(RING_R - RING_W * 1.7, RING_R + RING_W * 1.7, 256, 1), baseRingMat);
  const progressRing = new THREE.Mesh(new THREE.RingGeometry(RING_R - RING_W * 1.7, RING_R + RING_W * 1.7, 256, 1), progressRingMat);
  baseRing.renderOrder = 2;
  progressRing.renderOrder = 3;
  ringGroup.add(baseRing);
  ringGroup.add(progressRing);

  // A single satellite marker riding the ring. It is depth-tested like the
  // ring, so it visibly slips behind the planet limb on the far side -- the
  // clearest possible "this is 3D" cue, for one Vector3 update per frame.
  const satMat = new THREE.ShaderMaterial({
    uniforms: { uScale: pointScale, uColor: { value: new THREE.Color(0xbff4ff) } },
    vertexShader: `
      uniform float uScale;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = uScale * clamp(420.0 / -mv.z, 6.0, 22.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, d);
        a = a * a * 0.9 + smoothstep(0.12, 0.0, d);
        gl_FragColor = vec4(uColor, a);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const satGeom = new THREE.BufferGeometry();
  satGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
  const satellite = new THREE.Points(satGeom, satMat);
  satellite.frustumCulled = false;
  satellite.renderOrder = 4;
  ringGroup.add(satellite);
  const SAT_PERIOD_S = 48;

  // ===========================================================================
  // Camera: slow orbit + mouse parallax
  // ===========================================================================
  // The orbit swings the camera a few degrees around the planet centre over
  // tens of seconds (barely perceptible frame to frame). Parallax is applied
  // afterwards as a pure translation in the camera's own plane, so near
  // layers shift more than far ones.

  const CAM_BASE = new THREE.Vector3(0, 0, 0);
  const ORBIT_YAW = THREE.MathUtils.degToRad(3.2);
  const ORBIT_PITCH = THREE.MathUtils.degToRad(1.1);
  const ORBIT_YAW_PERIOD = 70, ORBIT_PITCH_PERIOD = 47;
  const PARALLAX_X = 0.9, PARALLAX_Y = 0.45;   // world units of camera shift
  const UI_PARALLAX_PX = 14;                  // glass layer shift, opposite to the scene

  const parallax = { x: 0, y: 0, sx: 0, sy: 0 };
  if (!REDUCED_MOTION) {
    window.addEventListener("mousemove", (e) => {
      parallax.x = (e.clientX / window.innerWidth - 0.5) * 2;
      parallax.y = (e.clientY / window.innerHeight - 0.5) * 2;
    });
  }

  const _offset = new THREE.Vector3();
  const _target = new THREE.Vector3();
  const _right = new THREE.Vector3();
  const _up = new THREE.Vector3();
  const _euler = new THREE.Euler();
  const rootStyle = document.documentElement.style;
  let lastPlx = null, lastPly = null;

  function updateCamera(t) {
    const yaw = REDUCED_MOTION ? 0 : Math.sin((t / ORBIT_YAW_PERIOD) * Math.PI * 2) * ORBIT_YAW;
    const pitch = REDUCED_MOTION ? 0 : Math.sin((t / ORBIT_PITCH_PERIOD) * Math.PI * 2) * ORBIT_PITCH;
    _euler.set(pitch, yaw, 0, "YXZ");

    _offset.copy(CAM_BASE).sub(PLANET_CENTER).applyEuler(_euler);
    camera.position.copy(PLANET_CENTER).add(_offset);
    _target.copy(LOOK_TARGET).sub(PLANET_CENTER).applyEuler(_euler).add(PLANET_CENTER);
    camera.lookAt(_target);
    bgMat.uniforms.uPan.value.set(yaw * 1.2, -pitch * 1.2);

    parallax.sx += (parallax.x - parallax.sx) * 0.05;
    parallax.sy += (parallax.y - parallax.sy) * 0.05;
    camera.updateMatrixWorld();
    _right.setFromMatrixColumn(camera.matrixWorld, 0);
    _up.setFromMatrixColumn(camera.matrixWorld, 1);
    camera.position.addScaledVector(_right, parallax.sx * PARALLAX_X);
    camera.position.addScaledVector(_up, -parallax.sy * PARALLAX_Y);

    // Glass panels move WITH the pointer while the scene moves against it,
    // so they separate from the background as foreground layers. Only write
    // the CSS variables when the rounded value changes -- no style churn
    // once the pointer is still.
    const plx = Math.round(parallax.sx * UI_PARALLAX_PX * 10) / 10;
    const ply = Math.round(parallax.sy * UI_PARALLAX_PX * 0.6 * 10) / 10;
    if (plx !== lastPlx) { rootStyle.setProperty("--plx", plx + "px"); lastPlx = plx; }
    if (ply !== lastPly) { rootStyle.setProperty("--ply", ply + "px"); lastPly = ply; }
  }

  function updateSatellite(t) {
    const a = (t / SAT_PERIOD_S) * Math.PI * 2;
    const arr = satGeom.attributes.position.array;
    arr[0] = Math.sin(a) * RING_R;
    arr[1] = Math.cos(a) * RING_R;
    arr[2] = 0;
    satGeom.attributes.position.needsUpdate = true;
  }

  // ===========================================================================
  // Render loop -- frame-limited, GPU-only per-frame work, pauses when hidden
  // ===========================================================================

  const FRAME_BUDGET_MS = 1000 / 60;
  const clockStart = performance.now();
  let lastFrameTime = performance.now();
  let rafId = null;
  let running = false;

  // frame-time telemetry: log one average after warmup, then stop.
  let ftSamples = [];
  let ftLogged = false;

  function tick(now) {
    rafId = requestAnimationFrame(tick);

    const elapsed = now - lastFrameTime;
    if (elapsed < FRAME_BUDGET_MS - 1) return; // frame limiter
    lastFrameTime = now;

    const t0 = performance.now();
    const t = (now - clockStart) / 1000;

    shared.uTime.value = REDUCED_MOTION ? 0 : t;
    updateCamera(t);
    updateSatellite(REDUCED_MOTION ? 0 : t);
    updateCountdownRing(now);

    renderer.render(scene, camera);

    const dtMs = performance.now() - t0;
    if (!ftLogged) {
      ftSamples.push(dtMs);
      if (ftSamples.length >= 180) {
        const avg = ftSamples.reduce((a, b) => a + b, 0) / ftSamples.length;
        console.log(`[nakshatra-ui] avg frame render time: ${avg.toFixed(3)}ms over ${ftSamples.length} frames`);
        ftLogged = true;
        ftSamples = null;
      }
    }
  }

  function startLoop() {
    if (running) return;
    running = true;
    lastFrameTime = performance.now();
    rafId = requestAnimationFrame(tick);
  }

  function stopLoop() {
    running = false;
    if (rafId !== null) cancelAnimationFrame(rafId);
    rafId = null;
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopLoop();
    else startLoop();
  });

  startLoop();

  // ===========================================================================
  // Countdown ring state machine (WAKE = 8s depletion window)
  // ===========================================================================

  const WAKE_WINDOW_MS = 8000;
  const BASE_RING_IDLE = 0.16, BASE_RING_ACTIVE = 0.3;
  let ringState = "idle"; // "idle" | "flash" | "counting" | "fade"
  let ringT0 = 0;

  function ringWake() {
    ringState = "flash";
    ringT0 = performance.now();
  }

  function ringSettle() {
    // called on transcript/content -- ring fades out rather than snapping
    ringState = "fade";
    ringT0 = performance.now();
  }

  function ringIdle() {
    ringState = "idle";
  }

  function updateCountdownRing(now) {
    const base = ringUniformsBase.uOpacity;
    const prog = progressRingMat.uniforms;

    if (ringState === "idle") {
      base.value = BASE_RING_IDLE;
      prog.uOpacity.value = 0;
      return;
    }

    if (ringState === "flash") {
      const t = (now - ringT0) / 160; // quick 160ms flash-in
      prog.uProgress.value = 1.0;
      prog.uOpacity.value = Math.min(t, 1) * 0.95;
      base.value = BASE_RING_ACTIVE;
      if (t >= 1) {
        ringState = "counting";
        ringT0 = now;
      }
      return;
    }

    if (ringState === "counting") {
      const remaining = 1 - (now - ringT0) / WAKE_WINDOW_MS;
      prog.uProgress.value = Math.max(remaining, 0);
      prog.uOpacity.value = 0.95;
      if (remaining <= 0) ringState = "idle";
      return;
    }

    if (ringState === "fade") {
      const t = (now - ringT0) / 500;
      prog.uOpacity.value = Math.max(0.95 * (1 - t), 0);
      base.value = Math.max(BASE_RING_ACTIVE * (1 - t), BASE_RING_IDLE);
      if (t >= 1) ringState = "idle";
      return;
    }
  }

  // ===========================================================================
  // DOM refs
  // ===========================================================================

  const el = {
    connBanner: document.getElementById("conn-banner"),
    statusLabel: document.getElementById("status-label"),
    statusText: document.getElementById("status-text"),
    idleBlock: document.getElementById("idle-block"),
    transcriptPanel: document.getElementById("transcript-panel"),
    transcriptLabelText: document.getElementById("transcript-label-text"),
    transcriptText: document.getElementById("transcript-text"),
    levelBar: document.getElementById("level-bar"),
    cardWrap: document.getElementById("card-wrap"),
    cardSlot: document.getElementById("card-slot"),
    hlMs: document.getElementById("hl-ms"),
    wfBar: document.getElementById("wf-bar"),
    mConfidence: document.getElementById("m-confidence"),
    mCount: document.getElementById("m-count"),
    edgeDot: document.getElementById("edge-dot"),
    edgeText: document.getElementById("edge-text"),
  };

  // ===========================================================================
  // App state machine: IDLE -> WAKE -> TRANSCRIBING -> CONTENT -> IDLE
  // ===========================================================================

  let appState = "idle";
  let detectionCount = 0;
  let contentHoldTimer = null;
  let cardLeaveTimer = null;
  let lastShownText = "";
  let confirmedSegments = [];
  let currentLatency = { t1: null, t2: null, t3: null, t4: null, t5: null };

  function setStatusLabel(text, mode) {
    el.statusText.textContent = text;
    el.statusLabel.classList.toggle("wake", mode === "wake");
    el.statusLabel.classList.toggle("matched", mode === "matched");
  }

  function clearContentHold() {
    if (contentHoldTimer) {
      clearTimeout(contentHoldTimer);
      contentHoldTimer = null;
    }
  }

  // Transcript size tier by length (font size in vh): short questions read
  // from across the room at ~88px on 1080p; long sentences step down but
  // never below the 34px floor set in CSS.
  function transcriptTier(text) {
    const n = text.length;
    if (n <= 32) return 8.1;
    if (n <= 64) return 7.0;
    if (n <= 110) return 5.9;
    return 5.0;
  }

  function setTier(text) {
    el.transcriptPanel.style.setProperty("--tier", String(transcriptTier(text)));
  }

  function hideCard(immediate) {
    if (cardLeaveTimer) { clearTimeout(cardLeaveTimer); cardLeaveTimer = null; }
    if (el.cardWrap.hidden) return;
    if (immediate) {
      el.cardWrap.classList.remove("show", "leaving");
      el.cardWrap.hidden = true;
      el.cardSlot.innerHTML = "";
      return;
    }
    el.cardWrap.classList.add("leaving");
    cardLeaveTimer = setTimeout(() => {
      el.cardWrap.classList.remove("show", "leaving");
      el.cardWrap.hidden = true;
      el.cardSlot.innerHTML = "";
      cardLeaveTimer = null;
    }, 450);
  }

  function resetTranscript() {
    el.transcriptPanel.classList.remove("locked", "settle", "sweep", "paired");
    el.transcriptLabelText.textContent = "Hearing";
    el.transcriptText.innerHTML = "";
    lastShownText = "";
    confirmedSegments = [];
  }

  function toIdle() {
    appState = "idle";
    clearContentHold();
    ringIdle();
    setStatusLabel("Listening", null);
    el.idleBlock.hidden = false;
    el.transcriptPanel.hidden = true;
    resetTranscript();
    hideCard(true);
  }

  function toWake(prob) {
    appState = "wake";
    clearContentHold();
    ringWake();
    setStatusLabel("Speak now", "wake");
    hideCard(true);
    el.idleBlock.hidden = true;
    resetTranscript();
    el.transcriptPanel.hidden = false;
    setTier("");
    el.transcriptText.innerHTML = '<span class="waiting">listening<i>.</i><i>.</i><i>.</i></span>';
    detectionCount += 1;
    el.mCount.textContent = String(detectionCount);
    el.mConfidence.textContent = prob != null ? prob.toFixed(2) : "—";
    currentLatency = { t1: null, t2: null, t3: null, t4: null, t5: null };
    renderWaterfall();
  }

  function renderGrowingText(fullText, locked) {
    if (fullText === lastShownText && !locked) return;

    setTier(fullText);
    let html;
    if (fullText.indexOf(lastShownText) === 0 && lastShownText.length > 0) {
      const already = lastShownText;
      const added = fullText.slice(already.length).trim();
      html = staticWords(already) + (added ? " " + wrapWords(added) : "");
    } else {
      html = wrapWords(fullText);
    }
    el.transcriptText.innerHTML = html;
    lastShownText = fullText;

    if (locked) {
      const p = el.transcriptPanel;
      p.classList.remove("sweep", "settle");
      void p.offsetWidth; // restart the one-shot animations
      p.classList.add("locked", "settle", "sweep");
      el.transcriptLabelText.textContent = "Confirmed";
      setTimeout(() => p.classList.remove("settle"), 520);
    }
  }

  // Words already on screen keep their final state (no re-animation) when
  // the partial grows; only the newly arrived words fade up.
  function staticWords(text) {
    return text
      .split(/\s+/)
      .filter(Boolean)
      .map((w) => `<span class="word" style="animation:none;opacity:1;transform:none">${escapeHtml(w)}</span>`)
      .join(" ");
  }

  function wrapWords(text) {
    return text
      .split(/\s+/)
      .filter(Boolean)
      .map((w, i) => `<span class="word" style="animation-delay:${i * 30}ms">${escapeHtml(w)}</span>`)
      .join(" ");
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function toTranscribing() {
    if (appState === "content") return; // don't step backwards over a shown card
    appState = "transcribing";
    setStatusLabel("Transcribing", "wake");
    el.idleBlock.hidden = true;
    el.transcriptPanel.hidden = false;
  }

  function onPartial(text) {
    toTranscribing();
    const combined = [...confirmedSegments, text].join(" ").trim();
    if (combined) renderGrowingText(combined, false);
  }

  function onFinalSegment(text) {
    if (!text) return;
    toTranscribing();
    confirmedSegments.push(text);
    renderGrowingText(confirmedSegments.join(" ").trim(), false);
  }

  function onTranscript(msg) {
    toTranscribing();
    currentLatency.t1 = msg.t1;
    currentLatency.t2 = msg.t2;
    currentLatency.t3 = msg.t3;
    currentLatency.t4 = msg.t4;

    const finalText = msg.text || "(no speech detected)";
    renderGrowingText(finalText, true);

    if (msg.latency_ms != null) {
      el.hlMs.textContent = Math.round(msg.latency_ms);
    } else if (msg.t1 != null && msg.t2 != null) {
      el.hlMs.textContent = Math.round(msg.t2 - msg.t1);
    } else {
      el.hlMs.textContent = "—";
    }
    ringSettle();
    renderWaterfall();
  }

  // FLIP: the transcript's layout position jumps when the card takes space
  // below it; animate from where it was so it visibly glides up instead.
  function flipTranscript(mutate) {
    const p = el.transcriptPanel;
    if (p.hidden || REDUCED_MOTION) { mutate(); return; }
    const before = p.getBoundingClientRect();
    mutate();
    const after = p.getBoundingClientRect();
    const dy = before.top - after.top;
    const s = after.height > 0 ? before.height / after.height : 1;
    if (Math.abs(dy) < 1 && Math.abs(s - 1) < 0.01) return;
    p.animate(
      [
        { transform: `translateY(${dy}px) scale(${s})`, transformOrigin: "50% 0%" },
        { transform: "translateY(0) scale(1)", transformOrigin: "50% 0%" },
      ],
      { duration: 650, easing: "cubic-bezier(0.16, 1, 0.3, 1)" }
    );
  }

  function onContent(msg) {
    appState = "content";
    currentLatency.t5 = msg.t5;
    renderWaterfall();

    // Keep the transcript on screen, above the card. If the transcript
    // event somehow never arrived, show the content's transcript instead.
    if (!lastShownText && msg.transcript) renderGrowingText(msg.transcript, true);
    el.idleBlock.hidden = true;
    el.transcriptPanel.hidden = false;
    setStatusLabel(msg.match ? "Matched" : "No match", msg.match ? "matched" : null);

    if (cardLeaveTimer) { clearTimeout(cardLeaveTimer); cardLeaveTimer = null; }
    el.cardWrap.classList.remove("show", "leaving");
    el.cardSlot.innerHTML = msg.match ? missionCardHtml(msg.match) : noMatchCardHtml();
    wireImageFallback();

    flipTranscript(() => {
      el.transcriptPanel.classList.add("paired");
      el.cardWrap.hidden = false;
    });
    // Commit the card's start pose (pushed back + tilted) with a forced style
    // flush, then add .show so it transitions up into the near plane. Not a
    // rAF: rAF can be throttled and would delay the entrance.
    void el.cardWrap.offsetWidth;
    el.cardWrap.classList.add("show");

    clearContentHold();
    contentHoldTimer = setTimeout(() => {
      if (appState === "content") toIdleAnimated();
    }, CONTENT_HOLD_MS);
  }

  function toIdleAnimated() {
    hideCard(false);
    setTimeout(() => { if (appState === "content") toIdle(); }, 460);
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function formatDate(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || "");
    if (!m) return iso || "";
    return `${parseInt(m[3], 10)} ${MONTHS[parseInt(m[2], 10) - 1]} ${m[1]}`;
  }

  // KB statuses are sentences like "Mission complete (Chandrayaan-3 landing
  // successful)": the head goes in the pill, the parenthetical underneath.
  function splitStatus(status) {
    const m = /^([^(]+?)\s*\((.+)\)\s*$/.exec(status || "");
    return m ? { head: m[1], detail: m[2] } : { head: status || "", detail: "" };
  }

  function statusTone(head) {
    const s = head.toLowerCase();
    if (s.startsWith("operational")) return "good";
    if (s.startsWith("in development") || s.includes("lost")) return "warn";
    return "";
  }

  // Last line of defence: internal annotations like "(TODO: verify ...)"
  // must never reach the screen, even if one slips back into the KB
  // (isro_matcher.py also warns about them at startup).
  const ANNOTATION_RE = /\s*\((?:todo|fixme|tbd|verify)\b[^)]*\)/gi;
  function clean(text) {
    return String(text || "").replace(ANNOTATION_RE, "").trim();
  }

  // Shown in the image frame when an entity has no photo yet (image: null)
  // or its file fails to load: an orbit emblem with the name and status, so
  // a missing photo reads as a design choice rather than a broken image.
  function noPhotoPanelHtml(m, st) {
    return `
      <div class="no-photo">
        <div class="np-emblem"><span class="np-orbit"></span><span class="np-body"></span><span class="np-sat"></span></div>
        <div class="np-name">${escapeHtml(clean(m.name))}</div>
        ${st.head ? `<div class="np-status">${escapeHtml(st.head)}</div>` : ""}
      </div>`;
  }

  function missionCardHtml(m) {
    const facts = (m.key_facts || []).slice(0, 3).map((f) => `<li>${escapeHtml(clean(f))}</li>`).join("");
    const st = splitStatus(clean(m.status));
    const conf = m.confidence != null ? `${Math.round(m.confidence * 100)}% · ${escapeHtml(m.matched_by || "")}` : "";
    const launchText = m.launch_date ? formatDate(m.launch_date) : clean(m.launch_note);
    const launch = launchText
      ? `<div class="meta-field"><div class="mf-label">Launch</div><div class="mf-value">${escapeHtml(launchText)}</div></div>`
      : "";
    const status = st.head
      ? `<div class="meta-field"><div class="mf-label">Status</div>
           <span class="status-pill ${statusTone(st.head)}">${escapeHtml(st.head)}</span>
           ${st.detail ? `<div class="status-detail">${escapeHtml(st.detail)}</div>` : ""}
         </div>`
      : "";
    let media;
    if (m.image) {
      const imgUrl = `/assets/isro/${encodeURIComponent(m.image)}`;
      media = `
          <div class="ambient" style="background-image:url('${imgUrl}')"></div>
          <div class="grid"></div>
          <img src="${imgUrl}" alt="${escapeHtml(m.name || "")}">
          <template class="no-photo-tpl">${noPhotoPanelHtml(m, st)}</template>`;
    } else {
      media = `<div class="grid"></div>${noPhotoPanelHtml(m, st)}`;
    }
    return `
      <div class="card glass">
        <div class="card-image">
          ${media}
          <span class="corner tl"></span><span class="corner tr"></span>
          <span class="corner bl"></span><span class="corner br"></span>
        </div>
        <div class="card-body">
          <div class="card-kicker">${escapeHtml(clean(m.org_unit) || "ISRO")}</div>
          <div class="card-name">${escapeHtml(clean(m.name))}</div>
          <div class="card-meta">${launch}${status}</div>
          <div class="card-oneline">${escapeHtml(clean(m.one_line))}</div>
          <ul class="card-facts">${facts}</ul>
          ${conf ? `<div class="card-confidence mono">match confidence: ${conf}</div>` : ""}
        </div>
      </div>
    `;
  }

  function noMatchCardHtml() {
    return `
      <div class="card glass no-match">
        <div class="nm-title">No matching mission found</div>
        <div class="nm-hint">Try asking about a mission by name &mdash; e.g. Chandrayaan-3 or Aditya-L1</div>
      </div>
    `;
  }

  function wireImageFallback() {
    const img = el.cardSlot.querySelector(".card-image img");
    if (!img) return;
    const swap = () => {
      const frame = img.closest(".card-image");
      const tpl = frame.querySelector(".no-photo-tpl");
      frame.querySelectorAll(".ambient, img, .no-photo-tpl").forEach((n) => n.remove());
      if (tpl) frame.insertAdjacentHTML("afterbegin", tpl.innerHTML);
    };
    img.addEventListener("error", swap);
    if (img.complete && img.naturalWidth === 0) swap(); // failed before the listener attached
  }

  // ---- metrics waterfall ------------------------------------------------------

  function renderWaterfall() {
    const { t1, t2, t3, t4, t5 } = currentLatency;
    el.wfBar.innerHTML = "";
    if (t1 == null || t2 == null) return;

    const segs = [];
    segs.push({ key: "t1t2", from: "T1", to: "T2", ms: t2 - t1 });

    const midEnd = t3 != null ? t3 : t2;
    if (t3 != null) segs.push({ key: "t2t3", from: "T2", to: "T3", ms: t3 - t2 });

    if (t4 != null) segs.push({ key: "t3t4", from: t3 != null ? "T3" : "T2", to: "T4", ms: t4 - midEnd });

    if (t5 != null && t4 != null) segs.push({ key: "t4t5", from: "T4", to: "T5", ms: t5 - t4 });

    segs.forEach((s) => {
      const div = document.createElement("div");
      div.className = "wf-seg";
      div.dataset.seg = s.key;
      div.style.flex = `${Math.max(s.ms, 1)} 0 0`;
      div.dataset.tip = `${s.from}→${s.to}: ${Math.round(s.ms)}ms`;
      el.wfBar.appendChild(div);
    });
  }

  // ===========================================================================
  // WebSocket client -- auto-reconnect with backoff
  // ===========================================================================

  let ws = null;
  let reconnectDelay = 1000;
  const RECONNECT_MAX = 8000;

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws/ui`;
  }

  function connect() {
    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      reconnectDelay = 1000;
      el.connBanner.classList.remove("show");
    };

    ws.onclose = () => {
      el.connBanner.classList.add("show");
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 1.7, RECONNECT_MAX);
    };

    ws.onerror = () => {
      try { ws.close(); } catch (e) { /* noop */ }
    };

    ws.onmessage = (evt) => {
      let msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (e) {
        return;
      }
      handleEvent(msg);
    };
  }

  function handleEvent(msg) {
    switch (msg.event) {
      case "edge_status":
        el.edgeDot.classList.toggle("connected", !!msg.connected);
        el.edgeText.textContent = msg.connected ? "connected" : "disconnected";
        break;
      case "level":
        if (appState === "idle" || appState === "wake") {
          const pct = Math.min(Math.max((msg.rms || 0) * 420, 0), 100);
          el.levelBar.style.width = pct + "%";
        }
        break;
      case "wake":
        toWake(msg.prob);
        break;
      case "partial":
        onPartial(msg.text || "");
        break;
      case "final_segment":
        onFinalSegment(msg.text || "");
        break;
      case "transcript":
        onTranscript(msg);
        break;
      case "content":
        onContent(msg);
        break;
      default:
        // unknown event types are ignored, not fatal -- forward compatible
        break;
    }
  }

  // Rehearsal/debug hook: drive the UI without an edge device, e.g. from the
  // devtools console: nakshatraUI.inject({event: "wake", prob: 0.93})
  window.nakshatraUI = { inject: handleEvent };

  connect();
  toIdle();
})();
