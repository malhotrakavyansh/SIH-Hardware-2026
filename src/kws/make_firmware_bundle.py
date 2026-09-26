"""
make_firmware_bundle.py -- assemble the firmware handoff bundle for one
checkpoint (default nakshatra_v5) into <repo>/firmware_handoff/.

    model/            model_data.cc/.h, model_int8.tflite, quant_params.json
    features/         kws_mel_filterbank.h  (mel filterbank, DCT matrix, Hann window)
    test_vectors/     parity vectors: int16 input + exact MFCC + int8 features
                      + int8 model output, as C headers a C test asserts against
    parity/           parity_test.c, mfcc_reference.c (slow host-only cross-check),
                      kws_frontend_constants.h (generated from config)
    PROTOCOL.md       edge <-> server wire protocol (copied)
    MANIFEST.md, SHA256SUMS.txt, manifest.json

The bundle is rebuilt from scratch every run (its contents are deleted first), so
a stale file can never survive a rebuild. Everything numeric is generated from
config.py / features.py -- nothing is typed in by hand -- and the script asserts
the v5 facts it was written against before writing anything.

Run:
    python make_firmware_bundle.py [--run nakshatra_v5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import config
import dataset
import eval_streaming as ev
import export_c
import features
import make_test_vectors as mtv

REPO = config.ROOT.parent.parent
BUNDLE = REPO / "firmware_handoff"
PROTOCOL_MD = config.ROOT.parent / "PROTOCOL.md"

# Facts the bundle is asserted against (checked from the files, not trusted).
EXPECTED = {
    "run": "nakshatra_v5",
    "model_bytes": 44832,
    "in_scale": 0.448253, "in_zp": 67,
    "out_scale": 0.145795, "out_zp": 2,
    "contract_hash": "eaec172cd4689c14",
    "theta": 0.75, "k": 3, "window": 5, "refractory_ms": 1000,
}

PARITY_TOL = 0.05   # from the Step 2 spec (check_parity.py) -- do not loosen

# The go/no-go pair: two takes from the held-out session (config.HELDOUT_SESSION,
# never trained on or tuned on), pinned by file name so the bundle is reproducible.
# Chosen by rule, not cherry-picked, from per-take peak keyword probability under
# the v5 int8 model:
#   go   = the positive whose peak is closest to the median positive peak
#          (0.9948), among those with > 0.9 and >= K of the 5 hops ending at the
#          peak above theta -- a typical positive that fires, not the best one;
#   nogo = the hard negative with the highest peak that still stays below 0.5.
# `hop` is the hop index (of the take embedded in 1 s of silence either side, as
# eval_heldout.py does) whose window is the peak. build_gonogo() re-checks that
# each still fires / stays quiet, so a retrained model fails the build loudly.
GONOGO = {
    "go_positive_test_027": dict(sub="positive", file="positive_laptop_test_027.wav", hop=12, fire=True,
                                 why="held-out positive (session test): the model MUST fire on this"),
    "nogo_hardneg_test_001": dict(sub="hardneg", file="hardneg_laptop_test_001.wav", hop=7, fire=False,
                                  why="held-out hard negative (session test): the model must NOT fire on this"),
}
STREAM_WINDOWS = config.SMOOTH_WINDOW_HOPS   # windows in each go/no-go stream (5)


# =============================================================================
# Vectors
# =============================================================================

def _gsc_clip(word: str) -> np.ndarray:
    path = sorted((config.DATA_DIR / "gsc" / word).glob("*.wav"))[0]
    x, sr = sf.read(path, dtype="int16")
    assert sr == config.SAMPLE_RATE and x.ndim == 1, path
    out = np.zeros(config.CLIP_SAMPLES, dtype=np.int16)
    out[:min(len(x), config.CLIP_SAMPLES)] = x[:config.CLIP_SAMPLES]
    return out


def build_vectors() -> dict[str, tuple[str, np.ndarray]]:
    """name -> (what it exercises, int16 clip). Deterministic."""
    n = config.CLIP_SAMPLES
    t = np.arange(n) / config.SAMPLE_RATE
    rng = np.random.default_rng(config.SEED)

    chirp_phase = 2 * np.pi * (100.0 * t + 0.5 * (3800.0 - 100.0) * t ** 2)  # 1 s: 100 -> 3800 Hz
    square = np.where(np.sin(2 * np.pi * 200.0 * t) >= 0, 32767, -32768).astype(np.int16)
    impulse = np.zeros(n, dtype=np.int16)
    impulse[8000] = 20000

    vecs = {name: (why, clip) for name, why, clip in [
        ("silence", "all zeros: every mel bin is exactly the log floor, log(1e-6)", mtv._silence()),
        ("tone_440hz", "pure tone, amplitude 0.5: peaky spectrum, window/FFT scaling", mtv._tone()),
        ("white_noise", "uniform noise +-0.2 FS: flat spectrum, filterbank gains", mtv._white_noise()),
        ("chirp_100_3800hz", "linear sweep across the whole mel range: every filter is exercised",
         (0.5 * np.sin(chirp_phase) * 32767.0).astype(np.int16)),
        ("square_fullscale", "200 Hz square at +32767/-32768: clipping extremes, -32768 -> -1.0",
         square),
        ("impulse", "single 20000 spike at sample 8000: pre-emphasis + frame edges + floor",
         impulse),
        ("quiet_noise_8lsb", "uniform noise +-8 LSB: mel energies near LOG_MEL_FLOOR, float32 sensitivity",
         rng.integers(-8, 9, n).astype(np.int16)),
        ("speech_gsc_yes", "real speech (Google Speech Commands, 'yes'): what the model actually sees",
         _gsc_clip("yes")),
        ("speech_gsc_no", "real speech (Google Speech Commands, 'no')",
         _gsc_clip("no")),
    ]}
    return vecs


def build_gonogo(tflite_path: Path, q: dict) -> dict[str, dict]:
    """The go/no-go pair (see GONOGO). For each: the 1 s peak window, the
    STREAM_WINDOWS-window stream ending at it, Python's P(keyword) for every
    window, and whether the k-of-window detector fires. Asserts the pair still
    does what its name says."""
    out = {}
    for name, g in GONOGO.items():
        path = config.RECORDINGS_DIR / g["sub"] / g["file"]
        assert dataset.speaker_id(path) == config.HELDOUT_SESSION, f"{path} is not from the held-out session"
        raw, sr = sf.read(path, dtype="int16")
        assert sr == config.SAMPLE_RATE and raw.ndim == 1, path
        pad = np.zeros(config.CLIP_SAMPLES, dtype=np.int16)
        padded = np.concatenate([pad, raw, pad])                       # same embedding as eval_heldout.take_stream
        start = (g["hop"] - (STREAM_WINDOWS - 1)) * config.HOP_SAMPLES
        stream = padded[start:start + config.CLIP_SAMPLES + (STREAM_WINDOWS - 1) * config.HOP_SAMPLES]
        assert len(stream) == config.CLIP_SAMPLES + (STREAM_WINDOWS - 1) * config.HOP_SAMPLES, name
        probs = [run_model(tflite_path, features.extract_mfcc(stream[j * config.HOP_SAMPLES:
                                                                     j * config.HOP_SAMPLES + config.CLIP_SAMPLES]),
                           **q)[2] for j in range(STREAM_WINDOWS)]
        fired = len(ev.post_process(np.array(probs, dtype=np.float32), config.DETECT_THRESHOLD, config.DETECT_K,
                                    config.SMOOTH_WINDOW_HOPS, config.REFRACTORY_HOPS)) > 0
        last_over = probs[-1] > config.DETECT_THRESHOLD
        assert fired == g["fire"] and last_over == g["fire"], (
            f"{name}: expected fire={g['fire']} but detector fired={fired}, last-window P={probs[-1]:.4f} "
            f"(stream {[round(p, 3) for p in probs]}) -- the pinned selection no longer holds for this model")
        out[name] = dict(g, stream=stream, clip=stream[-config.CLIP_SAMPLES:], stream_prob=probs,
                         source=f"{g['sub']}/{g['file']}", n_over=sum(p > config.DETECT_THRESHOLD for p in probs))
    return out


def run_model(tflite_path: Path, feats: np.ndarray, in_scale: float, in_zp: int,
              out_scale: float, out_zp: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Same quantise-in / dequantise-out path as edge_agent.py."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    q = np.clip(np.round(feats[..., None] / in_scale + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q[None, ...])
    interp.invoke()
    out_q = interp.get_tensor(out["index"])[0].astype(np.int8)
    logits = (out_q.astype(np.float32) - out_zp) * out_scale
    e = np.exp(logits - logits.max())
    prob = e / e.sum()
    return q[..., 0], out_q, float(prob[config.KEYWORD_INDEX])


# =============================================================================
# C emitters
# =============================================================================

def _f(x: float) -> str:
    return f"{x:.8e}f"          # 9 significant digits: exact float32 round-trip


def c_array(ctype: str, name: str, values: np.ndarray, fmt, per_line: int) -> str:
    flat = [fmt(v) for v in values.ravel()]
    lines = [", ".join(flat[i:i + per_line]) for i in range(0, len(flat), per_line)]
    return f"static const {ctype} {name}[{len(flat)}] = {{\n    " + ",\n    ".join(lines) + "\n};"


def write_vector_header(path: Path, name: str, why: str, clip: np.ndarray, mfcc: np.ndarray,
                        q: np.ndarray, out_q: np.ndarray, kw_prob: float, gonogo: dict | None = None) -> None:
    ident = export_c.c_identifier(f"vec_{name}")
    stream_block: list[str] = []
    if gonogo is not None:
        stream_block = [
            "",
            f"/* Go/no-go: the {STREAM_WINDOWS}-window stream (hop {config.HOP_MS} ms) whose LAST window is the input above.",
            f" * Window j = stream[j*{config.HOP_SAMPLES} .. j*{config.HOP_SAMPLES}+{config.CLIP_SAMPLES}). Source: {gonogo['source']} */",
            f"#define {ident.upper()}_STREAM_LEN {len(gonogo['stream'])}",
            c_array("int16_t", f"{ident}_stream", gonogo["stream"], lambda v: str(int(v)), 16),
            f"/* Python P(keyword) for each of the {STREAM_WINDOWS} windows */",
            c_array("float", f"{ident}_stream_prob", np.array(gonogo["stream_prob"], dtype=np.float32), _f, 5),
            f"/* k-of-window detector over those {STREAM_WINDOWS} probabilities: 1 = must fire, 0 = must not */",
            f"#define {ident.upper()}_EXPECT_FIRE {int(gonogo['fire'])}",
        ]
    mfcc_rows = ",\n".join(
        "    { " + ", ".join(_f(float(v)) for v in row) + " }" for row in mfcc)
    text = "\n".join([
        export_c.format_banner("make_firmware_bundle.py"),
        f"/* {name}: {why} */",
        "#include <stdint.h>",
        "",
        c_array("int16_t", f"{ident}_input", clip, lambda v: str(int(v)), 16),
        "",
        f"/* Python features.extract_mfcc(input), float32 [{config.NUM_FRAMES}][{config.NUM_MFCC}] */",
        f"static const float {ident}_mfcc[{config.NUM_FRAMES}][{config.NUM_MFCC}] = {{\n{mfcc_rows}\n}};",
        "",
        "/* int8 model input: clip(round(mfcc / in_scale + in_zp), -128, 127), row-major [frame][coef] */",
        c_array("int8_t", f"{ident}_q", q, lambda v: str(int(v)), 20),
        "",
        "/* v5 model output (int8 logits) for that input, and the softmax keyword probability */",
        c_array("int8_t", f"{ident}_model_out_q", out_q, lambda v: str(int(v)), 3),
        f"#define {ident.upper()}_KEYWORD_PROB {_f(kw_prob)}",
        *stream_block,
        "",
    ])
    path.write_text(text, encoding="utf-8")


def write_index_header(path: Path, names: list[str], q: dict, gonogo: set[str]) -> None:
    ids = {n: export_c.c_identifier(f"vec_{n}") for n in names}
    includes = "\n".join(f'#include "vec_{n}.h"' for n in names)

    def row(n: str) -> str:
        tail = (f"{ids[n]}_stream, {ids[n]}_stream_prob, {ids[n].upper()}_EXPECT_FIRE" if n in gonogo
                else "0, 0, -1")
        return (f'    {{ "{n}", {ids[n]}_input, {ids[n]}_mfcc, {ids[n]}_q, {ids[n]}_model_out_q, '
                f'{ids[n].upper()}_KEYWORD_PROB, {tail} }}')

    rows = ",\n".join(row(n) for n in names)
    text = "\n".join([
        export_c.format_banner("make_firmware_bundle.py"),
        "#ifndef KWS_PARITY_VECTORS_H_",
        "#define KWS_PARITY_VECTORS_H_",
        "",
        "#include <stdint.h>",
        "",
        f'#define KWS_PARITY_CONTRACT_HASH  "{config.FEATURE_CONTRACT_HASH}"',
        f"#define KWS_PARITY_CLIP_SAMPLES   {config.CLIP_SAMPLES}",
        f"#define KWS_PARITY_NUM_FRAMES     {config.NUM_FRAMES}",
        f"#define KWS_PARITY_NUM_MFCC       {config.NUM_MFCC}",
        f"#define KWS_PARITY_MFCC_TOL       {PARITY_TOL}f   /* max abs diff, Step 2 spec */",
        f"#define KWS_PARITY_IN_SCALE       {_f(q['in_scale'])}",
        f"#define KWS_PARITY_IN_ZERO_POINT  {q['in_zp']}",
        f"#define KWS_PARITY_OUT_SCALE      {_f(q['out_scale'])}",
        f"#define KWS_PARITY_OUT_ZERO_POINT {q['out_zp']}",
        f"#define KWS_PARITY_NUM_VECTORS    {len(names)}",
        f"#define KWS_PARITY_HOP_SAMPLES    {config.HOP_SAMPLES}",
        f"#define KWS_PARITY_STREAM_WINDOWS {STREAM_WINDOWS}",
        f"#define KWS_PARITY_THETA          {config.DETECT_THRESHOLD}f",
        f"#define KWS_PARITY_K              {config.DETECT_K}",
        f"#define KWS_PARITY_WINDOW         {config.SMOOTH_WINDOW_HOPS}",
        f"#define KWS_PARITY_REFRACTORY_HOPS {config.REFRACTORY_HOPS}",
        "",
        includes,
        "",
        "typedef struct {",
        "    const char    *name;",
        "    const int16_t *input;                                  /* KWS_PARITY_CLIP_SAMPLES */",
        "    const float  (*mfcc)[KWS_PARITY_NUM_MFCC];             /* expected MFCC [frames][coefs] */",
        "    const int8_t  *q;                                      /* expected int8 model input, frames*coefs */",
        "    const int8_t  *model_out_q;                            /* expected int8 model output, 3 classes */",
        "    float          keyword_prob;                           /* expected softmax P(keyword) */",
        "    const int16_t *stream;                                 /* go/no-go only, else 0: STREAM_WINDOWS windows, last == input */",
        "    const float   *stream_prob;                            /* go/no-go only: expected P(keyword) per stream window */",
        "    int            expect_fire;                            /* 1 go (must fire), 0 no-go (must not), -1 parity-only vector */",
        "} kws_parity_vector_t;",
        "",
        "static const kws_parity_vector_t kws_parity_vectors[KWS_PARITY_NUM_VECTORS] = {",
        rows,
        "};",
        "",
        "#endif  /* KWS_PARITY_VECTORS_H_ */",
        "",
    ])
    path.write_text(text, encoding="utf-8")


FRONTEND_CONSTANTS_H = """\
{banner}
/* Front-end constants that features.py uses but kws_mel_filterbank.h does not
 * carry. Generated from config.py -- all of them are inside the contract hash. */
#ifndef KWS_FRONTEND_CONSTANTS_H_
#define KWS_FRONTEND_CONSTANTS_H_

#define KWS_PREEMPHASIS    {pre}f    /* y[0] = x[0]; y[n] = x[n] - PRE*x[n-1], over the WHOLE 1 s clip */
#define KWS_LOG_MEL_FLOOR  {floor}f  /* log(mel_energy + FLOOR), natural log */
#define KWS_INT16_SCALE    32768.0f  /* x = pcm / 32768 (so -32768 -> -1.0) */

#endif
"""

MFCC_REFERENCE_C = r'''/*
 * mfcc_reference.c -- SLOW, HOST-ONLY reference of features.extract_mfcc() in
 * float32, written the way an MCU port would do it (radix-2 FFT, float
 * accumulation). It exists to (1) prove the vectors and tables in this bundle
 * are consistent from C, and (2) give a known-good implementation to diff your
 * firmware against. Replace it with your own kws_extract_mfcc() on the device.
 *
 * Pipeline (identical to features.py):
 *   pcm/32768 -> pre-emphasis (whole clip) -> 49 frames of 640, hop 320
 *   -> symmetric Hann -> zero-pad to 1024 -> |FFT|^2 (bins 0..512, no 1/N)
 *   -> 40 mel filters -> log(x + 1e-6) -> 10x40 DCT-II matrix
 */
#include <math.h>
#include <stdint.h>

#include "kws_frontend_constants.h"
#include "kws_mel_filterbank.h"

#define PI_F 3.14159265358979323846f

static void fft_radix2(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {           /* bit-reversal */
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                float ang = -2.0f * PI_F * (float)k / (float)len;
                float wr = cosf(ang), wi = sinf(ang);
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
            }
        }
    }
}

void kws_extract_mfcc(const int16_t *pcm, float out[KWS_NUM_FRAMES][KWS_NUM_MFCC]) {
    static float x[KWS_NUM_FRAMES * KWS_STRIDE_LEN + KWS_FRAME_LEN];
    const int n_samples = (KWS_NUM_FRAMES - 1) * KWS_STRIDE_LEN + KWS_FRAME_LEN;   /* 16000 */

    x[0] = (float)pcm[0] / KWS_INT16_SCALE;
    for (int n = 1; n < n_samples; n++)
        x[n] = (float)pcm[n] / KWS_INT16_SCALE - KWS_PREEMPHASIS * ((float)pcm[n - 1] / KWS_INT16_SCALE);

    for (int f = 0; f < KWS_NUM_FRAMES; f++) {
        float re[KWS_NFFT], im[KWS_NFFT], power[KWS_NUM_FREQ_BINS], logmel[KWS_NUM_MEL];
        for (int i = 0; i < KWS_NFFT; i++) { re[i] = 0.0f; im[i] = 0.0f; }
        for (int i = 0; i < KWS_FRAME_LEN; i++) re[i] = x[f * KWS_STRIDE_LEN + i] * kws_hann_window[i];
        fft_radix2(re, im, KWS_NFFT);
        for (int k = 0; k < KWS_NUM_FREQ_BINS; k++) power[k] = re[k] * re[k] + im[k] * im[k];
        for (int m = 0; m < KWS_NUM_MEL; m++) {
            float e = 0.0f;
            for (int k = 0; k < KWS_NUM_FREQ_BINS; k++) e += kws_mel_filterbank[m][k] * power[k];
            logmel[m] = logf(e + KWS_LOG_MEL_FLOOR);
        }
        for (int c = 0; c < KWS_NUM_MFCC; c++) {
            float s = 0.0f;
            for (int m = 0; m < KWS_NUM_MEL; m++) s += kws_dct_matrix[c][m] * logmel[m];
            out[f][c] = s;
        }
    }
}
'''

PARITY_TEST_C = r'''/*
 * parity_test.c -- asserts kws_extract_mfcc() against the Python vectors.
 *
 *   host:  gcc -O2 -std=c99 -I features -I test_vectors -I parity \
 *              parity/parity_test.c parity/mfcc_reference.c -lm -o parity_test && ./parity_test
 *   yours: link your own kws_extract_mfcc() instead of mfcc_reference.c.
 *
 * FAIL if, for any vector, max |mfcc - expected| >= KWS_PARITY_MFCC_TOL, or any
 * int8 model-input value is off by more than 1. Exit code 0 = all pass.
 *
 * A correct float32 port lands around 1e-4..1e-3. The 0.05 gate is the spec's,
 * but it is loose: a periodic (N instead of N-1) Hann window, for one, only
 * reaches ~0.06 on some clips and slips under it on others. So any vector above
 * KWS_SUSPICIOUS is flagged WARN -- treat WARN as a bug to find, not a pass.
 */
#include <math.h>
#include <stdio.h>

#include "kws_mel_filterbank.h"
#include "kws_parity_vectors.h"

#define KWS_SUSPICIOUS 5e-3f

void kws_extract_mfcc(const int16_t *pcm, float out[KWS_NUM_FRAMES][KWS_NUM_MFCC]);

int main(void) {
    static float got[KWS_PARITY_NUM_FRAMES][KWS_PARITY_NUM_MFCC];
    int failed = 0, warned = 0;

    if (sizeof(KWS_PARITY_CONTRACT_HASH) != sizeof(KWS_FEATURE_CONTRACT_HASH) ||
        __builtin_strcmp(KWS_PARITY_CONTRACT_HASH, KWS_FEATURE_CONTRACT_HASH) != 0) {
        printf("FAIL contract hash: vectors %s vs tables %s\n", KWS_PARITY_CONTRACT_HASH, KWS_FEATURE_CONTRACT_HASH);
        return 1;
    }
    printf("contract hash %s\n\n%-20s %10s %9s %8s %7s\n", KWS_PARITY_CONTRACT_HASH,
           "vector", "max|diff|", "at f,c", "q!=exp", "q>1lsb");

    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        kws_extract_mfcc(vec->input, got);

        float maxd = 0.0f; int mf = 0, mc = 0, qdiff = 0, qbig = 0;
        for (int f = 0; f < KWS_PARITY_NUM_FRAMES; f++) {
            for (int c = 0; c < KWS_PARITY_NUM_MFCC; c++) {
                float d = fabsf(got[f][c] - vec->mfcc[f][c]);
                if (d > maxd) { maxd = d; mf = f; mc = c; }
                float qf = roundf(got[f][c] / KWS_PARITY_IN_SCALE + (float)KWS_PARITY_IN_ZERO_POINT);
                int qi = qf > 127.0f ? 127 : (qf < -128.0f ? -128 : (int)qf);
                int dq = qi - vec->q[f * KWS_PARITY_NUM_MFCC + c];
                if (dq != 0) qdiff++;
                if (dq > 1 || dq < -1) qbig++;
            }
        }
        int ok = (maxd < KWS_PARITY_MFCC_TOL) && qbig == 0 && isfinite(maxd);
        int warn = ok && maxd > KWS_SUSPICIOUS;
        printf("%-20s %10.2e %4d,%-4d %8d %7d  %s\n", vec->name, maxd, mf, mc, qdiff, qbig,
               !ok ? "FAIL" : (warn ? "WARN" : "PASS"));
        failed += !ok;
        warned += warn;
    }
    printf("\n%s (%d/%d vectors)%s\n", failed ? "PARITY FAILED" : "PARITY PASSED",
           KWS_PARITY_NUM_VECTORS - failed, KWS_PARITY_NUM_VECTORS,
           warned ? "  -- WARN: max diff above 5e-3 on some vectors, find the cause" : "");
    return failed != 0;
}
'''


GO_NOGO_TEST_C = r'''/*
 * go_nogo_test.c -- does the firmware actually WAKE? (parity_test.c only proves the
 * front end; a port can pass parity and still never fire.)
 *
 * You provide the whole inference path for one 1 s window:
 *     float kws_keyword_prob(const int16_t *window16000);
 *   = MFCC -> int8 quantise -> model -> dequantise -> softmax -> P(keyword).
 *
 * For each go/no-go vector this checks, using YOUR probabilities:
 *   1. P(keyword) on the peak window is within 0.03 of Python's, and on the right side of THETA
 *      (go: > THETA, no-go: < THETA);
 *   2. the same holds for each of the 5 stream windows (catches a model that only works on the peak);
 *   3. the k-of-window rule over those 5 probabilities (at least K above THETA) fires for go and
 *      not for no-go. Your real detector should be run over the stream too and agree.
 * Exit code 0 = GO. Anything else = do not flash.
 */
#include <math.h>
#include <stdio.h>

#include "kws_parity_vectors.h"

float kws_keyword_prob(const int16_t *window16000);

int main(void) {
    int failed = 0, seen = 0;
    printf("theta %.2f  k %d of %d\n\n%-22s %-6s %9s %9s %9s %6s  %s\n", KWS_PARITY_THETA, KWS_PARITY_K,
           KWS_PARITY_WINDOW, "vector", "want", "P(peak)", "P(python)", "max|dP|", "fires", "result");
    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        if (vec->expect_fire < 0) continue;
        seen++;
        float p = kws_keyword_prob(vec->input);
        float maxd = fabsf(p - vec->keyword_prob);
        int over = 0, side_ok = 1;
        for (int j = 0; j < KWS_PARITY_STREAM_WINDOWS; j++) {
            float pj = kws_keyword_prob(vec->stream + j * KWS_PARITY_HOP_SAMPLES);
            float d = fabsf(pj - vec->stream_prob[j]);
            if (!(d <= maxd)) maxd = d;                     /* also catches NaN */
            over += pj > KWS_PARITY_THETA;
        }
        if (vec->expect_fire ? !(p > KWS_PARITY_THETA) : !(p < KWS_PARITY_THETA)) side_ok = 0;
        int fires = over >= KWS_PARITY_K;
        int ok = side_ok && maxd < 0.03f && fires == vec->expect_fire;
        printf("%-22s %-6s %9.4f %9.4f %9.4f %6s  %s\n", vec->name, vec->expect_fire ? "FIRE" : "quiet", p,
               vec->keyword_prob, maxd, fires ? "yes" : "no", ok ? "PASS" : "FAIL");
        failed += !ok;
    }
    if (seen != 2) { printf("FAIL: expected exactly one go and one no-go vector, found %d\n", seen); return 1; }
    printf("\n%s\n", failed ? "NO-GO" : "GO");
    return failed != 0;
}
'''

GO_NOGO_STUB_C = r'''/*
 * go_nogo_stub.c -- HOST-ONLY. NOT A MODEL. It answers kws_keyword_prob() by looking the window up in
 * the vectors' own streams and returning Python's stored probability. Its only purpose is to prove
 * go_nogo_test.c itself compiles and reports GO when handed correct answers. Never link it into
 * firmware; link your own kws_keyword_prob() instead.
 */
#include <string.h>

#include "kws_parity_vectors.h"

float kws_keyword_prob(const int16_t *window16000) {
    for (int v = 0; v < KWS_PARITY_NUM_VECTORS; v++) {
        const kws_parity_vector_t *vec = &kws_parity_vectors[v];
        if (vec->expect_fire < 0) continue;
        for (int j = 0; j < KWS_PARITY_STREAM_WINDOWS; j++)
            if (memcmp(window16000, vec->stream + j * KWS_PARITY_HOP_SAMPLES, KWS_PARITY_CLIP_SAMPLES * sizeof(int16_t)) == 0)
                return vec->stream_prob[j];
    }
    return -1.0f;
}
'''


# =============================================================================
# Manifest
# =============================================================================

def _gcc_run(sources: list[str], label: str) -> str:
    """Compile and run part of the bundle with gcc, if there is one. The output
    goes into MANIFEST.md, so the recorded result is always from the files
    actually shipped."""
    gcc = shutil.which("gcc")
    if gcc is None:
        return f"(gcc not found when this bundle was built -- {label} was NOT run)"
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "t.exe"
        cmd = [gcc, "-O2", "-std=c99", "-Wall", "-Wextra", "-I", "features", "-I", "test_vectors", "-I", "parity",
               *sources, "-lm", "-o", str(exe)]
        built = subprocess.run(cmd, cwd=BUNDLE, capture_output=True, text=True)
        if built.returncode != 0:
            raise RuntimeError(f"bundle {label} failed to compile:\n" + built.stderr)
        ran = subprocess.run([str(exe)], capture_output=True, text=True)
        out = ran.stdout.rstrip()
        if ran.returncode != 0:
            raise RuntimeError(f"bundle {label} FAILED against its own vectors:\n" + out)
        return out


def run_c_parity() -> str:
    return _gcc_run(["parity/parity_test.c", "parity/mfcc_reference.c"], "parity_test")


def run_c_gonogo_selfcheck() -> str:
    return _gcc_run(["parity/go_nogo_test.c", "parity/go_nogo_stub.c"], "go_nogo_test (stub self-check)")


MANIFEST_MD = """\
# Nakshatra wake-word -- firmware handoff bundle ({run})

Everything here is generated by `src/kws/make_firmware_bundle.py` (rebuild, never
hand-edit). **If you copied files from an earlier handoff, delete them and use only
this folder** -- see "Discard" at the bottom.

## What this model is

| | |
|---|---|
| Checkpoint | `{run}` |
| Model | int8 TFLite, DS-CNN-S, **{model_bytes:,} bytes** (`g_model_len`) |
| Input | int8 `[1, {frames}, {mfcc}, 1]`, **scale {in_scale:.6f}, zero_point {in_zp}** |
| Output | int8 `[1, 3]` = `[silence, unknown, keyword]`, **scale {out_scale:.6f}, zero_point {out_zp}** |
| Feature contract hash | **`{hash}`** (`KWS_FEATURE_CONTRACT_HASH` in `features/kws_mel_filterbank.h`) |
| Audio | 16 kHz mono int16, 1 s window = {clip} samples, hop 100 ms |

**Model identity:** the contract hash and the model size are IDENTICAL for every
checkpoint we ever trained (v1, v2, v3, v5 share architecture and front end), so
neither can tell you whether you have the right model. The input/output scale and
zero-point above, and the sha256 in the table below, can.

## Detection (must match `edge_agent.py`)

Every 100 ms: take the latest 1 s of audio, compute MFCC, quantise, run the model,
dequantise and **softmax** the 3 logits. `p = softmax(...)[2]` (keyword).

    fire when at least K={k} of the last WINDOW={window} values of p exceed THETA={theta}
    (needs a full window), then ignore fires for REFRACTORY = {refractory_ms} ms ({refractory_hops} hops)

Threshold the softmax probability, not the raw int8 logit.
Held-out result at this operating point: TPR 89.8%, hard-negative TNR 96.7%, 0 false
triggers in a 300 s talking recording.

## Front-end porting notes (where silent drift comes from)

1. pcm -> float: `x = pcm / 32768.0` (so -32768 -> -1.0).
2. Pre-emphasis `y[0] = x[0]; y[n] = x[n] - {pre} * x[n-1]` over the **whole 1 s window,
   recomputed each hop**. Do not carry the previous hop's last sample into `y[0]`.
3. 49 frames of 640 samples, hop 320. **Symmetric Hann** `0.5 - 0.5*cos(2*pi*n/(N-1))` -- use
   the table in `kws_hann_window`, do not call a library window function.
4. Zero-pad to NFFT=1024, `|FFT|^2` for bins 0..512, **no 1/N scaling**.
5. `kws_mel_filterbank[40][513]` matrix product, then `ln(x + 1e-6)`.
6. `kws_dct_matrix[10][40]` matrix product -> 10 coefficients per frame.
7. Model input `q = clip(round(mfcc / in_scale + in_zp), -128, 127)`, layout `[frame][coef]`.
   (numpy rounds half-to-even, `roundf` half-away; ties are vanishingly rare and the
   test allows +-1 LSB.)

`parity/kws_frontend_constants.h` carries the constants that are not in the filterbank header.

## Parity test (do this before anything else)

    gcc -O2 -std=c99 -I features -I test_vectors -I parity \\
        parity/parity_test.c parity/mfcc_reference.c -lm -o parity_test && ./parity_test

`parity_test.c` calls `void kws_extract_mfcc(const int16_t *pcm, float out[49][10])`.
`mfcc_reference.c` is a slow host-only float32 implementation of it; on the device, link
YOUR `kws_extract_mfcc` instead and run the same test (include one `test_vectors/vec_*.h`
at a time if flash is tight -- each is self-contained).

Each `vec_<name>.h` holds: the int16 input, the exact float32 MFCC Python produced, the
int8 model input, the v5 model's int8 output and keyword probability. Gate: max |diff| < {tol}
per vector. **Read the printed max diff**: a correct float32 port is ~1e-4..1e-3; anything
above ~0.01 passes the gate but means a subtle mismatch that will cost accuracy.

Vectors: {vector_list}. The two speech clips `speech_gsc_*` come from Google Speech Commands
v0.02 (CC BY 4.0). Those 9 parity-only vectors are not the wake word (the model scores them all
~0): they prove the front end matches, not that the device wakes. The go/no-go pair below does.

## GO / NO-GO PAIR -- the vectors that decide whether the firmware wakes

Parity passing is necessary, not sufficient: a port can match the MFCC to 1e-4 and still never
fire. These two are real held-out recordings (session `test`, never trained on or tuned on):

| vector | source | want | P(keyword), peak window | int8 model out `[sil,unk,kw]` | Python stream P (5 windows) | k-of-5 detector |
|---|---|---|---|---|---|---|
| **`go_positive_test_027`** | `{go_source}` | **FIRE** | **{go_p:.4f}** (> {theta}) | {go_out} | {go_stream} | **fires** ({go_over} of 5 > {theta}) |
| **`nogo_hardneg_test_001`** | `{nogo_source}` | **stay quiet** | **{nogo_p:.4f}** (< {theta}) | {nogo_out} | {nogo_stream} | **does not fire** ({nogo_over} of 5 > {theta}) |

**Ship gate: both must pass.** With `kws_keyword_prob()` = your full MFCC -> int8 -> model -> softmax path:

    gcc -O2 -std=c99 -I features -I test_vectors -I parity \\
        parity/go_nogo_test.c <your kws_keyword_prob.c> -lm -o go_nogo_test && ./go_nogo_test

It prints GO or NO-GO. Then also run your real detector (k of window, refractory) over each vector's
`*_stream` (5 windows, 100 ms hop, last window = the vector's input): `EXPECT_FIRE` is 1 for go, 0 for
no-go. If go fails while parity passes, suspect the model path (input quantisation scale/zero-point,
dequantisation, using the wrong output index -- keyword is index 2 -- or thresholding a logit instead
of the softmax probability). If no-go fails, suspect a missing softmax or an inverted comparison.

Selection rule (so nobody thinks these were cherry-picked): go = the held-out positive whose peak
probability is closest to the median positive peak (0.9948) among those with peak > 0.9 and at least
K of the 5 hops ending at it above theta -- a typical positive, not the best one. no-go = the held-out
hard negative with the highest peak that stays below 0.5. `make_firmware_bundle.py` re-verifies both
against the model and fails the build if they stop holding. These are the maintainer's own voice.

`parity/go_nogo_stub.c` is a HOST-ONLY stand-in that returns Python's stored numbers by table lookup. It
exists only to prove `go_nogo_test.c` itself works (result below); **it is not a model and must never
be linked into firmware.**

### go_nogo_test.c self-check with the stub (harness only -- NOT evidence the model works on device)

```
{gonogo_out}
```

### Result when this bundle was built (`parity_test`, gcc, from these exact files)

```
{parity_out}
```

## Files

| file | sha256 (first 12) |
|---|---|
{file_table}

`SHA256SUMS.txt` and `manifest.json` carry the full hashes (and cover MANIFEST.md).

## Discard (from any earlier handoff)

Delete, do not merge:
- `model_data.cc`, `model_int8.tflite`, `quant_params.json` from `nakshatra_mvp_v1`, `nakshatra_v2_gsc` or `nakshatra_v3`
  (v1: input scale 0.367881 / zp 110, output 0.026062 / -20; v2: output 0.053518 / -48; v3: output 0.047999 / -43).
- any `kws_mel_filterbank.h` not carrying hash `{hash}`, and any hand-written mel/DCT/window tables.
- any older `*_input.h` / `*_mfcc_reference.*` parity files (3 synthetic clips only; superseded by `test_vectors/`).
- any operating point other than theta {theta}, k {k}, window {window}, refractory {refractory_ms} ms.
"""


def write_manifest(names: list[str], q: dict, model_len: int, run: str, gonogo: dict, meta: dict) -> None:
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file())
    table = "\n".join(f"| `{p.relative_to(BUNDLE).as_posix()}` | `{sha256(p)[:12]}` |" for p in files)
    go, nogo = gonogo["go_positive_test_027"], gonogo["nogo_hardneg_test_001"]
    fmt = lambda probs: "[" + ", ".join(f"{p:.3f}" for p in probs) + "]"
    (BUNDLE / "MANIFEST.md").write_text(MANIFEST_MD.format(
        go_source=go["source"], nogo_source=nogo["source"],
        go_p=go["stream_prob"][-1], nogo_p=nogo["stream_prob"][-1],
        go_out=meta["go_positive_test_027"]["model_out_q"], nogo_out=meta["nogo_hardneg_test_001"]["model_out_q"],
        go_stream=fmt(go["stream_prob"]), nogo_stream=fmt(nogo["stream_prob"]),
        go_over=go["n_over"], nogo_over=nogo["n_over"], gonogo_out=run_c_gonogo_selfcheck(),
        run=run, model_bytes=model_len, frames=config.NUM_FRAMES, mfcc=config.NUM_MFCC,
        in_scale=q["in_scale"], in_zp=q["in_zp"], out_scale=q["out_scale"], out_zp=q["out_zp"],
        hash=config.FEATURE_CONTRACT_HASH, clip=config.CLIP_SAMPLES, k=config.DETECT_K,
        window=config.SMOOTH_WINDOW_HOPS, theta=config.DETECT_THRESHOLD,
        refractory_ms=config.REFRACTORY_MS, refractory_hops=config.REFRACTORY_HOPS,
        pre=config.PREEMPHASIS, tol=PARITY_TOL, vector_list=", ".join(f"`{n}`" for n in names),
        parity_out=run_c_parity(), file_table=table), encoding="utf-8")


# =============================================================================
# Bundle
# =============================================================================

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=EXPECTED["run"])
    args = ap.parse_args()

    ck = config.CHECKPOINT_DIR / args.run
    quant = json.loads((ck / "quant_params.json").read_text(encoding="utf-8"))
    tflite = ck / "model_int8.tflite"
    q = {"in_scale": quant["input"]["scale"], "in_zp": quant["input"]["zero_point"],
         "out_scale": quant["output"]["scale"], "out_zp": quant["output"]["zero_point"]}

    # ---- assert the facts this bundle is supposed to carry --------------------
    cc_text = (ck / "model_data.cc").read_text(encoding="utf-8")
    model_len = int(cc_text.split("g_model_len =")[1].split(";")[0])
    checks = {
        "g_model_len == tflite size": model_len == tflite.stat().st_size,
        f"model bytes == {EXPECTED['model_bytes']}": model_len == EXPECTED["model_bytes"],
        "input scale/zp": (round(q["in_scale"], 6), q["in_zp"]) == (EXPECTED["in_scale"], EXPECTED["in_zp"]),
        "output scale/zp": (round(q["out_scale"], 6), q["out_zp"]) == (EXPECTED["out_scale"], EXPECTED["out_zp"]),
        "quant_params hash == config hash": quant["feature_contract_hash"] == config.FEATURE_CONTRACT_HASH,
        f"contract hash == {EXPECTED['contract_hash']}": config.FEATURE_CONTRACT_HASH == EXPECTED["contract_hash"],
        "theta/k/window/refractory in config": (config.DETECT_THRESHOLD, config.DETECT_K,
                                                config.SMOOTH_WINDOW_HOPS, config.REFRACTORY_MS)
                                               == (EXPECTED["theta"], EXPECTED["k"], EXPECTED["window"],
                                                   EXPECTED["refractory_ms"]),
    }
    for label, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {label}")
    assert all(checks.values()), "v5 facts do not match -- refusing to build the bundle"

    # ---- fresh folder ----------------------------------------------------------
    # Clear the CONTENTS, not the folder itself: on Windows the folder cannot be
    # removed while a shell/Explorer window has it open.
    BUNDLE.mkdir(exist_ok=True)
    for child in BUNDLE.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for sub in ("model", "features", "test_vectors", "parity"):
        (BUNDLE / sub).mkdir()

    # ---- model -----------------------------------------------------------------
    for name in ("model_data.cc", "model_data.h", "model_int8.tflite", "quant_params.json"):
        shutil.copy2(ck / name, BUNDLE / "model" / name)

    # ---- mel filterbank / DCT / Hann ------------------------------------------
    export_c.export_mel_filterbank()
    shutil.copy2(config.EXPORT_DIR / config.C_MEL_HEADER_NAME, BUNDLE / "features" / config.C_MEL_HEADER_NAME)

    # ---- protocol --------------------------------------------------------------
    shutil.copy2(PROTOCOL_MD, BUNDLE / "PROTOCOL.md")

    # ---- parity vectors ---------------------------------------------------------
    vecs = build_vectors()
    gonogo = build_gonogo(tflite, q)
    vecs.update({name: (g["why"], g["clip"]) for name, g in gonogo.items()})
    meta = {}
    for name, (why, clip) in vecs.items():
        mfcc = features.extract_mfcc(clip)
        qin, out_q, kw = run_model(tflite, mfcc, **q)
        write_vector_header(BUNDLE / "test_vectors" / f"vec_{name}.h", name, why, clip, mfcc, qin, out_q, kw,
                            gonogo.get(name))
        meta[name] = {"exercises": why, "input_sha256": hashlib.sha256(clip.tobytes()).hexdigest(),
                      "mfcc_min": float(mfcc.min()), "mfcc_max": float(mfcc.max()),
                      "model_out_q": out_q.tolist(), "keyword_prob": kw}
        print(f"vector {name:<18} mfcc [{mfcc.min():8.3f}, {mfcc.max():8.3f}]  out_q {out_q.tolist()}  P(kw)={kw:.4f}")
    write_index_header(BUNDLE / "test_vectors" / "kws_parity_vectors.h", list(vecs), q, set(gonogo))

    # ---- parity harness ---------------------------------------------------------
    (BUNDLE / "parity" / "kws_frontend_constants.h").write_text(FRONTEND_CONSTANTS_H.format(
        banner=export_c.format_banner("make_firmware_bundle.py"),
        pre=repr(float(config.PREEMPHASIS)), floor=f"{config.LOG_MEL_FLOOR:g}"), encoding="utf-8")
    (BUNDLE / "parity" / "mfcc_reference.c").write_text(MFCC_REFERENCE_C, encoding="utf-8")
    (BUNDLE / "parity" / "parity_test.c").write_text(PARITY_TEST_C, encoding="utf-8")
    (BUNDLE / "parity" / "go_nogo_test.c").write_text(GO_NOGO_TEST_C, encoding="utf-8")
    (BUNDLE / "parity" / "go_nogo_stub.c").write_text(GO_NOGO_STUB_C, encoding="utf-8")

    # ---- manifest ---------------------------------------------------------------
    write_manifest(list(vecs), q, model_len, args.run, gonogo, meta)
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file())
    sums = {p.relative_to(BUNDLE).as_posix(): sha256(p) for p in files}
    # newline="\n": write_text would emit CRLF on Windows and `sha256sum -c` then
    # reads the \r as part of every filename.
    with open(BUNDLE / "SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(f"{h}  {name}\n" for name, h in sums.items()))
    (BUNDLE / "manifest.json").write_text(json.dumps({
        "run": args.run, "feature_contract_hash": config.FEATURE_CONTRACT_HASH,
        "model_bytes": model_len,
        "input": {"scale": q["in_scale"], "zero_point": q["in_zp"]},
        "output": {"scale": q["out_scale"], "zero_point": q["out_zp"]},
        "detection": {"theta": config.DETECT_THRESHOLD, "k": config.DETECT_K,
                      "window_hops": config.SMOOTH_WINDOW_HOPS, "hop_ms": config.HOP_MS,
                      "refractory_ms": config.REFRACTORY_MS},
        "parity_mfcc_tolerance": PARITY_TOL, "vectors": meta, "sha256": sums}, indent=1), encoding="utf-8")
    print(f"\nbundle written to {BUNDLE}  ({len(files)} files + manifest.json)")


if __name__ == "__main__":
    main()
