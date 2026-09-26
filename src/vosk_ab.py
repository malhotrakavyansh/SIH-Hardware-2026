"""A/B two Vosk models on the ISRO proper nouns (TTS clips by default).

  python vosk_ab.py vosk-model-small-en-in-0.4 vosk-model-en-in-0.5
"""
import glob, json, os, sys, time, wave
import numpy as np, scipy.signal, vosk
sys.argv, models = sys.argv[:1], sys.argv[1:]
from cloud_server import ASR_VOCAB_GRAMMAR  # noqa: E402  (loads default model too; ignored)
vosk.SetLogLevel(-1)
ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = sorted(glob.glob(os.path.join(ROOT, "models/piper/asr_clips/*.wav")))

def load16(f):
    w = wave.open(f); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16); w.close()
    a = scipy.signal.resample_poly(a, 16000, sr).astype(np.int16)
    return np.concatenate([np.zeros(4000, np.int16), a, np.zeros(8000, np.int16)]).tobytes()

audio = {f: load16(f) for f in CLIPS}
allrows = []
for name in models:
    t0 = time.perf_counter(); m = vosk.Model(os.path.join(ROOT, "models/vosk", name)); load = time.perf_counter() - t0
    print(f"\n=== {name}  load {load:.2f}s")
    for mode, grammar in [("free", None), ("biased", ASR_VOCAB_GRAMMAR)]:
        hits = {}; lat = []
        for f, pcm in audio.items():
            word, spk, kind = os.path.basename(f)[:-4].split("__")
            r = vosk.KaldiRecognizer(m, 16000, grammar) if grammar else vosk.KaldiRecognizer(m, 16000)
            t = time.perf_counter()
            for i in range(0, len(pcm), 8000): r.AcceptWaveform(pcm[i:i+8000])  # 250 ms chunks
            text = json.loads(r.FinalResult())["text"]; lat.append((time.perf_counter() - t) * 1000)
            ok = word in text.replace(" ", "") if word == "pslv" else word in text.split()
            hits.setdefault(word, []).append((ok, kind, text))
            allrows.append(dict(model=name, mode=mode, word=word, spk=spk, kind=kind, text=text))
        print(f"-- {mode}: per-utterance latency mean {np.mean(lat):.0f} ms, p95 {np.percentile(lat,95):.0f} ms")
        for w, rows in hits.items():
            print(f"  {w:12s} {sum(o for o,_,_ in rows)}/{len(rows)}  e.g. {[t for o,_,t in rows if not o][:2]}")

json.dump(allrows, open(os.path.join(ROOT, "asr_partials.json"), "w"), indent=1)
