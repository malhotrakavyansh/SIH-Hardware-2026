"""Run the real Vosk partials from asr_partials.json through isro_matcher.

  python eval_partials.py [-v]
Outcomes per utterance: OK (right entity), WRONG (a different entity -- the
dangerous one on camera), NONE (no match), EMPTY (Vosk returned nothing, the
matcher is never invoked so it can't be helped here).
"""
import json, os, sys, collections
os.environ.setdefault("USE_TF", "0")
from isro_matcher import get_matcher
ROOT = os.path.dirname(os.path.abspath(__file__))
TRUTH = {"chandrayaan": "chandrayaan-", "gaganyaan": "gaganyaan", "mangalyaan": "mangalyaan",
         "aditya": "aditya-l1", "pslv": "pslv", "navic": "navic"}
m = get_matcher()
rows = json.load(open(os.path.join(ROOT, "asr_partials.json")))
cache = {}
tally = collections.Counter(); per = collections.defaultdict(collections.Counter); detail = {}
for r in rows:
    t = r["text"].strip()
    if not t: out = "EMPTY"; got = None
    else:
        if t not in cache: cache[t] = m.match(t)
        res = cache[t]; got = res["entity"]["id"] if res else None
        out = "NONE" if got is None else ("OK" if got.startswith(TRUTH[r["word"]]) else "WRONG")
    tally[out] += 1; per[r["word"]][out] += 1
    if t: detail[(r["word"], t)] = (out, got, cache[t]["layer"] if cache[t] else "-")
n = len(rows); nonempty = n - tally["EMPTY"]
print(f"\nTOTAL {n}: " + ", ".join(f"{k}={tally[k]}" for k in ("OK","WRONG","NONE","EMPTY")) +
      f"  | hit rate {tally['OK']}/{n} = {tally['OK']/n:.1%} (of non-empty: {tally['OK']/nonempty:.1%})")
for w in TRUTH: print(f"  {w:12s}", dict(per[w]))
if "-v" in sys.argv:
    for (w, t), (out, got, layer) in sorted(detail.items()):
        if out != "NONE" or "-vv" in sys.argv: print(f"  [{out:5s}] {w:12s} {t!r:40s} -> {got} ({layer})")
