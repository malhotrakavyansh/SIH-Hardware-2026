"""
test_matcher.py -- regression check for isro_matcher.py against realistic
ASR mis-transcriptions.

Not a pytest suite -- a standalone script that prints what each query
resolves to and which layer (fuzzy / semantic / phonetic) produced it, so a
threshold change can be eyeballed against this fixed set before committing
to it.

Run:
    python test_matcher.py
"""

from __future__ import annotations

from isro_matcher import ISROMatcher

# (query, expected entity id or None for "should not match")
CASES = [
    ("tell me about chandrian three", "chandrayaan-3"),
    ("what is gagan yaan", "gaganyaan"),
    ("show me the moon mission", "chandrayaan-3"),
    ("tell me about mars orbiter", "mangalyaan"),
    ("when was p s l v launched", "pslv"),
    ("the human spaceflight programme", "gaganyaan"),
    ("chandra yan", "chandrayaan-3"),
    ("aditya el one", "aditya-l1"),
]

# Every entity asked for by its real name (kb "name" field). Must resolve
# identically with descriptors on or off.
_NAME_QUERIES = {e["id"]: f"tell me about {e['name']}" for e in
                 __import__("json").load(open("isro_kb.json", encoding="utf-8"))}
NAME_CASES = [(q, i) for i, q in _NAME_QUERIES.items()]

# Descriptive queries with no proper noun in them. "moon mission" fits
# Chandrayaan-1/2/3 and must resolve to the most recent (Chandrayaan-3).
DESCRIPTOR_CASES = [
    ("what was the lunar mission", "chandrayaan-3"),
    ("the south pole landing", "chandrayaan-3"),
    ("the mission for sending astronauts", "gaganyaan"),
    ("the crewed mission", "gaganyaan"),
    ("the mars mission", "mangalyaan"),
    ("the workhorse rocket", "pslv"),
    ("the satellite launch vehicle", "pslv"),
    ("the mission to venus", "shukrayaan"),
    ("the moon rover", "pragyan-rover"),
    ("a satellite that sees through clouds", "risat"),
    ("the communication satellite", "insat"),
    ("the launch site", "sdsc-sriharikota"),
]

# The mission number is the only thing separating three different missions.
# Live ASR emits number words ("chandrayaan three"), tests use both forms.
NUMBER_CASES = [
    ("tell me about chandrayaan one", "chandrayaan-1"),
    ("tell me about chandrayaan two", "chandrayaan-2"),
    ("tell me about chandrayaan three", "chandrayaan-3"),
    ("chandrayaan 2", "chandrayaan-2"),
    ("chandrayaan 3", "chandrayaan-3"),
    ("what happened to chandrayaan-2", "chandrayaan-2"),
    ("when did chandrayaan 1 launch", "chandrayaan-1"),
    ("gslv mark three", "gslv-mk3"),
]

# Proper noun AND a descriptor phrase in one query. The descriptor path runs
# first, so these check it does not take over a name the user actually said.
MIXED_CASES = [
    ("chandrayaan 1 was the first moon mission", "chandrayaan-1"),
    ("the moon mission chandrayaan 2", "chandrayaan-2"),
    ("gaganyaan the human spaceflight programme", "gaganyaan"),
    ("is pslv the workhorse rocket", "pslv"),
    ("pragyan the moon rover", "pragyan-rover"),
    ("vikram the moon lander", "vikram-lander"),
    ("mangalyaan the mars mission", "mangalyaan"),
    ("navic the indian gps", "navic"),
    # name and descriptor disagree: the name the user said wins
    ("mangalyaan moon mission", "mangalyaan"),
    ("pslv the moon mission", "pslv"),
    ("gaganyaan the mars mission", "gaganyaan"),
    ("tell me about the vikram lander mission to the moon", "vikram-lander"),
]


# Constructor flags that restore the original behaviour, one per improvement.
# Only flags that exist in isro_matcher.py are listed (grown step by step).
ORIGINAL: dict = {"generic_guard": False, "digit_aware": False}


def run(matcher, cases) -> list[tuple[str, str | None, str | None, str, str]]:
    rows = []
    for query, expected in cases:
        result = matcher.match(query)
        got = result["entity"]["id"] if result else None
        rows.append((query, expected, got, result["layer"] if result else "-",
                     f"{result['confidence']:.3f}" if result else "-"))
    return rows


def show(title, before, after) -> int:
    print(f"\n== {title} ==")
    print(f"{'query':<38} {'expected':<16} {'BEFORE':<16} {'AFTER':<16} {'layer':<10} {'conf':>6}")
    print("-" * 110)
    for b, a in zip(before, after):
        tag = "" if a[2] == a[1] else "  MISS"
        change = "" if a[2] == b[2] else "  (changed)"
        print(f"{a[0]:<38} {a[1] or '(none)':<16} {b[2] or '(none)':<16} {a[2] or '(none)':<16} "
              f"{a[3]:<10} {a[4]:>6}{tag}{change}")
    nb = sum(r[2] == r[1] for r in before)
    na = sum(r[2] == r[1] for r in after)
    print(f"BEFORE {nb}/{len(before)}   AFTER {na}/{len(after)}")
    return na


def main() -> None:
    # BEFORE = the original matcher: every improvement switched off.
    old = ISROMatcher(use_descriptors=False, **ORIGINAL)
    new = ISROMatcher()
    for title, cases in (("existing set", CASES), ("real names, all 20 entities", NAME_CASES),
                         ("mission numbers", NUMBER_CASES),
                         ("proper noun + descriptor in one query", MIXED_CASES),
                         ("descriptor queries", DESCRIPTOR_CASES)):
        show(title, run(old, cases), run(new, cases))


if __name__ == "__main__":
    main()
