"""
isro_matcher.py -- matches a spoken transcript to an entity in isro_kb.json.

Layer order: exact descriptor phrase (2+ words) -> fuzzy (with a generic-word
guard) -> semantic -> phonetic. Fuzzy and semantic are combined by taking
whichever is more confident:

  Layer 1 (primary, always available): rapidfuzz token-set fuzzy matching of
  the transcript against each entity's name + aliases. Cheap, deterministic,
  no model download -- this must always work.

  Layer 2 (enhancement): sentence-transformers (all-MiniLM-L6-v2) embedding
  similarity between the transcript and a precomputed embedding of each
  entity's name + one_line + aliases. Catches paraphrases that don't share
  a fuzzy-matchable substring, e.g. "tell me about the moon mission" ->
  Chandrayaan. If the model can't be loaded (missing package, no internet
  on first run, etc.) this layer is disabled and the matcher silently runs
  on Layer 1 alone -- the server must not fail to start because of it.

  Layer 3 (last resort): metaphone overlap for garbled proper nouns.

  Descriptor path (runs FIRST): each entity's "descriptors" are natural
  spoken phrases ("moon mission", "sending astronauts") for people who
  describe an entity instead of naming it -- Vosk transcribes common English
  reliably but fails on Indian proper nouns. An exact multi-word phrase on
  token boundaries is stronger evidence than a fuzzy-65 hit, and a proper-noun
  query contains no descriptor phrase, so it does not take over name matches.
  Single-word descriptors are ignored (too weak to lead). The one exception
  to "descriptors first": if the query contains an entity's actual proper name
  ("mangalyaan moon mission") the descriptor path stands down and the
  proper-noun layers decide -- see _names_a_proper_noun. See
  _resolve_descriptor_collision for how a phrase that fits several entities
  is resolved.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

logger = logging.getLogger("isro_matcher")

KB_PATH = Path(__file__).resolve().parent / "isro_kb.json"

# Thresholds loosened from the original 80.0 / 0.55 -- ASR output is noisy
# (small Vosk model, unfamiliar proper nouns), so a garbled but semantically
# close transcript ("chandrian three") should still clear the bar rather
# than falling through to "no match". See test_matcher.py for the
# regression set these were tuned against.
FUZZY_THRESHOLD = 65.0       # rapidfuzz token_set_ratio, 0-100
EMBED_THRESHOLD = 0.40       # cosine similarity, 0-1

# A query token names an entity when it equals one of the entity's proper-name
# tokens, or (both >= 5 letters, so short words like "moon"/"mom" can't
# collide) is this similar to one -- tolerant of ASR garbling like
# "chandrian" for "chandrayaan".
PROPER_NAME_MIN_SIMILARITY = 75.0

# Reported for a descriptor hit (an exact 2+ word phrase match).
DESCRIPTOR_CONFIDENCE = 0.6

# A descriptor must be at least this many words to be checked ahead of fuzzy.
DESCRIPTOR_MIN_WORDS = 2

# Fields the UI renders. Internal notes belong in "_notes" (never rendered);
# an annotation like "(TODO: verify ...)" in any of these would end up on
# screen, so the matcher flags it at startup.
_VISIBLE_FIELDS = ("name", "one_line", "key_facts", "status", "org_unit", "launch_note")
_ANNOTATION_RE = re.compile(r"\b(todo|fixme|tbd|verify|placeholder)\b", re.IGNORECASE)


def _load_jellyfish():
    """Phonetic-match dependency, installed on first run if missing (same
    pattern as the other src/*.py entry points) -- failure here must not
    prevent the matcher from working, it only disables the phonetic layer."""
    try:
        import jellyfish
        return jellyfish
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "jellyfish"])
            import jellyfish
            return jellyfish
        except Exception as exc:  # noqa: BLE001
            logger.warning("jellyfish unavailable (%s) -- phonetic fallback layer disabled", exc)
            return None


_jellyfish = _load_jellyfish()
_WORD_RE = re.compile(r"[a-z]+")

# Function words and query-framing words. They carry no entity information but
# do real damage on garbled ASR output: token_set_ratio scores 100 for any
# transcript whose tokens are a subset of a candidate's ("and" vs "ISRO
# Telemetry Tracking and Command Network"), and metaphone collides common
# words with KB words ("mean" and "moon" are both MN). Matching runs on the
# transcript with these removed; a transcript that is only these words
# matches nothing.
_STOPWORDS = frozenset("""
a an the and or but of to in on at for with from by as into up down out over
is are was were be been am do does did doesn't doesnt don't dont didn't didnt
can could would will shall should may might have has had
i me my you your we us he she it its they them their this that these those
there here what which who whom when where why how
tell show give about above please let know want like some any mean means
just so than then now not no yes though
""".split())


def _content_text(text: str, digits: bool = False) -> str:
    """Transcript with stopwords removed ("" if nothing meaningful is left).

    digits=True keeps digits as tokens and rewrites number words to digits
    (see _number_tokens) -- the mission number is the only thing separating
    Chandrayaan-1/2/3. digits=False is the original behaviour, which
    silently dropped every digit ("chandrayaan-3" -> "chandrayaan")."""
    if digits:
        return " ".join(w for w in _number_tokens(text) if w not in _STOPWORDS)
    return " ".join(w for w in re.findall(r"[a-z']+", text.lower()) if w not in _STOPWORDS)


# Live ASR emits number words ("chandrayaan three"); the KB and typed text use
# digits. Both sides are rewritten to digits before comparing.
_NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                 "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def _number_tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, number words rewritten to digits.
    Hyphens split ("Chandrayaan-3" -> chandrayaan, 3); "lvm3" stays one token."""
    return [_NUMBER_WORDS.get(w, w) for w in re.findall(r"[a-z0-9']+", text.lower())]


def _mission_numbers(text: str) -> set[str]:
    """Standalone 1-2 digit numbers in `text` (after number-word rewriting).
    Years (2008) are not mission numbers and are ignored."""
    return {t for t in _number_tokens(text) if t.isdigit() and len(t) <= 2}

# Words that appear across the KB as generic descriptors, not identifiers
# ("Mars mission", "moon lander", ...). Too many entities share them, and
# common speech words collide with them phonetically, so they are not phonetic
# anchors.
_GENERIC_KB_WORDS = frozenset("""
mission moon lunar first probe india indian satellite launch vehicle polar
mars orbiter regional navigation constellation system series mapping human
spaceflight program programme astronaut workhorse rocket landing station
space centre center telemetry tracking command network imaging radar national
synthetic aperture sun solar venus lander rover series
one two three four five mark
""".split())

# Words that may not carry a fuzzy match on their own. Nearly every KB entry
# shares some of them ("Mars Orbiter Mission", "Radar Imaging Satellite"), so
# token_set_ratio happily scores 0.78 for a transcript whose only overlap is
# "mission". A fuzzy hit must be carried by at least one token that is NOT in
# this set -- same principle as the phonetic anchors above, but a narrower
# list: descriptive words like "moon" or "mars" still count as evidence here
# (the fuzzy layer has always been how "mars orbiter" resolves).
_FUZZY_GENERIC_WORDS = frozenset("""
mission missions satellite satellites vehicle vehicles launch launches launcher
rocket rockets space system systems programme program orbiter lander
india indian isro
""".split())

# A query token counts as matching a candidate token when they are this
# similar (rapidfuzz ratio, 0-100) or one contains the other (len >= 4).
FUZZY_CARRIER_MIN_SIMILARITY = 70.0

# A phonetic hit also needs the words to *look* alike. Metaphone alone maps
# unrelated words together; "chandrian"/"chandrayaan" is ~75 on this scale.
PHONETIC_MIN_SIMILARITY = 60.0

# ASR frequently spells out acronyms letter-by-letter when it doesn't
# recognize them as a word ("p s l v" for "PSLV") -- collapse any run of 2+
# single-letter words into one token before matching, so "when was p s l v
# launched" has a fighting chance against the "pslv" alias.
_SPELLED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\s+){1,}[a-z]\b")


def _collapse_spelled_acronyms(text: str) -> str:
    return _SPELLED_ACRONYM_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)


def _fuzzy_carrier_tokens(text: str) -> list[str]:
    """Tokens of `text` that are allowed to carry a fuzzy match: not generic,
    not a number, at least 3 letters. Possessives are stripped (India's)."""
    out = []
    for w in re.findall(r"[a-z']+", text.lower()):
        w = re.sub(r"'s?$", "", w)
        if len(w) >= 3 and w not in _FUZZY_GENERIC_WORDS and w not in _STOPWORDS:
            out.append(w)
    return out


def _carried_by_discriminative_token(query: str, candidate: str) -> bool:
    cand = _fuzzy_carrier_tokens(candidate)
    for q in _fuzzy_carrier_tokens(query):
        for c in cand:
            if fuzz.ratio(q, c) >= FUZZY_CARRIER_MIN_SIMILARITY:
                return True
            if len(q) >= 4 and len(c) >= 4 and (q in c or c in q):
                return True
    return False


def _proper_candidates(text: str) -> list[str]:
    """Tokens of `text` that could be (part of) a proper name: not generic
    (either generic list), not a stopword, not a number, 3+ letters."""
    out = []
    for w in re.findall(r"[a-z']+", text.lower()):
        w = re.sub(r"'s?$", "", w)
        if (len(w) >= 3 and w not in _STOPWORDS and w not in _GENERIC_KB_WORDS
                and w not in _FUZZY_GENERIC_WORDS):
            out.append(w)
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _resolve_descriptor_collision(entities: list[dict], candidate_idx: list[int]) -> int:
    """Pick one entity when a descriptor fits several (e.g. "moon mission"
    fits Chandrayaan-1, -2 and -3). Explicit rule, in order:

      1. the most recently launched entity (ISO launch_date, so string order
         is date order) -- Chandrayaan-3 for "moon mission";
      2. an entity with no launch_date (planned / organisation) only wins if
         none of the candidates has one;
      3. remaining ties go to the earlier entry in isro_kb.json.

    Returns an index into `entities`. Never returns "ambiguous".
    """
    dated = [i for i in candidate_idx if entities[i].get("launch_date")]
    if dated:
        latest = max(entities[i]["launch_date"] for i in dated)
        return min(i for i in dated if entities[i]["launch_date"] == latest)
    return min(candidate_idx)


class ISROMatcher:
    def __init__(self, kb_path: Path = KB_PATH, use_descriptors: bool = True,
                 generic_guard: bool = True, digit_aware: bool = True):
        self.use_descriptors = use_descriptors
        self.generic_guard = generic_guard
        self.digit_aware = digit_aware
        with open(kb_path, "r", encoding="utf-8") as f:
            self.entities: list[dict] = json.load(f)
        self._warn_on_annotations()

        # Flattened list of (entity_index, candidate_string) for fuzzy match
        # over every name/alias individually, so a strong alias match isn't
        # diluted by averaging against the entity's other aliases.
        self._fuzzy_candidates: list[tuple[int, str]] = []
        for i, ent in enumerate(self.entities):
            self._fuzzy_candidates.append((i, ent["name"]))
            for alias in ent.get("aliases", []):
                self._fuzzy_candidates.append((i, alias))
        if self.digit_aware:
            # Same normalisation as the query side, so "Chandrayaan-3",
            # "Chandrayaan 3" and "Chandrayaan three" are the same string.
            self._fuzzy_candidates = [(i, " ".join(_number_tokens(s)))
                                      for i, s in self._fuzzy_candidates]

        # Proper-name tokens: what an entity is actually CALLED, as opposed to
        # what it is described as. Name/alias words minus every generic word
        # ("moon", "mission", "satellite", ...), stopwords, numbers and
        # anything under 3 letters -> chandrayaan, gaganyaan, pslv, vikram...
        self._proper_tokens: dict[str, set[int]] = {}
        for i, ent in enumerate(self.entities):
            for text in [ent["name"], *ent.get("aliases", [])]:
                for tok in _proper_candidates(text):
                    self._proper_tokens.setdefault(tok, set()).add(i)

        # Mission numbers each entity is known by (name + aliases).
        self._entity_numbers: list[set[str]] = [
            _mission_numbers(" ".join([ent["name"], *ent.get("aliases", [])]))
            for ent in self.entities]

        # Phonetic index: metaphone code -> owning entity indices, built from
        # each entity's name/alias words (skipping short words like "the",
        # "of" -- too many collisions to be useful as phonetic anchors).
        self._phonetic_index: dict[str, list[tuple[int, str]]] = {}
        if _jellyfish is not None:
            for i, ent in enumerate(self.entities):
                for text in [ent["name"], *ent.get("aliases", [])]:
                    for tok in _WORD_RE.findall(text.lower()):
                        if len(tok) < 4 or tok in _GENERIC_KB_WORDS:
                            continue
                        code = _jellyfish.metaphone(tok)
                        self._phonetic_index.setdefault(code, []).append((i, tok))

        # descriptor phrase (token tuple) -> entity indices that list it.
        self._descriptor_index: dict[tuple[str, ...], list[int]] = {}
        for i, ent in enumerate(self.entities):
            for phrase in ent.get("descriptors", []):
                toks = tuple(_tokens(phrase))
                if len(toks) < DESCRIPTOR_MIN_WORDS:
                    if toks:
                        logger.info("descriptor %r on %s ignored: fewer than %d words",
                                    phrase, ent.get("id"), DESCRIPTOR_MIN_WORDS)
                    continue
                if toks:
                    self._descriptor_index.setdefault(toks, []).append(i)
        for toks, idx in self._descriptor_index.items():
            if len(idx) > 1:
                winner = _resolve_descriptor_collision(self.entities, idx)
                logger.info("descriptor %r fits %s -> resolves to %s", " ".join(toks),
                            [self.entities[i]["id"] for i in idx], self.entities[winner]["id"])

        self._embedder = None
        self._entity_embeddings = None
        self._load_embedder()

    def _warn_on_annotations(self) -> None:
        for ent in self.entities:
            for field in _VISIBLE_FIELDS:
                value = ent.get(field)
                for text in (value if isinstance(value, list) else [value]):
                    if isinstance(text, str) and _ANNOTATION_RE.search(text):
                        logger.warning("isro_kb.json: internal annotation in user-visible "
                                       "%s.%s: %r -- move it to _notes", ent.get("id"), field, text)

    # -- Layer 2 setup ----------------------------------------------------

    def _load_embedder(self) -> None:
        # transformers imports TensorFlow too when it's installed (it is --
        # the KWS training uses it), which took ~96 s at server startup.
        # This layer is PyTorch-only, so keep TF out of the import.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
        except ImportError as exc:
            logger.warning("sentence-transformers not available (%s); "
                            "falling back to fuzzy-only matching", exc)
            return

        try:
            # Cached copy first: the default load checks the Hugging Face hub
            # for updates (~15 s, and it stalls offline). Only go online if
            # the model has never been downloaded.
            try:
                model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            except Exception:  # noqa: BLE001 -- not cached yet
                model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # noqa: BLE001 -- any load failure is non-fatal
            logger.warning("failed to load all-MiniLM-L6-v2 (%s); "
                            "falling back to fuzzy-only matching", exc)
            return

        texts = [self._entity_text(ent) for ent in self.entities]
        try:
            embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to precompute entity embeddings (%s); "
                            "falling back to fuzzy-only matching", exc)
            return

        # Warm-up: the first encode() in a process pays PyTorch's one-time
        # kernel/thread initialisation (~2 s inside cloud_server) -- without
        # this, the first question of every demo session sat on that before
        # its card appeared. Warm queries take ~60-90 ms.
        try:
            model.encode("warm up", convert_to_tensor=True, show_progress_bar=False)
        except Exception:  # noqa: BLE001 -- warm-up is best effort
            pass

        self._embedder = model
        self._entity_embeddings = embeddings
        self._st_util = st_util
        logger.info("Layer 2 (sentence-transformers) matching enabled: %d entities embedded", len(texts))

    @staticmethod
    def _entity_text(entity: dict) -> str:
        parts = [entity["name"], entity.get("one_line", ""), *entity.get("aliases", [])]
        return ". ".join(p for p in parts if p)

    # -- matching -----------------------------------------------------------

    def _number_ok(self, query_numbers: set[str], entity_idx: int) -> bool:
        """A query carrying a mission number must not match an entity that is
        known by different numbers. Entities with no number are unaffected."""
        if not self.digit_aware or not query_numbers:
            return True
        ent_nums = self._entity_numbers[entity_idx]
        return not ent_nums or bool(query_numbers & ent_nums)

    def match(self, transcript: str) -> dict | None:
        """Returns a match dict {entity, confidence, layer} or None.

        Layer 1 (fuzzy) and Layer 2 (semantic) run first; whichever is more
        confident wins. Only if BOTH of those come up empty does the
        phonetic layer get a chance -- it's the coarsest of the three (a
        metaphone collision says much less than a fuzzy or semantic match),
        so it's a last resort, not a peer to average against.
        """
        transcript = re.sub(r"\[unk\]", " ", transcript or "").strip()
        if not transcript:
            return None

        collapsed = _collapse_spelled_acronyms(transcript)
        variants = [transcript] if collapsed == transcript else [transcript, collapsed]

        # Nothing but filler ("about", "and", "tell me") -> nothing to match.
        digits = self.digit_aware
        qnums = _mission_numbers(collapsed) if digits else set()
        if not _content_text(collapsed, digits):
            logger.info("match %r -> no match (no content words)", transcript)
            return None

        # Exact descriptor phrase first: stronger evidence than any fuzzy hit --
        # unless the query actually names an entity, in which case the name wins.
        named = self._names_a_proper_noun(collapsed)
        if named:
            logger.info("match %r names %s -- descriptor path skipped", transcript,
                        sorted(self.entities[i]["id"] for i in named))
        if self.use_descriptors and not named:
            descriptor_result = self._match_descriptor(collapsed, qnums)
            if descriptor_result:
                logger.info("match %r -> %s (descriptor, confidence=%.3f)",
                            transcript, descriptor_result["entity"]["name"],
                            descriptor_result["confidence"])
                return descriptor_result

        candidates = []
        for variant in variants:
            candidates.append(self._match_fuzzy(_content_text(variant, digits), qnums))
            candidates.append(self._match_embed(variant, qnums))
        candidates = [r for r in candidates if r is not None]
        if candidates:
            result = max(candidates, key=lambda r: r["confidence"])
            logger.info("match %r -> %s (%s, confidence=%.3f)",
                        transcript, result["entity"]["name"], result["layer"], result["confidence"])
            return result

        phonetic_result = self._match_phonetic(_content_text(collapsed), qnums)
        if phonetic_result:
            logger.info("match %r -> %s (phonetic, confidence=%.3f)",
                        transcript, phonetic_result["entity"]["name"], phonetic_result["confidence"])
            return phonetic_result

        logger.info("match %r -> no match (fuzzy/semantic/phonetic/descriptor all missed)", transcript)
        return None

    def _names_a_proper_noun(self, transcript: str) -> set[int]:
        """Entities whose proper name appears in the transcript (see
        _proper_tokens). Empty for a purely descriptive query. A presence
        check on the query, not a confidence comparison."""
        found: set[int] = set()
        for q in _proper_candidates(transcript):
            for p, owners in self._proper_tokens.items():
                if q == p or (len(q) >= 5 and len(p) >= 5 and (
                        fuzz.ratio(q, p) >= PROPER_NAME_MIN_SIMILARITY or q in p or p in q)):
                    found |= owners
        return found

    def _match_descriptor(self, transcript: str, qnums: set[str] = frozenset()) -> dict | None:
        """Whole-phrase containment of a descriptor in the transcript (token
        boundaries, so "sun mission" doesn't match inside "sunday mission").
        The longest matching phrase wins -- it is the most specific -- and if
        several entities list that same phrase, _resolve_descriptor_collision
        picks one -- after dropping entities whose mission number contradicts
        a number in the query ("moon mission 2" -> Chandrayaan-2)."""
        toks = _tokens(transcript)
        best: tuple[str, ...] | None = None
        for phrase in self._descriptor_index:
            n = len(phrase)
            if best is not None and n <= len(best):
                continue
            if any(tuple(toks[j:j + n]) == phrase for j in range(len(toks) - n + 1)):
                best = phrase
        if best is None:
            return None
        allowed = [i for i in self._descriptor_index[best] if self._number_ok(qnums, i)]
        if not allowed:
            return None
        idx = _resolve_descriptor_collision(self.entities, allowed)
        return {"entity": self.entities[idx], "confidence": DESCRIPTOR_CONFIDENCE,
                "layer": "descriptor"}

    def _match_fuzzy(self, transcript: str, qnums: set[str] = frozenset()) -> dict | None:
        if not transcript or not self._fuzzy_candidates:
            return None
        strings = [c[1] for c in self._fuzzy_candidates]
        # Best-scoring candidate first; with the guard on, skip candidates whose
        # overlap with the transcript is only generic words and take the next.
        scored = process.extract(transcript, strings, scorer=fuzz.token_set_ratio,
                                 score_cutoff=FUZZY_THRESHOLD, limit=None)
        for matched_string, score, idx in scored:
            if self.generic_guard and not _carried_by_discriminative_token(transcript, matched_string):
                continue
            if not self._number_ok(qnums, self._fuzzy_candidates[idx][0]):
                continue
            break
        else:
            return None
        entity_idx = self._fuzzy_candidates[idx][0]
        return {
            "entity": self.entities[entity_idx],
            "confidence": score / 100.0,
            "layer": "fuzzy",
        }

    def _match_embed(self, transcript: str, qnums: set[str] = frozenset()) -> dict | None:
        if self._embedder is None:
            return None
        query_emb = self._embedder.encode(transcript, convert_to_tensor=True, show_progress_bar=False)
        sims = self._st_util.cos_sim(query_emb, self._entity_embeddings)[0]
        for i in range(len(self.entities)):
            if not self._number_ok(qnums, i):
                sims[i] = -1.0
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])
        if best_score < EMBED_THRESHOLD:
            return None
        return {
            "entity": self.entities[best_idx],
            "confidence": best_score,
            "layer": "semantic",
        }

    def _match_phonetic(self, transcript: str, qnums: set[str] = frozenset()) -> dict | None:
        """Metaphone-code overlap between transcript words and entity
        name/alias words -- catches ASR garbling that shares no substring
        with the real word (e.g. "chandrian" / "chandrayan" -> Chandrayaan)
        but still sounds like it. Confidence is deliberately capped well
        below what fuzzy/semantic can report -- a phonetic hit is a much
        weaker signal than either of those."""
        if _jellyfish is None or not self._phonetic_index:
            return None

        hits: dict[int, int] = {}
        for tok in _WORD_RE.findall(transcript.lower()):
            if len(tok) < 4:
                continue
            code = _jellyfish.metaphone(tok)
            for entity_idx, kb_tok in self._phonetic_index.get(code, []):
                if fuzz.ratio(tok, kb_tok) >= PHONETIC_MIN_SIMILARITY                         and self._number_ok(qnums, entity_idx):
                    hits[entity_idx] = hits.get(entity_idx, 0) + 1

        if not hits:
            return None

        best_idx = max(hits, key=lambda i: hits[i])
        # 1 matched word -> 0.5, 2 -> 0.65, 3+ -> capped at 0.8. Always below
        # FUZZY_THRESHOLD/100 and EMBED_THRESHOLD so a real fuzzy/semantic
        # hit always outranks a phonetic one if both somehow fire.
        confidence = min(0.5 + 0.15 * (hits[best_idx] - 1), 0.8)
        return {
            "entity": self.entities[best_idx],
            "confidence": confidence,
            "layer": "phonetic",
        }


_matcher: ISROMatcher | None = None


def get_matcher() -> ISROMatcher:
    """Lazy process-wide singleton -- embeddings are precomputed once here,
    not per query, and not until the first call (cloud_server.py calls this
    at startup so that first call happens before any request arrives)."""
    global _matcher
    if _matcher is None:
        _matcher = ISROMatcher()
    return _matcher


if __name__ == "__main__":
    import sys

    m = get_matcher()
    query = " ".join(sys.argv[1:]) or "tell me about the moon mission"
    result = m.match(query)
    print(f"query: {query!r}")
    print(json.dumps(result, indent=2))
