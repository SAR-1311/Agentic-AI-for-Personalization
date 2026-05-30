"""Reasoning Gatekeeper — Eq. 1 of the proposal:

    I(m) = α · f(m) + β · c(m) + γ · e(m)

For each user utterance the gatekeeper:
  1. Asks the LLM to extract candidate "memory atoms" (statements of fact /
     preference / event) with structured output.
  2. Computes f(m), c(m), e(m) for each atom.
  3. Returns only atoms whose I(m) >= IMPORTANCE_THRESHOLD.

This is the core innovation that distinguishes the system from flat-RAG: noise
is filtered *before* it reaches storage.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from backend.config import get_settings
from backend.models.schemas import MemoryAtom
from backend.utils.llm_client import get_llm, safe_json_parse
from backend.utils.sentiment import emotional_intensity

if TYPE_CHECKING:
    from backend.storage.persona_db import PersonaDB
    from backend.storage.vector_db import VectorStore

logger = logging.getLogger(__name__)


# --- Linguistic cues for confidence c(m) ----------------------------------

HEDGES = {
    "maybe", "might", "perhaps", "possibly", "probably", "i think",
    "i guess", "kind of", "sort of", "i suppose", "could be", "not sure",
    "i'm not sure", "potentially",
}
ABSOLUTES = {
    "always", "never", "definitely", "absolutely", "certainly", "of course",
    "without a doubt", "i love", "i hate", "every day", "every single",
    "i am", "i'm a",
}


def linguistic_confidence(text: str) -> float:
    """Heuristic confidence in [0, 1] from hedges/absolutes."""
    t = text.lower()
    score = 0.5  # neutral
    for cue in ABSOLUTES:
        if cue in t:
            score += 0.15
    for cue in HEDGES:
        if cue in t:
            score -= 0.15
    return max(0.0, min(1.0, score))


# --- Atom extraction prompt -----------------------------------------------

EXTRACTION_SYSTEM = """You analyse a single user message and extract distinct
'memory atoms' — facts, preferences, dislikes, routines, goals, or life events
the user has revealed about themselves. Ignore questions, greetings, and
content directed at the assistant. If nothing is revealed, return an empty list.

Return STRICT JSON with this schema:
{
  "atoms": [
    {
      "text": "<concise paraphrase, first-person>",
      "trait_type": "preference|dietary|occupation|health|relationship|goal|dislike|routine|fact|other",
      "llm_confidence": <float 0-1, how certain the user sounds>
    }
  ]
}
Examples:
  "I love pasta and go running daily" ->
    [{"text": "loves pasta", "trait_type": "preference", "llm_confidence": 0.95},
     {"text": "runs daily", "trait_type": "routine", "llm_confidence": 0.9}]
  "What's the weather?" -> []
  "Maybe I'll try yoga" ->
    [{"text": "considering trying yoga", "trait_type": "goal", "llm_confidence": 0.4}]
"""


class Gatekeeper:
    """Implements Eq. 1: I(m) = α·f(m) + β·c(m) + γ·e(m)."""

    def __init__(self, vector_store: "VectorStore | None" = None,
                 persona_db: "PersonaDB | None" = None):
        s = get_settings()
        self.alpha = s.weight_frequency
        self.beta = s.weight_confidence
        self.gamma = s.weight_emotion
        self.threshold = s.importance_threshold
        self.vector_store = vector_store
        self.persona_db = persona_db
        self.llm = get_llm()
        # Per-user atom counter; replaces the old global counter so concurrent
        # users no longer share a denominator in f(m).
        self._user_count: dict[str, int] = {}

    # -------------------------------------------------------------------
    # Eq. 2 — Frequency f(m)
    # -------------------------------------------------------------------
    def frequency(self, atom_text: str, user_id: str,
                  trait_hint: str | None = None) -> float:
        """Eq. 2 — recurrence of this concept for this user, in [0, 1].

        Phase-2 refinements:
          * Per-user interaction counter (no cross-user pollution).
          * Tiered surface similarity: matches above 0.85 count fully, 0.70-0.85
            count at half weight, so paraphrases of an existing memory still
            register.
          * Persona-reinforcement bridge: a matching active trait that has been
            reinforced k times contributes (k - 1) prior occurrences. Ties this
            equation to the reinforcement_count introduced in Phase 2.1.
          * Laplace smoothing (Beta(1, 1) prior): numerator +1 for the current
            atom as one occurrence, denominator +2, which dampens cold-start
            inflation without erasing first-mention signal.
        """
        self._user_count[user_id] = self._user_count.get(user_id, 0) + 1
        n = self._user_count[user_id]

        if self.vector_store is None:
            return 0.0

        strong = self.vector_store.count_similar(atom_text, user_id, sim_threshold=0.85)
        loose = self.vector_store.count_similar(atom_text, user_id, sim_threshold=0.70)
        sim_evidence = strong + 0.5 * max(0, loose - strong)

        persona_evidence = self._persona_recurrence(atom_text, trait_hint, user_id)

        occurrences = sim_evidence + persona_evidence + 1.0  # +1 for current atom
        return min(1.0, occurrences / (n + 2.0))

    # --- helper: bridge to Phase-2.1 reinforcement ---------------------
    # Generic preference verbs/adverbs are stripped so matching is driven by
    # content nouns (e.g. "biryani") rather than scaffolding ("enjoys", "every").
    _STOPWORDS = {
        "the","a","an","is","of","and","or","to","in","on","at","with","for",
        "i","you","he","she","it","my","our","your","this","that","be","am",
        "are","was","were","do","does","did","have","has","had","not","no",
        # generic trait-construction verbs and adverbs
        "enjoys","likes","loves","prefers","dislikes","hates","wants","needs",
        "eats","eat","drinks","drink","plays","play","goes","gets","makes",
        "really","absolutely","very","just","honestly","truly",
        "every","day","daily","weekly","always","never","often","sometimes",
        "could","would","should","might","may","can","will",
    }

    @classmethod
    def _content_tokens(cls, s: str) -> set[str]:
        return {w.lower().strip(".,!?;:'\"")
                for w in (s or "").split()
                if len(w) > 2 and w.lower() not in cls._STOPWORDS}

    def _persona_recurrence(self, atom_text: str, trait_hint: str | None,
                            user_id: str) -> float:
        """Find an active persona trait that semantically matches this atom and
        return its prior reinforcement count (reinforcement_count - 1).

        Uses min-based overlap (Szymkiewicz-Simpson coefficient) on content
        tokens, which handles short trait values robustly. No extra LLM call.
        """
        if self.persona_db is None:
            return 0.0
        try:
            actives = self.persona_db.get_active(user_id)
        except Exception:
            return 0.0
        if not actives:
            return 0.0
        atom_tokens = self._content_tokens(atom_text)
        if not atom_tokens:
            return 0.0
        best = 0.0
        for t in actives:
            if trait_hint and t.get("trait_type") != trait_hint:
                continue
            v_tokens = self._content_tokens(t.get("value", ""))
            if not v_tokens:
                continue
            shared = len(atom_tokens & v_tokens)
            overlap = shared / min(len(atom_tokens), len(v_tokens))
            if overlap >= 0.5 and shared >= 1:
                recurrence = max(0, (t.get("reinforcement_count", 1) - 1))
                if recurrence > best:
                    best = float(recurrence)
        return best

    # -------------------------------------------------------------------
    # Confidence c(m) — combine linguistic cues + LLM rating
    # -------------------------------------------------------------------
    @staticmethod
    def confidence(text: str, llm_conf: float) -> float:
        ling = linguistic_confidence(text)
        return 0.5 * ling + 0.5 * float(llm_conf)

    # -------------------------------------------------------------------
    # Eq. 3 — Emotion e(m)
    # -------------------------------------------------------------------
    @staticmethod
    def emotion(text: str) -> float:
        return emotional_intensity(text)

    # -------------------------------------------------------------------
    # Eq. 1 — Importance I(m)
    # -------------------------------------------------------------------
    def importance(self, f: float, c: float, e: float) -> float:
        return self.alpha * f + self.beta * c + self.gamma * e

    # -------------------------------------------------------------------
    # Atom extraction (LLM call)
    # -------------------------------------------------------------------
    def extract_atoms(self, message: str) -> list[dict]:
        out = self.llm.generate(
            prompt=f"User message:\n\"\"\"\n{message}\n\"\"\"",
            system=EXTRACTION_SYSTEM,
            json_mode=True,
            temperature=0.1,
        )
        parsed = safe_json_parse(out) or {}
        atoms = parsed.get("atoms", []) if isinstance(parsed, dict) else []
        # Defensive validation
        valid = []
        for a in atoms:
            if not isinstance(a, dict) or "text" not in a:
                continue
            valid.append({
                "text": str(a["text"]).strip(),
                "trait_type": a.get("trait_type", "other"),
                "llm_confidence": float(a.get("llm_confidence", 0.5)),
            })
        return valid

    # -------------------------------------------------------------------
    # Public entrypoint
    # -------------------------------------------------------------------
    def process(self, message: str, user_id: str,
                session_id: str | None = None,
                turn: int | None = None) -> tuple[list[MemoryAtom], list[MemoryAtom]]:
        """Process one user message.

        Returns (passed, rejected) — atoms with I(m) >= threshold and below.
        """
        raw = self.extract_atoms(message)
        passed: list[MemoryAtom] = []
        rejected: list[MemoryAtom] = []
        for a in raw:
            f = self.frequency(a["text"], user_id, trait_hint=a.get("trait_type"))
            c = self.confidence(a["text"], a["llm_confidence"])
            e = self.emotion(a["text"])
            I = self.importance(f, c, e)
            atom = MemoryAtom(
                user_id=user_id,
                text=a["text"],
                trait_type=a["trait_type"],
                frequency=f,
                confidence=c,
                emotion=e,
                importance=I,
                decayed_importance=I,
                source_session=session_id,
                source_turn=turn,
            )
            if I >= self.threshold:
                passed.append(atom)
            else:
                rejected.append(atom)
        logger.info(f"Gatekeeper: {len(passed)} passed, {len(rejected)} rejected "
                    f"(α={self.alpha} β={self.beta} γ={self.gamma} τ={self.threshold})")
        return passed, rejected