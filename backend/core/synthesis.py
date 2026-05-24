"""Synthesis Layer — turns raw memory atoms into structured persona traits.

Implements the proposal's §3.1 'Synthesis Layer' that 'identifies implicit
user traits' and feeds them into the Structured Persona Profile.
"""
from __future__ import annotations

import logging
from typing import Optional

from backend.models.schemas import MemoryAtom, PersonaTrait
from backend.utils.llm_client import get_llm, safe_json_parse

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM = """You consolidate a memory atom about a user into a single
persona trait suitable for long-term storage. The trait should be concise,
generalisable, and stated as a stable property of the user.

Detect contradictions: if the new atom contradicts a current trait of the same
type, mark `contradicts_current=true` and the new value should REPLACE the old.

Return STRICT JSON:
{
  "trait_type": "<one of: preference|dietary|occupation|health|relationship|goal|dislike|routine|fact|other>",
  "value": "<concise stable description, third person singular>",
  "confidence": <0-1>,
  "contradicts_current": <bool>
}

Examples:
  atom="loves pasta", current_traits=[] ->
    {"trait_type": "preference", "value": "enjoys pasta", "confidence": 0.9, "contradicts_current": false}
  atom="started eating meat after seeing doctor", current_traits=["dietary: vegetarian"] ->
    {"trait_type": "dietary", "value": "eats meat (recently changed from vegetarian)", "confidence": 0.85, "contradicts_current": true}
"""


class SynthesisLayer:
    def __init__(self):
        self.llm = get_llm()

    def synthesize(self, atom: MemoryAtom, current_traits: list[dict]) -> Optional[PersonaTrait]:
        """Convert one atom + existing persona context into a PersonaTrait.

        Returns None if the atom is too thin to form a trait.
        """
        ctx_lines = [f"- {t['trait_type']}: {t['value']}" for t in current_traits[:20]]
        ctx = "\n".join(ctx_lines) if ctx_lines else "(none)"
        prompt = (
            f"User memory atom: \"{atom.text}\"\n"
            f"Atom trait_type (hint): {atom.trait_type}\n"
            f"Atom confidence: {atom.confidence:.2f}\n\n"
            f"Current persona traits:\n{ctx}\n"
        )
        out = self.llm.generate(
            prompt=prompt, system=SYNTHESIS_SYSTEM,
            json_mode=True, temperature=0.1, max_tokens=300,
        )
        parsed = safe_json_parse(out)
        if not isinstance(parsed, dict) or "value" not in parsed:
            logger.debug(f"Synthesis returned no trait for atom {atom.id}")
            return None
        try:
            trait = PersonaTrait(
                user_id=atom.user_id,
                trait_type=parsed.get("trait_type", atom.trait_type),
                value=str(parsed["value"]).strip(),
                evidence=[atom.id],
                confidence=float(parsed.get("confidence", atom.confidence)),
            )
            # Stash whether to supersede on the trait (consumed by caller).
            trait.__dict__["_contradicts_current"] = bool(parsed.get("contradicts_current", False))
            return trait
        except Exception as e:
            logger.warning(f"Failed to build PersonaTrait: {e}")
            return None
