from __future__ import annotations

import logging
from typing import Optional

from backend.models.schemas import MemoryAtom, PersonaTrait
from backend.utils.llm_client import get_llm, safe_json_parse

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = """You are a persona synthesis assistant.
Given a user memory atom and the current persona traits, return a single JSON object only.
Do not include any markdown fences, explanation text, or additional keys beyond the requested structure.
Use the existing atom trait_type as a hint and only include supersedes/reinforces indices if applicable.
"""

class SynthesisLayer:
    def __init__(self):
        self.llm = get_llm()

    def synthesize(self, atom: MemoryAtom, current_traits: list[dict]) -> Optional[PersonaTrait]:
        """Convert one atom + existing persona context into a PersonaTrait.

        Returns None if the atom is too thin to form a trait.
        """
        recent = current_traits[:20]
        ctx_lines = [f"[{i}] {t['trait_type']}: {t['value']}" for i, t in enumerate(recent)]
        ctx = "\n".join(ctx_lines) if ctx_lines else "(none)"
        prompt = (
            f"User memory atom: \"{atom.text}\"\n"
            f"Atom trait_type (hint): {atom.trait_type}\n"
            f"Atom confidence: {atom.confidence:.2f}\n\n"
            f"Current persona traits (indexed):\n{ctx}\n"
        )
        prompt += (
            "\nRespond with valid JSON only. The object should look like:\n"
            "{\n  \"trait_type\": \"...\",\n  \"value\": \"...\",\n  \"confidence\": 0.0,\n  \"supersedes\": [0],\n  \"reinforces\": 0\n}\n"
            "Include only keys relevant to this trait. If there is no supersession or reinforcement, omit those fields or use empty/null values."
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
            # Map LLM-returned indices back to actual trait IDs to supersede.
            # This allows cross-type supersession (e.g. a dietary change
            # invalidating a food preference), not just same-type replacement.
            supersede_idxs = parsed.get("supersedes", [])
            supersedes_ids: list[str] = []
            if isinstance(supersede_idxs, list):
                for idx in supersede_idxs:
                    try:
                        i = int(idx)
                    except (ValueError, TypeError):
                        continue
                    if 0 <= i < len(recent):
                        supersedes_ids.append(recent[i]["id"])
            trait.__dict__["_supersedes_ids"] = supersedes_ids
            # Map the optional reinforces index to a concrete trait id.
            reinforces_id = None
            ridx = parsed.get("reinforces", None)
            if ridx is not None:
                try:
                    i = int(ridx)
                    if 0 <= i < len(recent) and recent[i]["id"] not in supersedes_ids:
                        reinforces_id = recent[i]["id"]
                except (ValueError, TypeError):
                    reinforces_id = None
            trait.__dict__["_reinforces_id"] = reinforces_id
            return trait
        except Exception as e:
            logger.warning(f"Failed to build PersonaTrait: {e}")
            return None