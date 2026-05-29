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

You are also given the user's CURRENT persona traits, each with an index like [0].
Decide which of those existing traits this new atom makes outdated or
contradicts. A new trait supersedes an existing one when they cannot both be
true at the same time — REGARDLESS of trait_type. For example, a dietary change
("does not eat chicken") supersedes a food preference that depends on it
("enjoys chicken biryani"), but does NOT supersede a preference that is still
valid ("enjoys biryani", which may be vegetable biryani). Only list indices you
are confident are now outdated.

Also decide whether this atom merely RESTATES an existing trait that is still
true (same meaning, no change). If so, set "reinforces" to that trait's index so
the system strengthens it instead of storing a duplicate. A trait must not both
reinforce and supersede the same index; if the atom genuinely restates, prefer
"reinforces" and leave "supersedes" empty.

Return STRICT JSON:
{
  "trait_type": "<one of: preference|dietary|occupation|health|relationship|goal|dislike|routine|fact|other>",
  "value": "<concise stable description, third person singular>",
  "confidence": <0-1>,
  "supersedes": [<indices of current traits this new trait makes outdated; [] if none>],
  "reinforces": <index of an existing trait this atom restates, or null>
}

Examples:
  atom="loves pasta", current=(none) ->
    {"trait_type": "preference", "value": "enjoys pasta", "confidence": 0.9, "supersedes": [], "reinforces": null}
  atom="stopped eating chicken",
  current=[0] preference: enjoys chicken biryani / [1] occupation: software engineer ->
    {"trait_type": "dietary", "value": "does not eat chicken (recently changed)", "confidence": 0.85, "supersedes": [0], "reinforces": null}
  atom="i could honestly eat biryani daily",
  current=[0] preference: enjoys biryani ->
    {"trait_type": "preference", "value": "enjoys biryani", "confidence": 0.9, "supersedes": [], "reinforces": 0}
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