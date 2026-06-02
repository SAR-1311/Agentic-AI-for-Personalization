from __future__ import annotations

import logging
import uuid
from datetime import datetime

from backend.core.forgetting import ForgettingEngine
from backend.core.gatekeeper import Gatekeeper
from backend.core.memory import Memory
from backend.core.synthesis import SynthesisLayer
from backend.models.schemas import ChatResponse, MemoryAtom, PersonaTrait
from backend.utils.llm_client import get_llm

logger = logging.getLogger(__name__)


RESPONSE_SYSTEM_TEMPLATE = """You are a personalised long-term assistant.
You have access to:
  • A structured persona profile of the user (consolidated traits).
  • Top retrieved memories from prior conversations.
  • The recent dialogue context.

Use these to give responses that are consistent, personal, and considerate of
the user's preferences and history. Do NOT mention 'memories' or expose the
internals — speak naturally as a friend or assistant who simply remembers.
If a retrieved memory contradicts something the user just said, trust the
*newer* statement.
"""


class Agent:
    def __init__(self):
        self.memory = Memory()
        self.gatekeeper = Gatekeeper(vector_store=self.memory.ltm,
                                     persona_db=self.memory.persona)
        self.synthesis = SynthesisLayer()
        self.forgetting = ForgettingEngine(self.memory.ltm, self.memory.persona)
        self.llm = get_llm()

    # -------------------------------------------------------------------
    def chat(self, user_id: str, message: str,
             session_id: str | None = None,
             generate_reply: bool = True) -> ChatResponse:
        session_id = session_id or str(uuid.uuid4())
        history = self.memory.history(user_id)
        turn = len(history) // 2  # rough turn index

        # Push to STM
        self.memory.push_user(user_id, message)

        # Gatekeeper — score atoms, pass only high-signal
        passed, rejected = self.gatekeeper.process(message, user_id, session_id, turn)

        # Synthesis — for each passed atom, build/upsert a persona trait
        synthesized: list[dict] = []
        current = self.memory.persona.get_active(user_id)
        for atom in passed:
            self.memory.store_atom(atom)
            trait = self.synthesis.synthesize(atom, current)
            if trait is None:
                continue
            reinforces_id = trait.__dict__.get("_reinforces_id")
            supersedes_ids = trait.__dict__.get("_supersedes_ids", [])

            # Restatement → strengthen the existing trait, don't duplicate.
            if reinforces_id:
                res = self.memory.persona.reinforce_trait(
                    reinforces_id, atom.id, trait.confidence)
                if res is not None:
                    synthesized.append({"trait_type": trait.trait_type,
                                        "value": res["value"],
                                        "reinforced": True,
                                        "reinforcement_count": res["reinforcement_count"],
                                        "superseded_count": 0})
                    current = self.memory.persona.get_active(user_id)
                    continue
                # trait vanished/superseded since synthesis → fall through to insert

            # New or contradictory trait → insert, then supersede outdated ones.
            self.memory.persona.upsert_trait(trait, supersede_existing=False)
            res = self.memory.persona.supersede_by_ids(supersedes_ids, trait.id)
            for ev_id in res["evidence_ids"]:
                self.memory.ltm.soft_delete(ev_id, kind="superseded")
            synthesized.append({"trait_type": trait.trait_type,
                                "value": trait.value,
                                "reinforced": False,
                                "superseded_existing": res["count"] > 0,
                                "superseded_count": res["count"]})
            current = self.memory.persona.get_active(user_id)  # refresh

        # Importance-weighted retrieval + reply generation
        # Skip these steps during cheap ingest (used by PersonaMem eval).
        if not generate_reply:
            diagnostics = {
                "session_id": session_id,
                "turn": turn,
                "gatekeeper": {
                    "passed": [self._atom_brief(a) for a in passed],
                    "rejected": [self._atom_brief(a) for a in rejected],
                },
                "synthesized": synthesized,
                "persona_size": len(current),
                "timestamp": datetime.utcnow().isoformat(),
                "ingest_only": True,
            }
            return ChatResponse(reply="", session_id=session_id, diagnostics=diagnostics)

        retrieved = self.memory.retrieve(user_id, message)

        # Build response prompt
        persona_block = self.memory.persona_summary(user_id)
        memory_block = "\n".join(
            f"  • [{m.trait_type} | sim={m.similarity:.2f} | I={m.importance:.2f}] {m.text}"
            for m in retrieved
        ) or "  (none)"
        history_block = "\n".join(
            f"{h['role'].capitalize()}: {h['text']}"
            for h in history[-10:]   # last 10 turns for prompt economy
        ) or "(no prior turns)"

        prompt = (
            f"=== USER PERSONA (structured) ===\n{persona_block}\n\n"
            f"=== RETRIEVED MEMORIES (importance-weighted) ===\n{memory_block}\n\n"
            f"=== RECENT DIALOGUE ===\n{history_block}\n\n"
            f"=== CURRENT USER MESSAGE ===\n{message}\n\n"
            f"Reply naturally as the personalised assistant."
        )

        reply = self.llm.generate(
            prompt=prompt, system=RESPONSE_SYSTEM_TEMPLATE,
            temperature=0.6, max_tokens=600,
        )

        # Push reply to STM
        self.memory.push_assistant(user_id, reply)

        diagnostics = {
            "session_id": session_id,
            "turn": turn,
            "gatekeeper": {
                "passed": [self._atom_brief(a) for a in passed],
                "rejected": [self._atom_brief(a) for a in rejected],
                "weights": {
                    "alpha": self.gatekeeper.alpha,
                    "beta": self.gatekeeper.beta,
                    "gamma": self.gatekeeper.gamma,
                    "threshold": self.gatekeeper.threshold,
                },
            },
            "synthesized": synthesized,
            "retrieved": [m.model_dump() for m in retrieved],
            "persona_size": len(current),
            "timestamp": datetime.utcnow().isoformat(),
        }
        return ChatResponse(reply=reply, session_id=session_id, diagnostics=diagnostics)

    # -------------------------------------------------------------------
    @staticmethod
    def _atom_brief(a: MemoryAtom) -> dict:
        return {
            "text": a.text,
            "trait_type": a.trait_type,
            "f": round(a.frequency, 3),
            "c": round(a.confidence, 3),
            "e": round(a.emotion, 3),
            "I": round(a.importance, 3),
        }


# Module-level singleton (cheap lazy init for FastAPI workers).
_agent: Agent | None = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent