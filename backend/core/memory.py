"""Active Context (Short-Term Memory) and the public Memory facade.

The STM is a per-user deque of recent turns. The LTM is delegated to ChromaDB
via VectorStore. This module exposes the unified `Memory` interface that the
agent uses.
"""
from __future__ import annotations

from collections import deque
from typing import Deque

from backend.config import get_settings
from backend.models.schemas import MemoryAtom, RetrievedMemory
from backend.storage.persona_db import PersonaDB
from backend.storage.vector_db import VectorStore


class ActiveContext:
    """Per-user FIFO buffer of (role, text) turns."""

    def __init__(self):
        self._buffers: dict[str, Deque[dict]] = {}
        self.maxlen = get_settings().active_context_size

    def append(self, user_id: str, role: str, text: str) -> None:
        buf = self._buffers.setdefault(user_id, deque(maxlen=self.maxlen))
        buf.append({"role": role, "text": text})

    def get(self, user_id: str) -> list[dict]:
        return list(self._buffers.get(user_id, []))

    def clear(self, user_id: str) -> None:
        if user_id in self._buffers:
            self._buffers[user_id].clear()


class Memory:
    """Unified memory facade: STM + LTM + structured persona."""

    def __init__(self):
        self.stm = ActiveContext()
        self.ltm = VectorStore()
        self.persona = PersonaDB()
        self.top_k = get_settings().retrieval_top_k

    # -------------------------------------------------------------------
    # STM
    # -------------------------------------------------------------------
    def push_user(self, user_id: str, text: str) -> None:
        self.stm.append(user_id, "user", text)

    def push_assistant(self, user_id: str, text: str) -> None:
        self.stm.append(user_id, "assistant", text)

    def history(self, user_id: str) -> list[dict]:
        return self.stm.get(user_id)

    # -------------------------------------------------------------------
    # LTM
    # -------------------------------------------------------------------
    def store_atom(self, atom: MemoryAtom) -> None:
        self.ltm.add(atom)

    def retrieve(self, user_id: str, query: str) -> list[RetrievedMemory]:
        return self.ltm.retrieve(query, user_id, top_k=self.top_k)

    # -------------------------------------------------------------------
    # Persona
    # -------------------------------------------------------------------
    def persona_summary(self, user_id: str) -> str:
        return self.persona.summary(user_id)
