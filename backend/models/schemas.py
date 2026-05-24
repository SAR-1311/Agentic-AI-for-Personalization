"""Pydantic schemas — single source of truth for data structures."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


TraitType = Literal[
    "preference", "dietary", "occupation", "health",
    "relationship", "goal", "dislike", "routine", "fact", "other",
]


class MemoryAtom(BaseModel):
    """A discrete piece of information extracted from a user utterance."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    text: str
    trait_type: TraitType = "other"
    # Score components — Eq. 1
    frequency: float = 0.0
    confidence: float = 0.0
    emotion: float = 0.0
    importance: float = 0.0      # I(m)
    # Time tracking (for decay — Eq. 7)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_reinforced_at: datetime = Field(default_factory=datetime.utcnow)
    decayed_importance: float = 0.0
    # Status
    superseded: bool = False
    revoked: bool = False
    # Source
    source_session: Optional[str] = None
    source_turn: Optional[int] = None


class PersonaTrait(BaseModel):
    """A consolidated trait stored in the structured persona."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    trait_type: TraitType
    value: str
    evidence: list[str] = []      # supporting memory atom ids
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    superseded_by: Optional[str] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None


class RetrievedMemory(BaseModel):
    id: str
    text: str
    importance: float
    similarity: float
    trait_type: str


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    diagnostics: dict           # gatekeeper scores, retrieved memories, persona snapshot
