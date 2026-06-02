"""Pydantic schemas — single source of truth for data structures."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4


TraitType = Literal[
    "preference", "dietary", "occupation", "health",
    "relationship", "goal", "dislike", "routine", "fact",
    # Expanded vocabulary — these are common categories the gatekeeper LLM
    # produces and they're genuinely useful for downstream analysis.
    "emotion", "hobby", "skill", "interest", "opinion",
    "lifestyle", "experience", "demographic", "belief",
    "other",
]

_TRAIT_TYPE_ALIASES: set[str] = {
    "preference", "dietary", "occupation", "health",
    "relationship", "goal", "dislike", "routine", "fact",
    "emotion", "hobby", "skill", "interest", "opinion",
    "lifestyle", "experience", "demographic", "belief", "other",
}


def _coerce_trait_type(v):
    """Normalise trait_type so LLM-produced variants outside the allowlist
    collapse to 'other' instead of raising a validation error mid-ingest."""
    if isinstance(v, str):
        v_norm = v.strip().lower()
        if v_norm in _TRAIT_TYPE_ALIASES:
            return v_norm
    return "other"


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

    @field_validator("trait_type", mode="before")
    @classmethod
    def _normalise_trait_type(cls, v):
        return _coerce_trait_type(v)


class PersonaTrait(BaseModel):
    """A consolidated trait stored in the structured persona."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    trait_type: TraitType
    value: str
    evidence: list[str] = []      # supporting memory atom ids
    confidence: float = 0.0
    reinforcement_count: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_reinforced_at: datetime = Field(default_factory=datetime.utcnow)
    superseded_by: Optional[str] = None

    @field_validator("trait_type", mode="before")
    @classmethod
    def _normalise_trait_type(cls, v):
        return _coerce_trait_type(v)


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