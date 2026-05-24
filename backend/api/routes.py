"""FastAPI routes — public API of the agentic memory system."""
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from backend.core.agent import get_agent
from backend.models.schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.user_id or not req.message:
        raise HTTPException(400, "user_id and message are required")
    return get_agent().chat(req.user_id, req.message, req.session_id)


@router.get("/persona/{user_id}")
def get_persona(user_id: str, history: bool = False):
    a = get_agent()
    if history:
        return {"user_id": user_id, "traits": a.memory.persona.get_history(user_id)}
    return {"user_id": user_id, "traits": a.memory.persona.get_active(user_id),
            "summary": a.memory.persona.summary(user_id)}


@router.get("/memory/{user_id}")
def list_memories(user_id: str, include_pruned: bool = False):
    a = get_agent()
    return {"user_id": user_id,
            "memories": a.memory.ltm.list_for_user(user_id, include_pruned)}


@router.delete("/memory/{memory_id}")
def revoke_memory(memory_id: str, hard: bool = True):
    get_agent().forgetting.revoke_memory(memory_id, hard=hard)
    return {"status": "revoked", "memory_id": memory_id, "hard": hard}


@router.delete("/memory/cluster/{user_id}/{trait_type}")
def revoke_cluster(user_id: str, trait_type: str):
    res = get_agent().forgetting.revoke_cluster(user_id, trait_type)
    return {"status": "revoked", "user_id": user_id, "trait_type": trait_type, **res}


@router.post("/forgetting/run")
def run_decay(user_id: str | None = None):
    return get_agent().forgetting.run_decay_sweep(user_id)


@router.delete("/session/{user_id}")
def clear_session(user_id: str):
    get_agent().memory.stm.clear(user_id)
    return {"status": "cleared", "user_id": user_id}
