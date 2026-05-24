"""Controlled Forgetting — Eq. 7 of the proposal:

    I_t(m) = I_0(m) · e^(-λt)

Two modes:
  1. Temporal decay  — automatic, runs as a sweep over LTM.
  2. User revocation — explicit deletion of single memory or trait cluster
                       (the GDPR-style 'right to be forgotten').
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

from backend.config import get_settings
from backend.storage.persona_db import PersonaDB
from backend.storage.vector_db import VectorStore

logger = logging.getLogger(__name__)


class ForgettingEngine:
    def __init__(self, vector_store: VectorStore, persona_db: PersonaDB):
        s = get_settings()
        self.vec = vector_store
        self.persona = persona_db
        self.lam = s.decay_lambda
        self.floor = s.forget_floor

    # -------------------------------------------------------------------
    # Eq. 7 — temporal decay
    # -------------------------------------------------------------------
    def decayed(self, base_importance: float, age_days: float) -> float:
        return base_importance * math.exp(-self.lam * age_days)

    def run_decay_sweep(self, user_id: str | None = None) -> dict:
        """Iterate all (or one user's) atoms, recompute decayed importance,
        soft-delete those below the forget floor."""
        atoms = self.vec.coll.get(
            where={"user_id": user_id} if user_id else None
        )
        ids = atoms.get("ids", [])
        metas = atoms.get("metadatas", [])
        now = datetime.utcnow()
        updated = 0
        soft_deleted = 0
        for _id, meta in zip(ids, metas):
            if meta.get("revoked", False):
                continue
            base = float(meta.get("importance", 0.0))
            last = meta.get("last_reinforced_at") or meta.get("created_at")
            try:
                last_dt = datetime.fromisoformat(last) if isinstance(last, str) else now
            except Exception:
                last_dt = now
            age_days = max(0.0, (now - last_dt).total_seconds() / 86400.0)
            new = self.decayed(base, age_days)
            self.vec.update_importance(_id, new)
            updated += 1
            if new < self.floor:
                self.vec.soft_delete(_id, kind="superseded")  # decayed-out, not user-revoked
                soft_deleted += 1
        logger.info(f"Decay sweep: updated={updated} soft_deleted={soft_deleted}")
        return {"updated": updated, "soft_deleted": soft_deleted}

    # -------------------------------------------------------------------
    # User-driven revocation
    # -------------------------------------------------------------------
    def revoke_memory(self, memory_id: str, hard: bool = True) -> None:
        if hard:
            self.vec.hard_delete(memory_id)
        else:
            self.vec.soft_delete(memory_id, kind="revoked")
        logger.info(f"Revoked memory {memory_id} (hard={hard})")

    def revoke_cluster(self, user_id: str, trait_type: str) -> dict:
        """Hard-delete every atom AND every persona trait of this type."""
        n_atoms = self.vec.hard_delete_by_trait(user_id, trait_type)
        n_traits = self.persona.revoke_by_type(user_id, trait_type)
        logger.info(f"Revoked cluster {trait_type}: atoms={n_atoms} traits={n_traits}")
        return {"atoms_deleted": n_atoms, "traits_revoked": n_traits}
