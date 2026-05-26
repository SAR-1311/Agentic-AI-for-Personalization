"""Structured Persona store — SQLite via SQLAlchemy.

Stores consolidated persona traits (one row per active trait per user) plus
audit history of superseded traits. This is the proposal's "Living Persona".
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from sqlalchemy import (Boolean, Column, DateTime, Float, String, Text,
                        create_engine, select, update)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import get_settings
from backend.models.schemas import PersonaTrait

logger = logging.getLogger(__name__)
Base = declarative_base()


class PersonaTraitRow(Base):
    __tablename__ = "persona_traits"
    id = Column(String, primary_key=True)
    user_id = Column(String, index=True, nullable=False)
    trait_type = Column(String, index=True, nullable=False)
    value = Column(Text, nullable=False)
    evidence = Column(Text, default="")          # comma-separated atom ids
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    superseded = Column(Boolean, default=False)
    superseded_by = Column(String, default=None)


class PersonaDB:
    def __init__(self):
        s = get_settings()
        Path(s.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{s.sqlite_path}", echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def _session(self) -> Iterator[Session]:
        sess = self.Session()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    # -------------------------------------------------------------------
    def upsert_trait(self, trait: PersonaTrait, supersede_existing: bool = True) -> str:
        """Insert a new trait. If `supersede_existing` and a trait of the same
        type already exists for this user, mark it superseded (life-transition)."""
        with self._session() as s:
            if supersede_existing:
                rows = s.execute(select(PersonaTraitRow).where(
                    PersonaTraitRow.user_id == trait.user_id,
                    PersonaTraitRow.trait_type == trait.trait_type,
                    PersonaTraitRow.superseded == False,
                )).scalars().all()
                for r in rows:
                    r.superseded = True
                    r.superseded_by = trait.id
                    r.updated_at = datetime.utcnow()
            row = PersonaTraitRow(
                id=trait.id,
                user_id=trait.user_id,
                trait_type=trait.trait_type,
                value=trait.value,
                evidence=",".join(trait.evidence),
                confidence=trait.confidence,
                created_at=trait.created_at,
                updated_at=trait.updated_at,
                superseded=False,
            )
            s.add(row)
            return trait.id

    def supersede_by_ids(self, trait_ids: list[str], superseded_by: str) -> dict:
        """Mark specific traits (by id) superseded, regardless of trait_type.

        Enables cross-category life-transition reconciliation: a new dietary
        trait can supersede a contradictory food *preference*, etc.
        Returns {"count": n, "evidence_ids": [...]} so the caller can also
        soft-delete the backing memory atoms from the vector store.
        """
        if not trait_ids:
            return {"count": 0, "evidence_ids": []}
        evidence_ids: list[str] = []
        with self._session() as s:
            rows = s.execute(select(PersonaTraitRow).where(
                PersonaTraitRow.id.in_(trait_ids),
                PersonaTraitRow.superseded == False,
            )).scalars().all()
            for r in rows:
                r.superseded = True
                r.superseded_by = superseded_by
                r.updated_at = datetime.utcnow()
                if r.evidence:
                    evidence_ids.extend(e for e in r.evidence.split(",") if e)
            return {"count": len(rows), "evidence_ids": evidence_ids}

    def get_active(self, user_id: str) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(PersonaTraitRow).where(
                PersonaTraitRow.user_id == user_id,
                PersonaTraitRow.superseded == False,
            )).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def get_history(self, user_id: str) -> list[dict]:
        with self._session() as s:
            rows = s.execute(select(PersonaTraitRow).where(
                PersonaTraitRow.user_id == user_id,
            ).order_by(PersonaTraitRow.created_at.asc())).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    def revoke_trait(self, trait_id: str) -> None:
        with self._session() as s:
            s.execute(update(PersonaTraitRow)
                      .where(PersonaTraitRow.id == trait_id)
                      .values(superseded=True, updated_at=datetime.utcnow()))

    def revoke_by_type(self, user_id: str, trait_type: str) -> int:
        with self._session() as s:
            rows = s.execute(select(PersonaTraitRow).where(
                PersonaTraitRow.user_id == user_id,
                PersonaTraitRow.trait_type == trait_type,
                PersonaTraitRow.superseded == False,
            )).scalars().all()
            for r in rows:
                r.superseded = True
                r.updated_at = datetime.utcnow()
            return len(rows)

    @staticmethod
    def _row_to_dict(r: PersonaTraitRow) -> dict:
        return {
            "id": r.id,
            "user_id": r.user_id,
            "trait_type": r.trait_type,
            "value": r.value,
            "evidence": r.evidence.split(",") if r.evidence else [],
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "superseded": r.superseded,
            "superseded_by": r.superseded_by,
        }

    def summary(self, user_id: str) -> str:
        """Compact natural-language summary used in the LLM prompt."""
        active = self.get_active(user_id)
        if not active:
            return "No persona information yet."
        by_type: dict[str, list[str]] = {}
        for t in active:
            by_type.setdefault(t["trait_type"], []).append(t["value"])
        lines = [f"- {tt}: {', '.join(vals)}" for tt, vals in by_type.items()]
        return "\n".join(lines)