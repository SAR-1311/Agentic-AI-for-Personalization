"""Vector store wrapper around ChromaDB.

Stores memory atoms with their importance metadata. Retrieval combines cosine
similarity with importance for the proposal's importance-weighted recall.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.models.schemas import MemoryAtom, RetrievedMemory
from backend.utils.llm_client import get_llm

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        s = get_settings()
        Path(s.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=s.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.coll = self.client.get_or_create_collection(
            name="memory_atoms",
            metadata={"hnsw:space": "cosine"},
        )
        self.llm = get_llm()
        self.sim_w = s.retrieval_sim_weight
        self.imp_w = s.retrieval_imp_weight

    # -------------------------------------------------------------------
    def add(self, atom: MemoryAtom) -> None:
        emb = self.llm.embed(atom.text)
        self.coll.add(
            ids=[atom.id],
            embeddings=[emb],
            documents=[atom.text],
            metadatas=[{
                "user_id": atom.user_id,
                "trait_type": atom.trait_type,
                "importance": atom.importance,
                "decayed_importance": atom.decayed_importance,
                "frequency": atom.frequency,
                "confidence": atom.confidence,
                "emotion": atom.emotion,
                "created_at": atom.created_at.isoformat(),
                "last_reinforced_at": atom.last_reinforced_at.isoformat(),
                "superseded": atom.superseded,
                "revoked": atom.revoked,
                "source_session": atom.source_session or "",
                "source_turn": atom.source_turn or -1,
            }],
        )

    # -------------------------------------------------------------------
    def retrieve(self, query: str, user_id: str, top_k: int = 5) -> list[RetrievedMemory]:
        """Importance-weighted top-K retrieval.

        ranking = sim_w * cosine_sim + imp_w * decayed_importance
        """
        try:
            emb = self.llm.embed(query)
            res = self.coll.query(
                query_embeddings=[emb],
                n_results=max(top_k * 4, 12),
                where={
                    "$and": [
                        {"user_id": user_id},
                        {"revoked": False},
                        {"superseded": False},
                    ]
                },
            )
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        ranked: list[tuple[float, RetrievedMemory]] = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(dist)              # cosine distance → similarity
            imp = float(meta.get("decayed_importance", meta.get("importance", 0.0)))
            score = self.sim_w * sim + self.imp_w * imp
            ranked.append((score, RetrievedMemory(
                id=_id, text=doc, importance=imp, similarity=sim,
                trait_type=str(meta.get("trait_type", "other")),
            )))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in ranked[:top_k]]

    # -------------------------------------------------------------------
    def count_similar(self, text: str, user_id: str, sim_threshold: float = 0.85) -> int:
        """Used by Gatekeeper.frequency()."""
        try:
            emb = self.llm.embed(text)
            res = self.coll.query(
                query_embeddings=[emb],
                n_results=20,
                where={"user_id": user_id, "revoked": False},
            )
        except Exception:
            return 0
        dists = res.get("distances", [[]])[0]
        return sum(1 for d in dists if (1.0 - float(d)) >= sim_threshold)

    # -------------------------------------------------------------------
    def list_for_user(self, user_id: str, include_pruned: bool = False) -> list[dict]:
        where = {"user_id": user_id} if include_pruned else {
            "$and": [{"user_id": user_id}, {"revoked": False}]
        }
        res = self.coll.get(where=where)
        out = []
        for _id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append({"id": _id, "text": doc, **meta})
        return out

    def update_importance(self, memory_id: str, new_importance: float) -> None:
        """Used by ForgettingEngine to apply decay."""
        existing = self.coll.get(ids=[memory_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta["decayed_importance"] = new_importance
        self.coll.update(ids=[memory_id], metadatas=[meta])

    def soft_delete(self, memory_id: str, kind: str = "revoked") -> None:
        existing = self.coll.get(ids=[memory_id])
        if not existing["ids"]:
            return
        meta = existing["metadatas"][0]
        meta[kind] = True
        self.coll.update(ids=[memory_id], metadatas=[meta])

    def hard_delete(self, memory_id: str) -> None:
        try:
            self.coll.delete(ids=[memory_id])
        except Exception as e:
            logger.warning(f"Hard delete failed for {memory_id}: {e}")

    def hard_delete_by_trait(self, user_id: str, trait_type: str) -> int:
        res = self.coll.get(where={
            "$and": [{"user_id": user_id}, {"trait_type": trait_type}]
        })
        ids = res["ids"]
        if ids:
            self.coll.delete(ids=ids)
        return len(ids)
