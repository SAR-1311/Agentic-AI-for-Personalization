"""Evaluation metrics — implements all equations from §3.3 of the proposal.

  Eq. 4: Precision@K = |relevant ∩ top-K| / K
  Eq. 5: RNR        = |irrelevant retrieved| / |total retrieved|
  Eq. 6: AL         = t_adapt − t_change
  Eq. 7: I_t(m)     = I_0(m) · exp(-λt)        (in core/forgetting.py)

Plus auxiliary metrics for the dissertation: Recall@K, F1, MCQ accuracy,
trait-set Jaccard, forgetting efficacy.
"""
from __future__ import annotations

import math
from typing import Iterable


# ---------------------------------------------------------------------- retrieval

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Eq. 4."""
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / len(relevant_ids)


def retrieval_noise_ratio(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Eq. 5."""
    if not retrieved_ids:
        return 0.0
    irrelevant = sum(1 for x in retrieved_ids if x not in relevant_ids)
    return irrelevant / len(retrieved_ids)


# ---------------------------------------------------------------------- adaptation

def adaptation_latency(t_change: int, agent_decisions: list[bool]) -> int | None:
    """Eq. 6.

    `agent_decisions[t]` is True iff the agent's response at turn `t` already
    reflects the post-change preference. Returns AL in turns, or None if the
    agent never adapts within the observation window.
    """
    for t, adapted in enumerate(agent_decisions):
        if t >= t_change and adapted:
            return t - t_change
    return None


# ---------------------------------------------------------------------- persona

def trait_jaccard(predicted: Iterable[str], ground_truth: Iterable[str]) -> float:
    a, b = set(predicted), set(ground_truth)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def trait_f1(predicted: Iterable[str], ground_truth: Iterable[str]) -> dict:
    a, b = set(predicted), set(ground_truth)
    tp = len(a & b)
    fp = len(a - b)
    fn = len(b - a)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------- forgetting

def forgetting_efficacy(retrieval_rate_after_revoke: float) -> float:
    """Returns 1 - retrieval_rate. Target = 1.0 (memory truly gone)."""
    return max(0.0, 1.0 - retrieval_rate_after_revoke)


# ---------------------------------------------------------------------- decay

def decayed_importance(I0: float, age_days: float, lam: float) -> float:
    """Eq. 7."""
    return I0 * math.exp(-lam * age_days)
