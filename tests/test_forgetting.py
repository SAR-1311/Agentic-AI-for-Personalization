"""Forgetting math tests (no DB)."""
import math
from evaluation.metrics import decayed_importance


def test_decay_matches_eq7():
    # I_t = I_0 · e^(-λt)
    I0, lam, t = 0.9, 0.05, 10
    expected = I0 * math.exp(-lam * t)
    assert decayed_importance(I0, t, lam) == expected


def test_decay_floor_threshold():
    # Memory with low base + age will fall below the floor
    floor = 0.05
    score = decayed_importance(0.4, 60, 0.05)
    assert score < floor


def test_decay_high_importance_persists():
    # A 0.95-importance memory should still be > floor after 30 days at λ=0.02
    score = decayed_importance(0.95, 30, 0.02)
    assert score > 0.05
