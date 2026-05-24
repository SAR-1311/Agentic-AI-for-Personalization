"""Tests for the Reasoning Gatekeeper (no LLM required)."""
import pytest

from backend.core.gatekeeper import linguistic_confidence
from backend.utils.sentiment import emotional_intensity
from evaluation.metrics import (
    decayed_importance, precision_at_k, recall_at_k,
    retrieval_noise_ratio, adaptation_latency, trait_f1,
)


def test_linguistic_confidence_hedges():
    assert linguistic_confidence("Maybe I'll try yoga") < 0.5

def test_linguistic_confidence_absolutes():
    assert linguistic_confidence("I always go running every morning") > 0.6

def test_linguistic_confidence_neutral():
    c = linguistic_confidence("It's Tuesday today")
    assert 0.4 <= c <= 0.6

def test_emotional_intensity_strong_positive():
    assert emotional_intensity("I absolutely love traveling!") > 0.5

def test_emotional_intensity_strong_negative():
    assert emotional_intensity("I really hate Mondays") > 0.4

def test_emotional_intensity_neutral():
    assert emotional_intensity("It's Tuesday") < 0.2

# ---------- Eq. 4 ----------
def test_precision_at_k_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

def test_precision_at_k_none():
    assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0

# ---------- Eq. 5 ----------
def test_rnr_clean():
    assert retrieval_noise_ratio(["a", "b"], {"a", "b"}) == 0.0

def test_rnr_dirty():
    assert retrieval_noise_ratio(["a", "x", "y"], {"a"}) == pytest.approx(2/3)

# ---------- Eq. 6 ----------
def test_adaptation_latency_immediate():
    assert adaptation_latency(0, [True, True, True]) == 0

def test_adaptation_latency_delayed():
    assert adaptation_latency(0, [False, False, True]) == 2

def test_adaptation_latency_never():
    assert adaptation_latency(0, [False, False, False]) is None

# ---------- Eq. 7 ----------
def test_decay_zero_age():
    assert decayed_importance(0.8, 0, 0.05) == 0.8

def test_decay_decreases_over_time():
    early = decayed_importance(0.8, 1, 0.05)
    late = decayed_importance(0.8, 30, 0.05)
    assert late < early < 0.8

# ---------- F1 helpers ----------
def test_trait_f1_full_match():
    m = trait_f1(["a", "b"], ["a", "b"])
    assert m["f1"] == 1.0

def test_trait_f1_partial():
    m = trait_f1(["a", "b", "c"], ["b", "c", "d"])
    assert 0 < m["f1"] < 1
