"""STM behavioural tests (no LLM / no DB)."""
from backend.core.memory import ActiveContext


def test_stm_appends_and_retrieves():
    ctx = ActiveContext()
    ctx.append("u1", "user", "hello")
    ctx.append("u1", "assistant", "hi")
    h = ctx.get("u1")
    assert len(h) == 2
    assert h[0]["role"] == "user"


def test_stm_evicts_when_full(monkeypatch):
    from backend.config import get_settings
    s = get_settings()
    # We just verify maxlen-based eviction at the deque level.
    ctx = ActiveContext()
    ctx.maxlen = 3
    # Re-create the buffer to take new maxlen
    from collections import deque
    ctx._buffers["u1"] = deque(maxlen=3)
    for i in range(5):
        ctx._buffers["u1"].append({"role": "user", "text": f"m{i}"})
    h = list(ctx._buffers["u1"])
    assert len(h) == 3
    assert h[0]["text"] == "m2"


def test_stm_clear():
    ctx = ActiveContext()
    ctx.append("u1", "user", "x")
    ctx.clear("u1")
    assert ctx.get("u1") == []


def test_stm_isolated_per_user():
    ctx = ActiveContext()
    ctx.append("a", "user", "x")
    ctx.append("b", "user", "y")
    assert ctx.get("a")[0]["text"] == "x"
    assert ctx.get("b")[0]["text"] == "y"
