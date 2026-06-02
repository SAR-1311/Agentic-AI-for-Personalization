from __future__ import annotations

import math
import os
import tempfile
import uuid
from datetime import datetime, timedelta

# Isolate from real data BEFORE any backend import.
# IMPORTANT: the chroma env var name must match the Settings field name
# (chroma_persist_dir), uppercased.
os.environ["SQLITE_PATH"]        = tempfile.mktemp(suffix=".db", prefix="forget_demo_")
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp(prefix="forget_demo_chroma_")

from backend.config import get_settings                       # noqa: E402
from backend.core.forgetting import ForgettingEngine          # noqa: E402
from backend.models.schemas import MemoryAtom                 # noqa: E402
from backend.storage.persona_db import PersonaDB              # noqa: E402
from backend.storage.vector_db import VectorStore             # noqa: E402

USER_ID = f"forget_{uuid.uuid4().hex[:8]}"


def _seed_atom(text: str, importance: float, days_ago: float) -> MemoryAtom:
    """Build an atom whose last_reinforced_at is backdated by `days_ago` days."""
    ts = datetime.utcnow() - timedelta(days=days_ago)
    return MemoryAtom(
        user_id=USER_ID,
        text=text,
        trait_type="preference",
        frequency=0.5, confidence=0.7, emotion=0.3,
        importance=importance,
        decayed_importance=importance,  # starts at I_0
        created_at=ts,
        last_reinforced_at=ts,
    )


# Scenarios chosen so the table tells a complete story:
#   * Ancient atoms decay below the floor regardless of original importance.
#   * Recent atoms survive regardless of how modest their importance.
#   * Middle-aged atoms degrade but stay above the floor.
SCENARIOS: list[tuple[str, MemoryAtom]] = [
    ("ancient, low-imp one-off",        _seed_atom("watched a film once",     importance=0.40, days_ago=120)),
    ("ancient, high-imp never restated", _seed_atom("loved Bali trip in 2020", importance=0.80, days_ago=120)),
    ("middle-aged, mid-imp",            _seed_atom("listens to jazz",         importance=0.55, days_ago=30)),
    ("very recent, mid-imp",            _seed_atom("started learning guitar", importance=0.55, days_ago=5)),
    ("recent, low-imp",                 _seed_atom("had coffee yesterday",    importance=0.40, days_ago=2)),
    ("recent, high-imp (reinforced)",   _seed_atom("loves biryani",           importance=0.80, days_ago=2)),
]


def main() -> None:
    s = get_settings()
    print("=" * 78)
    print("Forgetting demo  —  Eq. 7:  I_t(m) = I_0(m) · exp(-λ · t)")
    print("=" * 78)
    print(f"  λ (decay_lambda)    = {s.decay_lambda}")
    print(f"  forget_floor        = {s.forget_floor}")
    print(f"  user_id             = {USER_ID}")
    print(f"  chroma_persist_dir  = {s.chroma_persist_dir}")
    if "forget_demo_" not in s.chroma_persist_dir:
        print("\n  !!! WARNING: isolation may have failed. Aborting.")
        return
    print()

    vs = VectorStore()
    persona = PersonaDB()
    engine = ForgettingEngine(vs, persona)

    # ---------- predicted I_t (analytical Eq. 7) ----------
    print(f"  {'scenario':<37} {'I_0':>5} {'days':>5} {'I_t predicted':>14}")
    print(f"  {'-'*37} {'-'*5} {'-'*5} {'-'*14}")
    for label, atom in SCENARIOS:
        vs.add(atom)
        days = (datetime.utcnow() - atom.last_reinforced_at).total_seconds() / 86400.0
        expected = atom.importance * math.exp(-s.decay_lambda * days)
        print(f"  {label:<37} {atom.importance:>5.2f} {days:>5.0f} {expected:>14.4f}")
    print()

    # ---------- run the decay sweep ----------
    print("Running decay sweep…")
    res = engine.run_decay_sweep(USER_ID)
    print(f"  -> atoms updated:      {res['updated']}")
    print(f"  -> atoms soft-deleted: {res['soft_deleted']}   "
          f"(I_t fell below floor={s.forget_floor})")
    print()

    # ---------- post-sweep state ----------
    print("=" * 78)
    print("Post-sweep state")
    print("=" * 78)
    print(f"  {'scenario':<37} {'I_0':>5} {'I_t actual':>11} {'pruned?':>10}")
    print(f"  {'-'*37} {'-'*5} {'-'*11} {'-'*10}")
    n_survived = 0
    for label, atom in SCENARIOS:
        rec = vs.coll.get(ids=[atom.id])
        if not rec["ids"]:
            print(f"  {label:<37}   (atom missing!)")
            continue
        meta = rec["metadatas"][0]
        i_t = float(meta.get("decayed_importance", 0.0))
        pruned = bool(meta.get("superseded", False))
        flag = "YES" if pruned else "no"
        if not pruned:
            n_survived += 1
        print(f"  {label:<37} {atom.importance:>5.2f} {i_t:>11.4f} {flag:>10}")

    # ---------- retrieval-side proof ----------
    # Pruned atoms must no longer surface in retrieval. Confirm that by
    # listing what retrieve() would see now.
    print()
    print(f"Retrieval check (a generic query):")
    results = vs.retrieve("tell me something about me", USER_ID, top_k=10)
    visible = {r.id for r in results}
    seeded = {a.id for _, a in SCENARIOS}
    visible_count = len(visible & seeded)
    print(f"  retrievable now: {visible_count} / {len(SCENARIOS)} seeded atoms.")
    print(f"  ({len(SCENARIOS) - visible_count} have been excluded by the soft-delete "
          f"flag set during the sweep.)")
    print()

    print("Reading the table:")
    print("  - 'I_t predicted' is computed analytically from Eq. 7; the sweep should")
    print("    produce values that match to within floating-point precision.")
    print("  - Ancient atoms (120 days) fall to ~0.001-0.002, far below the floor,")
    print("    so both get pruned regardless of their original importance — high I_0")
    print("    is not enough on its own; reinforcement is what keeps memories alive.")
    print("  - Recent atoms (≤5 days) retain ~78-90% of their original importance.")
    print("  - The middle-aged atom (30 days) degrades visibly but stays above floor —")
    print("    Eq. 7 is gradual, not a cliff.")


if __name__ == "__main__":
    main()