from __future__ import annotations

import os
import tempfile
import uuid

# Isolate from real data BEFORE importing anything that touches config.
# IMPORTANT: pydantic-settings reads env vars by *Settings field name*,
# uppercased. The chroma path field is `chroma_persist_dir`, so the env var
# must be CHROMA_PERSIST_DIR — CHROMA_PATH would be silently ignored and the
# test would pollute your real ./data/chroma.
os.environ["SQLITE_PATH"]        = tempfile.mktemp(suffix=".db", prefix="retr_smoke_")
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp(prefix="retr_smoke_chroma_")

from backend.config import get_settings                              # noqa: E402
from backend.models.schemas import MemoryAtom                        # noqa: E402
from backend.storage.vector_db import VectorStore                    # noqa: E402
from evaluation.metrics import precision_at_k, retrieval_noise_ratio # noqa: E402


# Defence-in-depth: unique user id per run, so even if isolation somehow
# fails again the smoke test can't accidentally retrieve old atoms.
USER_ID = f"smoke_{uuid.uuid4().hex[:8]}"
TOP_K = 3


def _atom(text: str, importance: float, trait_type: str = "preference") -> MemoryAtom:
    return MemoryAtom(
        user_id=USER_ID,
        text=text,
        trait_type=trait_type,
        frequency=0.5, confidence=0.7, emotion=0.3,
        importance=importance, decayed_importance=importance,
    )


# (atom, set_of_relevance_tags). Decoys carry an empty set.
SEED = [
    # ---- food cluster: stable, high-importance preferences ----
    (_atom("loves chicken biryani", 0.82),                            {"food"}),
    (_atom("could eat biryani every single day", 0.70),               {"food"}),
    (_atom("enjoys Indian cuisine in general", 0.55),                 {"food"}),
    # ---- outdoor cluster: stable routines, moderate importance ----
    (_atom("goes hiking on weekends", 0.62, "routine"),               {"outdoor"}),
    (_atom("loves the Scottish highlands", 0.58),                     {"outdoor"}),
    (_atom("owns a pair of trail-running shoes", 0.40, "fact"),       {"outdoor"}),
    # ---- ADVERSARIAL food-adjacent decoys (lexically close, low signal) ----
    # Pure cosine should rank these high (shared food vocabulary); the blend
    # should demote them by importance, recovering precision on food queries.
    (_atom("tried sushi once and didn't really enjoy it", 0.25, "dislike"), set()),
    (_atom("considered going vegan briefly last year", 0.22, "goal"),       set()),
    (_atom("watched a cooking show on Sunday", 0.20, "fact"),               set()),
    # ---- ADVERSARIAL outdoor-adjacent decoys ----
    (_atom("thought about trying yoga at some point", 0.22, "goal"),        set()),
    (_atom("walked through the park yesterday", 0.20, "fact"),              set()),
    # ---- distant decoys (semantically far; cosine already filters these) ----
    (_atom("studies Informatics at Edinburgh", 0.50, "occupation"),         set()),
    (_atom("uses Python and TypeScript daily", 0.45, "routine"),            set()),
    (_atom("has a cat called Mochi", 0.42, "fact"),                         set()),
    (_atom("is learning Mandarin slowly", 0.28, "goal"),                    set()),
]

QUERIES = [
    ("what should I cook tonight?",            "food"),
    ("any ideas for a weekend activity?",      "outdoor"),
    ("recommend a restaurant for dinner",      "food"),
]


def _retrieve(vs: VectorStore, query: str, sim_w: float, imp_w: float, k: int):
    """retrieve() with a temporary weight override (restored on exit)."""
    orig = (vs.sim_w, vs.imp_w)
    vs.sim_w, vs.imp_w = sim_w, imp_w
    try:
        return vs.retrieve(query, USER_ID, top_k=k)
    finally:
        vs.sim_w, vs.imp_w = orig


def main() -> None:
    s = get_settings()
    print("=" * 78)
    print("Retrieval smoke test")
    print("=" * 78)
    print(f"  user_id            = {USER_ID}")
    print(f"  chroma_persist_dir = {s.chroma_persist_dir}")
    print(f"  sqlite_path        = {s.sqlite_path}")
    if "retr_smoke_" not in s.chroma_persist_dir:
        print()
        print("  !!! WARNING: chroma_persist_dir does NOT look like a temp path.")
        print("  !!! Isolation may have failed. Aborting to avoid polluting data.")
        return
    print()

    vs = VectorStore()

    label: dict[str, set[str]] = {}
    for atom, tags in SEED:
        vs.add(atom)
        label[atom.id] = tags

    configs = [
        ("Pure cosine      (1.00 / 0.00)", 1.0, 0.0),
        ("Cosine-favoured  (0.90 / 0.10)", 0.9, 0.1),
        ("Cosine-favoured  (0.80 / 0.20)", 0.8, 0.2),
        ("Balanced         (0.70 / 0.30)", 0.7, 0.3),
        ("Agentic blend    (0.60 / 0.40)", 0.6, 0.4),
        ("Importance-heavy (0.40 / 0.60)", 0.4, 0.6),
    ]

    print("=" * 78)
    print(f"Retrieval smoke test  —  N={len(SEED)} atoms,  top-{TOP_K}")
    print("=" * 78)
    agg: dict[str, dict[str, list[float]]] = {n: {"p": [], "r": []} for n, *_ in configs}

    for query, target_tag in QUERIES:
        relevant_ids = {aid for aid, tags in label.items() if target_tag in tags}
        print(f"\nQUERY: \"{query}\"")
        print(f"   relevant tag = '{target_tag}'   "
              f"({len(relevant_ids)} relevant atoms in store)")
        print(f"   {'config':<32} {'P@K':>8} {'RNR':>8}    top-K texts")
        for name, sw, iw in configs:
            results = _retrieve(vs, query, sw, iw, k=TOP_K)
            retr_ids = [r.id for r in results]
            p = precision_at_k(retr_ids, relevant_ids, TOP_K)
            rnr = retrieval_noise_ratio(retr_ids, relevant_ids)
            agg[name]["p"].append(p)
            agg[name]["r"].append(rnr)
            preview = ", ".join(
                ("✓" if rid in relevant_ids else "✗") + r.text[:28]
                for rid, r in zip(retr_ids, results)
            )
            print(f"   {name:<32} {p:>8.3f} {rnr:>8.3f}    {preview}")

    print("\n" + "=" * 78)
    print(f"AVERAGES over {len(QUERIES)} queries")
    print("=" * 78)
    print(f"   {'config':<32} {'P@K (↑)':>10} {'RNR (↓)':>10}")
    for name, *_ in configs:
        ps, rs = agg[name]["p"], agg[name]["r"]
        avg_p = sum(ps) / len(ps)
        avg_r = sum(rs) / len(rs)
        print(f"   {name:<32} {avg_p:>10.3f} {avg_r:>10.3f}")
    print()
    print("Read the table:")
    print("   - Precision@K ↑ better — fraction of top-K that are relevant.")
    print("   - RNR        ↓ better — fraction of top-K that are noise.")
    print("   - ✓/✗ next to each text marks relevant vs noise in the top-K.")


if __name__ == "__main__":
    main()