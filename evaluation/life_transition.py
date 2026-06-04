"""Life-Transition synthetic stress test — measures Adaptation Latency (Eq. 6).

Procedure (proposal §3.3):
  1. Establish phase: user expresses a stable preference (e.g. vegetarian)
     across several turns.
  2. Transition turn: user reverses the preference.
  3. Probe phase: ask preference-sensitive questions; measure how many probes
     are needed before the agent stops giving answers tied to the OLD preference.

Adaptation criterion (per scenario):
    adapted := the probe's reply does NOT contain any of the OLD preference's
               keywords. This is the cleanest, most defensible signal: if the
               agent is still recommending Edinburgh walks after the user moved
               to Berlin, it has not adapted; if it gives any reply that avoids
               the old keywords (Berlin-aware OR generic), it has.

A separate `new_keywords_hit` field is logged for inspection but is NOT used in
the adaptation_latency calculation, so the metric remains conservative.

Run: python -m evaluation.life_transition
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from uuid import uuid4

# Isolate from real data BEFORE backend imports.
os.environ["SQLITE_PATH"]        = tempfile.mktemp(suffix=".db", prefix="lt_eval_")
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp(prefix="lt_eval_chroma_")

from backend.config import get_settings                     # noqa: E402
from backend.core.agent import Agent                        # noqa: E402
from evaluation.metrics import adaptation_latency           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


SCENARIOS = [
    {
        "name": "vegetarian_to_omnivore",
        # Keywords characteristic of the OLD state (vegetarian).
        # Adapted = NONE of these appear in the reply.
        "old_keywords": ["vegetarian", "vegan", "tofu", "lentils", "plant-based"],
        # New-state keywords are LOGGED but not used in the metric.
        "new_keywords": ["meat", "chicken", "beef", "fish", "pork", "salmon"],
        "establish": [
            "I've been vegetarian for years.",
            "I always cook tofu and lentils for dinner.",
            "I never eat meat.",
        ],
        "transition": [
            "Quick update: my doctor told me I need more protein, so I started "
            "eating chicken and fish again last week.",
        ],
        "probes": [
            "What should I cook tonight?",
            "Suggest a high-protein dinner.",
            "Recommend a quick weekday meal.",
            "What's a good restaurant for me to try?",
        ],
    },
    {
        "name": "career_change",
        "old_keywords": ["software engineer", "software engineering", "engineer",
                         "debugging", "code review", "python"],
        "new_keywords": ["data scientist", "data science", "statistics",
                         "machine learning", " ml ", "model"],
        "establish": [
            "I work as a software engineer at a fintech startup.",
            "My day is mostly debugging and code reviews.",
            "I love writing Python.",
        ],
        "transition": [
            "I quit my engineering job and started a new role as a data scientist.",
        ],
        "probes": [
            "What kind of online course should I take to grow in my career?",
            "Recommend a podcast for me.",
            "What's a good book for someone in my field?",
            "Help me plan my workweek.",
        ],
    },
    {
        "name": "city_move",
        "old_keywords": ["edinburgh", "scotland", "scottish", "arthur's seat",
                         "old town", "scottish highlands"],
        "new_keywords": ["berlin", "germany", "german", "brandenburg",
                         "kreuzberg", "mitte"],
        "establish": [
            "I live in Edinburgh and love the old town.",
            "Most weekends I walk Arthur's Seat.",
        ],
        "transition": [
            "I just moved to Berlin for a new job — Edinburgh days are behind me.",
        ],
        "probes": [
            "Recommend a weekend activity for me.",
            "What's a nice neighbourhood for me to explore?",
            "Suggest a local food I should try.",
            "Where could I go for a Sunday walk?",
        ],
    },
]


def _adapted(reply: str, scenario: dict) -> tuple[bool, dict]:
    """Agent has adapted iff the reply avoids the OLD preference's keywords.

    Returns (is_adapted, diagnostics). Diagnostics records which old/new
    keywords actually appeared, for downstream inspection in the results JSON.
    """
    rl = (reply or "").lower()
    old_hits = [kw for kw in scenario["old_keywords"] if kw in rl]
    new_hits = [kw for kw in scenario["new_keywords"] if kw in rl]
    is_adapted = len(old_hits) == 0
    return is_adapted, {"old_keywords_hit": old_hits, "new_keywords_hit": new_hits}


def run(out_json: Path = Path("evaluation/results/life_transition.json")) -> dict:
    s = get_settings()
    log.info("=" * 72)
    log.info("Life-Transition synthetic evaluation")
    log.info("=" * 72)
    log.info(f"  chroma_persist_dir = {s.chroma_persist_dir}")
    log.info(f"  sqlite_path        = {s.sqlite_path}")
    if "lt_eval_" not in s.chroma_persist_dir:
        log.error("  !!! isolation failed — aborting.")
        return {}
    log.info("")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for sc in SCENARIOS:
        # Use a fresh user_id per scenario so memories don't bleed between them.
        agent = Agent()
        user_id = f"lt_{sc['name']}_{uuid4().hex[:6]}"
        log.info(f"[{sc['name']}] user={user_id}")

        # Establish: ingest-only, no reply needed.
        for utt in sc["establish"]:
            agent.chat(user_id, utt, generate_reply=False)

        # Transition: ingest-only too — the user's statement is what matters,
        # not how the agent acknowledges it.
        for utt in sc["transition"]:
            agent.chat(user_id, utt, generate_reply=False)

        # Probe: real replies, adaptation decision per probe.
        decisions: list[bool] = []
        replies: list[dict] = []
        for q in sc["probes"]:
            r = agent.chat(user_id, q, generate_reply=True)
            adapted, diag = _adapted(r.reply, sc)
            decisions.append(adapted)
            replies.append({
                "q": q,
                "reply": r.reply,
                "adapted": adapted,
                **diag,
            })
            tag = "[OK]" if adapted else "[old]"
            log.info(f"  {tag} q={q!r}  old_hit={diag['old_keywords_hit']}  "
                     f"new_hit={diag['new_keywords_hit']}")

        AL = adaptation_latency(0, decisions)  # t_change is the moment before probes
        results.append({
            "scenario": sc["name"],
            "adaptation_latency_turns": AL,
            "decisions": decisions,
            "replies": replies,
        })
        log.info(f"  -> Adaptation Latency = {AL} turns\n")

    mean_AL = _mean_skipping_none([r["adaptation_latency_turns"] for r in results])
    summary = {"scenarios": results, "mean_AL": mean_AL}
    out_json.write_text(json.dumps(summary, indent=2))

    log.info("=" * 72)
    log.info(f"SUMMARY")
    log.info("=" * 72)
    for r in results:
        log.info(f"  {r['scenario']:<28} AL = {r['adaptation_latency_turns']} turns")
    log.info(f"  {'mean':<28} AL = {mean_AL}")
    log.info(f"\n  wrote {out_json}")
    return summary


def _mean_skipping_none(xs):
    vals = [x for x in xs if x is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


if __name__ == "__main__":
    run()