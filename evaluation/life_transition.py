"""Life-Transition synthetic stress test — measures Adaptation Latency (Eq. 6).

Procedure (proposal §3.3):
  1. Sessions 1..N-1: user expresses a stable preference (e.g. vegetarian).
  2. Session N:       user reverses the preference.
  3. Sessions N+1..:  probe questions; measure how many turns until the agent
     stops giving the OLD preference's answers.

Run: python -m evaluation.life_transition
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

from backend.core.agent import Agent
from evaluation.metrics import adaptation_latency

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


SCENARIOS = [
    {
        "name": "vegetarian_to_omnivore",
        "old_pref_keywords": ["meat", "chicken", "beef", "fish", "pork"],   # appearance = adapted
        "old_pref_label": "vegetarian",
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
        "old_pref_keywords": ["software", "code", "developer", "engineer"],
        "old_pref_label": "software_engineer",
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
        "old_pref_keywords": ["edinburgh", "scotland", "scottish"],
        "old_pref_label": "edinburgh_resident",
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


def _adapted(reply: str, scenario: dict) -> bool:
    """Heuristic: agent has 'adapted' if response mentions the new context
    or AVOIDS the old preference's keywords."""
    rl = reply.lower()
    # If the old preference is dietary 'vegetarian', adaptation = mentions meat etc.
    return any(kw in rl for kw in scenario["old_pref_keywords"])


def run(out_json: Path = Path("evaluation/results/life_transition.json")) -> dict:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for sc in SCENARIOS:
        agent = Agent()
        user_id = f"lt_{sc['name']}_{uuid4().hex[:6]}"
        log.info(f"Running scenario: {sc['name']}  user={user_id}")

        # Establish phase — simulate 3 short sessions
        for utt in sc["establish"]:
            agent.chat(user_id, utt)

        # Transition turn (this is t_change)
        for utt in sc["transition"]:
            agent.chat(user_id, utt)
        t_change = 0  # we treat probes as a 0-indexed window after transition

        # Probe phase
        decisions: list[bool] = []
        replies: list[dict] = []
        for q in sc["probes"]:
            r = agent.chat(user_id, q)
            decisions.append(_adapted(r.reply, sc))
            replies.append({"q": q, "reply": r.reply, "adapted": decisions[-1]})

        AL = adaptation_latency(t_change, decisions)
        results.append({
            "scenario": sc["name"],
            "adaptation_latency_turns": AL,
            "decisions": decisions,
            "replies": replies,
        })
        log.info(f"  Adaptation Latency = {AL} turns")

    summary = {"scenarios": results,
               "mean_AL": _mean_skipping_none([r["adaptation_latency_turns"] for r in results])}
    out_json.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {out_json}")
    return summary


def _mean_skipping_none(xs: list[int | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


if __name__ == "__main__":
    run()
