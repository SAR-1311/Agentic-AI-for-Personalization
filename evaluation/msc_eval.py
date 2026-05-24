"""Multi-Session Chat (MSC) — long-term behavioural consistency evaluation.

For each multi-session dialogue:
  - Stream all sessions through the agent.
  - Between sessions, optionally trigger a forgetting decay sweep.
  - At the end, ask consistency probe(s); score against earlier-session facts.

Run: python -m evaluation.msc_eval --limit 100
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from uuid import uuid4

from datasets import load_dataset
from tqdm import tqdm

from backend.core.agent import Agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _load_msc():
    """Try several known MSC mirrors; fall back gracefully."""
    for repo in ("nayohan/multi_session_chat", "msc/msc"):
        try:
            return load_dataset(repo, split="validation[:500]")
        except Exception as e:
            log.info(f"MSC mirror {repo} failed: {e}")
    raise RuntimeError("Could not load MSC. Install ParlAI fallback.")


def _consistency_score(reply: str, established_facts: list[str]) -> float:
    """Crude lexical overlap — replace with NLI for the dissertation."""
    rl = reply.lower()
    if not established_facts:
        return 0.0
    hits = sum(1 for f in established_facts if any(tok in rl for tok in f.lower().split()[:3]))
    return hits / len(established_facts)


def run(out_csv: Path, limit: int) -> None:
    ds = _load_msc()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["dialogue_idx", "n_sessions",
                                                 "consistency_score"])
        writer.writeheader()
        for idx, item in enumerate(tqdm(ds, desc="MSC", total=min(limit, len(ds)))):
            if idx >= limit:
                break
            agent = Agent()
            user_id = f"msc_{idx}_{uuid4().hex[:6]}"
            sessions = item.get("sessions") or item.get("dialog") or []
            facts: list[str] = []
            for sess in sessions:
                turns = sess.get("dialog", sess) if isinstance(sess, dict) else sess
                for t in turns:
                    text = t.get("text") if isinstance(t, dict) else str(t)
                    if text:
                        agent.chat(user_id, text)
                        facts.append(text)
            probe = "Tell me what you remember about me."
            r = agent.chat(user_id, probe)
            score = _consistency_score(r.reply, facts[:10])
            writer.writerow({"dialogue_idx": idx,
                             "n_sessions": len(sessions),
                             "consistency_score": score})
    log.info(f"Wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/results/msc.csv")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()
    run(Path(a.out), a.limit)
