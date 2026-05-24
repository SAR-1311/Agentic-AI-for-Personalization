"""PersonaMem(-v2) benchmark runner.

Streams each dialogue through the agent, then asks the probe questions and
scores the agent's persona JSON against ground truth.

Usage:
    python -m evaluation.personamem_eval --data data/personamem --out evaluation/results/personamem.csv

NOTE: The exact field names in the public PersonaMem release may differ from
those used here. Inspect the dataset first and adjust the `_extract_*` helpers
below accordingly. The proposal indicates implicit + explicit trait annotations
and probe questions per session.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from tqdm import tqdm

from backend.core.agent import Agent
from evaluation.metrics import precision_at_k, recall_at_k, trait_f1

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _load_personamem(data_dir: Path) -> list[dict]:
    """Load the dataset. Adjust to whichever JSON / parquet format ships."""
    files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No PersonaMem files found in {data_dir}. "
                                f"Run data/download_personamem.py first.")
    items: list[dict] = []
    for f in files:
        if f.suffix == ".jsonl":
            with f.open() as fh:
                items.extend(json.loads(line) for line in fh if line.strip())
        else:
            with f.open() as fh:
                payload = json.load(fh)
                items.extend(payload if isinstance(payload, list) else [payload])
    return items


def _extract_dialogue(item: dict) -> list[dict]:
    """Adapt this to the actual schema. Expected output: [{role, text}, …]."""
    if "dialogue" in item:
        return item["dialogue"]
    if "messages" in item:
        return item["messages"]
    if "turns" in item:
        return [{"role": t.get("role", "user"), "text": t.get("text", t.get("content", ""))}
                for t in item["turns"]]
    return []


def _extract_ground_truth(item: dict) -> list[str]:
    """Ground-truth trait *values* as plain strings."""
    for key in ("persona_traits", "traits", "ground_truth_traits", "persona"):
        if key in item:
            v = item[key]
            if isinstance(v, list):
                return [str(x) if not isinstance(x, dict) else x.get("value", str(x)) for x in v]
            if isinstance(v, dict):
                return [str(x) for x in v.values()]
    return []


def _extract_probes(item: dict) -> list[dict]:
    return item.get("probe_questions") or item.get("probes") or []


def evaluate(data_dir: Path, out_csv: Path, limit: int | None = None) -> None:
    items = _load_personamem(data_dir)
    if limit:
        items = items[:limit]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["item_idx", "user_id", "n_turns", "n_gt_traits",
                  "n_pred_traits", "precision", "recall", "f1",
                  "p_at_5", "r_at_5", "probe_correct", "probe_total"]

    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for idx, item in enumerate(tqdm(items, desc="PersonaMem")):
            agent = Agent()                            # fresh per item — isolation
            user_id = f"pm_{idx}_{uuid4().hex[:8]}"
            dialogue = _extract_dialogue(item)
            gt_traits = _extract_ground_truth(item)
            probes = _extract_probes(item)

            # Stream user turns
            for turn in dialogue:
                if turn.get("role") == "user":
                    try:
                        agent.chat(user_id, turn.get("text") or turn.get("content", ""))
                    except Exception as e:
                        log.warning(f"chat failed turn={turn}: {e}")

            # Pull final persona traits from the agent
            pred = [t["value"] for t in agent.memory.persona.get_active(user_id)]

            # Trait-level scoring (string matching — see proposal §3.3)
            f1m = trait_f1(pred, gt_traits)

            # Retrieval scoring on probes (if probes ship with relevant_ids
            # we use them; else skip).
            p5 = r5 = 0.0
            for probe in probes:
                relevant_ids = set(probe.get("relevant_memory_ids", []))
                if relevant_ids:
                    retrieved = agent.memory.retrieve(user_id, probe.get("question", ""))
                    rids = [m.id for m in retrieved]
                    p5 += precision_at_k(rids, relevant_ids, 5)
                    r5 += recall_at_k(rids, relevant_ids, 5)
            denom = max(1, len([p for p in probes if p.get("relevant_memory_ids")]))
            p5, r5 = p5 / denom, r5 / denom

            # Probe MCQ accuracy (if `answer` and `options` provided)
            correct = total = 0
            for probe in probes:
                if "answer" in probe and "options" in probe:
                    total += 1
                    resp = agent.chat(user_id,
                        probe["question"] + "\nOptions:\n" +
                        "\n".join(f"- {o}" for o in probe["options"]) +
                        "\nReply with just the chosen option.").reply.lower()
                    if str(probe["answer"]).lower() in resp:
                        correct += 1

            writer.writerow({
                "item_idx": idx,
                "user_id": user_id,
                "n_turns": len(dialogue),
                "n_gt_traits": len(gt_traits),
                "n_pred_traits": len(pred),
                "precision": f1m["precision"],
                "recall": f1m["recall"],
                "f1": f1m["f1"],
                "p_at_5": p5,
                "r_at_5": r5,
                "probe_correct": correct,
                "probe_total": total,
            })

    log.info(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/personamem")
    ap.add_argument("--out", default="evaluation/results/personamem.csv")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N items (for fast iteration).")
    a = ap.parse_args()
    evaluate(Path(a.data), Path(a.out), limit=a.limit)
