"""LoCoMo-MC10 evaluation runner.

For each sampled multiple-choice question:
  1. Reads the question's pre-attached `haystack_sessions` (list of sessions,
     each a list of {role, content} dicts) from data/locomo/locomo_mc10.json.
  2. Builds the agent's memory by feeding *user*-role turns from those sessions
     through agent.chat(..., generate_reply=False) — gatekeeper, synthesis, and
     storage all run; reply generation is skipped to keep the run cheap.
  3. Builds an MCQ prompt from the `question` + `choices` (10 options labelled
     a-j) and asks the agent (now reply-generating) to pick one option.
  4. Parses the chosen letter, maps to choice index, compares to
     `correct_choice_index`.
  5. Reports per-question results and per-question-type accuracy.

LoCoMo: Maharana et al. 2024. MC10 conversion: Percena/locomo-mc10 on HF.

Usage:
    python -m evaluation.locomo_eval                              # n=3 shortest
    python -m evaluation.locomo_eval --n 30 --random              # n=30 random
    python -m evaluation.locomo_eval --n 30 --random --output runs/locomo_n30.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

# Isolate from real data BEFORE backend import.
os.environ["SQLITE_PATH"]        = tempfile.mktemp(suffix=".db", prefix="lc_eval_")
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp(prefix="lc_eval_chroma_")

import pandas as pd                                     # noqa: E402

from backend.config import get_settings                 # noqa: E402
from backend.core.agent import Agent                    # noqa: E402

DATA_FILE = Path("data/locomo/locomo_mc10.json")

LETTERS = "abcdefghij"  # 10 choices
LETTER_RE = re.compile(r"\(([a-jA-J])\)")


def _load_rows(path: Path) -> list[dict]:
    """Stream-load JSONL; tolerant of either a single JSON array or one
    record per line, same as the downloader's sniffer."""
    print(f"  loading {path} ...", end=" ", flush=True)
    with open(path, encoding="utf-8") as f:
        while True:
            c = f.read(1)
            if c == "" or not c.isspace():
                break
        if c == "[":
            f.seek(0)
            rows = json.load(f)
        else:
            f.seek(0)
            rows = [json.loads(line) for line in f if line.strip()]
    print(f"{len(rows)} rows.")
    return rows


def _format_mcq(question: str, choices: list[str]) -> str:
    bullet = "\n".join(f"  ({LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n"
        f"Choose the single best option from:\n{bullet}\n\n"
        "Reply with ONLY the option letter in parentheses, e.g. (c). "
        "No explanation, no extra text."
    )


def _parse_letter_to_index(text: str, n_choices: int = 10) -> int | None:
    if not text:
        return None
    allowed = LETTERS[:n_choices]
    m = LETTER_RE.search(text)
    if m:
        letter = m.group(1).lower()
        if letter in allowed:
            return allowed.index(letter)
    # Fallback: bare leading letter
    s = text.strip().lower()
    if s and s[0] in allowed:
        return allowed.index(s[0])
    return None


def run(n: int, randomize: bool, output_path: str | None) -> None:
    s = get_settings()
    print("=" * 78)
    print("LoCoMo-MC10 evaluation")
    print("=" * 78)
    print(f"  chroma_persist_dir = {s.chroma_persist_dir}")
    print(f"  sqlite_path        = {s.sqlite_path}")
    if "lc_eval_" not in s.chroma_persist_dir:
        print("\n  !!! isolation may have failed — aborting.")
        return
    print()

    if not DATA_FILE.exists():
        print(f"  ERROR: dataset not found at {DATA_FILE}.")
        print("  Run: python data/download_locomo.py")
        sys.exit(1)

    rows = _load_rows(DATA_FILE)
    df = pd.DataFrame(rows)
    print(f"  question types: {sorted(df['question_type'].unique().tolist())}")
    print(f"  session count range: "
          f"{df['num_sessions'].min()}–{df['num_sessions'].max()}")
    print()

    # Sample. Shortest-context first for pilot, random for the real run.
    if randomize:
        sampled = df.sample(n=min(n, len(df)), random_state=42)
    else:
        sampled = df.sort_values("num_sessions").head(n)
    sampled = sampled.reset_index(drop=True)
    print(f"  sampled {len(sampled)} questions "
          f"({'random' if randomize else 'shortest-sessions first'})")
    print()

    agent = Agent()
    results = []
    correct = 0
    t_start = time.time()

    for i, row in sampled.iterrows():
        qid = row["question_id"]
        qtype = row["question_type"]
        n_sess = int(row["num_sessions"])
        print(f"[{i+1}/{len(sampled)}] q={qid} type={qtype} "
              f"n_sessions={n_sess}")

        # Flatten haystack_sessions, then keep only user-role turns.
        haystack = row["haystack_sessions"] or []
        all_msgs: list[dict] = []
        for sess in haystack:
            if isinstance(sess, list):
                all_msgs.extend(sess)
        user_turns = [m["content"] for m in all_msgs
                      if isinstance(m, dict) and m.get("role") == "user"]

        eval_uid = f"lc_{qid}"
        ingested_ok = 0
        ingest_errors = 0
        first_errors: list[str] = []
        t0 = time.time()
        for j, utext in enumerate(user_turns):
            try:
                agent.chat(eval_uid, utext, generate_reply=False)
                ingested_ok += 1
            except Exception as e:
                ingest_errors += 1
                if len(first_errors) < 3:
                    first_errors.append(str(e).splitlines()[0][:120])
                continue
        ingest_secs = time.time() - t0

        # Ask the MCQ.
        mcq = _format_mcq(row["question"], list(row["choices"]))
        t0 = time.time()
        try:
            resp = agent.chat(eval_uid, mcq, generate_reply=True)
            reply = resp.reply or ""
        except Exception as e:
            print(f"    answer error: {e}")
            continue
        answer_secs = time.time() - t0

        picked_idx = _parse_letter_to_index(reply, n_choices=int(row.get("num_choices", 10)))
        gold_idx = int(row["correct_choice_index"])
        is_correct = picked_idx == gold_idx
        if is_correct:
            correct += 1

        flag = "OK" if is_correct else "XX"
        picked_letter = LETTERS[picked_idx] if picked_idx is not None else "?"
        gold_letter = LETTERS[gold_idx]
        print(f"    [{flag}] ingested {ingested_ok}/{len(user_turns)} user turns "
              f"({ingest_secs:.1f}s; {ingest_errors} errors) "
              f"| answer {answer_secs:.1f}s "
              f"| picked=({picked_letter}) gold=({gold_letter})")

        results.append({
            "question_id": qid,
            "question_type": qtype,
            "num_sessions": n_sess,
            "user_turns_total": len(user_turns),
            "user_turns_ingested": ingested_ok,
            "ingest_errors": ingest_errors,
            "picked_index": picked_idx,
            "picked_letter": picked_letter,
            "gold_index": gold_idx,
            "gold_letter": gold_letter,
            "correct": is_correct,
            "reply": reply,
            "ingest_seconds": round(ingest_secs, 2),
            "answer_seconds": round(answer_secs, 2),
        })

    total_secs = time.time() - t_start
    answered = len(results)
    accuracy = correct / answered if answered else 0.0
    print()
    print("=" * 78)
    print(f"SUMMARY  ->  {correct}/{answered}  ({accuracy:.1%})  "
          f"in {total_secs/60:.1f} min")
    print("=" * 78)
    if answered:
        per_type: dict[str, list[bool]] = defaultdict(list)
        for r in results:
            per_type[r["question_type"]].append(r["correct"])
        print("  by question_type:")
        for qt, marks in sorted(per_type.items()):
            print(f"    {qt:<22} {sum(marks)}/{len(marks)}  "
                  f"({sum(marks)/len(marks):.1%})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "n": answered,
                       "total_seconds": round(total_secs, 1),
                       "results": results}, f, indent=2)
        print(f"\n  wrote {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3,
                   help="number of questions (default 3 for a smoke test)")
    p.add_argument("--random", action="store_true",
                   help="random sample (default: shortest-sessions first)")
    p.add_argument("--output", type=str, default=None,
                   help="path to write JSON results")
    args = p.parse_args()
    run(args.n, args.random, args.output)


if __name__ == "__main__":
    main()