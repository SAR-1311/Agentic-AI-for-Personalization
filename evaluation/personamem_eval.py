from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Isolate from real data BEFORE any backend import.
os.environ["SQLITE_PATH"]        = tempfile.mktemp(suffix=".db", prefix="pm_eval_")
os.environ["CHROMA_PERSIST_DIR"] = tempfile.mkdtemp(prefix="pm_eval_chroma_")

import pandas as pd                                     # noqa: E402

from backend.config import get_settings                 # noqa: E402
from backend.core.agent import Agent                    # noqa: E402

DATA_DIR = Path("data/personamem")
QUESTIONS_CSV = DATA_DIR / "questions_32k.csv"
CONTEXTS_JSONL = DATA_DIR / "shared_contexts_32k.jsonl"

LETTER_RE = re.compile(r"\(([a-dA-D])\)")


def _build_context_index() -> dict[str, list[dict]]:
    file_size = CONTEXTS_JSONL.stat().st_size
    print(f"  indexing {CONTEXTS_JSONL.name} ({file_size:,} bytes) ...")

    by_hash: dict[str, list[dict]] = {}
    with open(CONTEXTS_JSONL, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                # Standard PersonaMem v1 shape: {hash: [messages]}.
                for k, v in rec.items():
                    if isinstance(v, list):
                        by_hash[k] = v
            elif isinstance(rec, list):
                # Defensive fallback: bare list, key by line index.
                by_hash[str(i)] = rec

    print(f"  -> loaded {len(by_hash)} contexts")
    if by_hash:
        first_key = next(iter(by_hash))
        first_val = by_hash[first_key]
        print(f"  -> sample key: {first_key[:32]}...")
        print(f"  -> first context has {len(first_val)} messages; "
              f"first role={first_val[0].get('role','?') if first_val else '-'}")
    return by_hash


def _ctx_for_question(row, ctx_by_hash):
    return ctx_by_hash.get(str(row["shared_context_id"]))


def _parse_letter(text):
    if not text:
        return None
    m = LETTER_RE.search(text)
    if m:
        return f"({m.group(1).lower()})"
    s = text.strip().lower()
    if s and s[0] in "abcd":
        return f"({s[0]})"
    return None


def _format_mcq(question, options_json):
    try:
        options = json.loads(options_json)
    except (json.JSONDecodeError, TypeError):
        options = [s.strip() for s in str(options_json).strip("[]").split('", "')]
    bullet = "\n".join(f"  {o}" for o in options)
    return (
        f"{question}\n\n"
        f"Choose the single best option from:\n{bullet}\n\n"
        "Reply with ONLY the option letter in parentheses, e.g. (c). "
        "No explanation, no extra text."
    )


def run(n, randomize, output_path):
    s = get_settings()
    print("=" * 78)
    print("PersonaMem v1 (32k) evaluation")
    print("=" * 78)
    print(f"  chroma_persist_dir = {s.chroma_persist_dir}")
    print(f"  sqlite_path        = {s.sqlite_path}")
    if "pm_eval_" not in s.chroma_persist_dir:
        print("\n  !!! WARNING: isolation may have failed. Aborting.")
        return
    print()

    if not QUESTIONS_CSV.exists() or not CONTEXTS_JSONL.exists():
        print(f"  ERROR: dataset not found at {DATA_DIR}.")
        print("  Run: python data/download_personamem.py")
        sys.exit(1)

    print(f"  loading {QUESTIONS_CSV.name} ...", end=" ", flush=True)
    df = pd.read_csv(QUESTIONS_CSV)
    print(f"{len(df)} questions.")

    ctx_by_hash = _build_context_index()
    print()

    if randomize:
        sampled = df.sample(n=min(n, len(df)), random_state=42)
    else:
        sampled = df.sort_values("context_length_in_tokens").head(n)
    sampled = sampled.reset_index(drop=True)
    print(f"  sampled {len(sampled)} questions "
          f"({'random' if randomize else 'shortest-context first'})")
    print()

    agent = Agent()

    results = []
    correct = 0
    t_start = time.time()

    for i, row in sampled.iterrows():
        print(f"[{i+1}/{len(sampled)}] q={row['question_id'][:8]} "
              f"persona={row['persona_id']} type={row['question_type']} "
              f"ctx_tokens={row['context_length_in_tokens']}")
        ctx_msgs = _ctx_for_question(row, ctx_by_hash)
        if ctx_msgs is None:
            print("    SKIP: context not resolvable")
            continue
        end = int(row["end_index_in_shared_context"])
        ctx_msgs = ctx_msgs[:end]
        user_turns = [m["content"] for m in ctx_msgs
                      if isinstance(m, dict) and m.get("role") == "user"]
        eval_uid = f"pm_{row['persona_id']}_{row['question_id'][:8]}"
        t0 = time.time()
        ingested_ok = 0
        ingest_errors = 0
        for j, utext in enumerate(user_turns):
            try:
                agent.chat(eval_uid, utext, generate_reply=False)
                ingested_ok += 1
            except Exception as e:
                ingest_errors += 1
                if ingest_errors <= 3:    # avoid log spam on bad runs
                    print(f"    ingest warning at turn {j}: "
                          f"{type(e).__name__}: {str(e)[:120]}")
                continue   # one bad atom must not kill the whole context
        ingest_secs = time.time() - t0

        mcq = _format_mcq(row["user_question_or_message"], row["all_options"])
        t0 = time.time()
        try:
            resp = agent.chat(eval_uid, mcq, generate_reply=True)
            reply = resp.reply
        except Exception as e:
            print(f"    answer error: {e}")
            continue
        answer_secs = time.time() - t0
        picked = _parse_letter(reply)
        gold = str(row["correct_answer"]).strip().lower()
        is_correct = picked == gold
        if is_correct:
            correct += 1
        flag = "OK" if is_correct else "XX"
        print(f"    [{flag}] ingested {ingested_ok}/{len(user_turns)} user turns "
              f"({ingest_secs:.1f}s; {ingest_errors} errors) "
              f"| answer {answer_secs:.1f}s | picked={picked} gold={gold}")
        results.append({
            "question_id": row["question_id"],
            "persona_id": int(row["persona_id"]),
            "question_type": row["question_type"],
            "topic": row["topic"],
            "context_tokens": int(row["context_length_in_tokens"]),
            "user_turns_total": len(user_turns),
            "user_turns_ingested": ingested_ok,
            "ingest_errors": ingest_errors,
            "picked": picked,
            "gold": gold,
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
        per_type = {}
        for r in results:
            per_type.setdefault(r["question_type"], []).append(r["correct"])
        print("  by question_type:")
        for qt, marks in per_type.items():
            print(f"    {qt:<35} {sum(marks)}/{len(marks)}  "
                  f"({sum(marks)/len(marks):.1%})")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"accuracy": accuracy, "n": answered,
                       "total_seconds": round(total_secs, 1),
                       "results": results}, f, indent=2)
        print(f"\n  wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--random", action="store_true")
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()
    run(args.n, args.random, args.output)


if __name__ == "__main__":
    main()