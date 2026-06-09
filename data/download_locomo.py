"""Download the LoCoMo-MC10 multiple-choice benchmark.

Pulls the single JSON file we need from the HuggingFace Hub directly,
bypassing `datasets.load_dataset()` because the repo ships three JSON files
with incompatible schemas (raw locomo10.json, locomo_mc10.json, and
locomo_mc10_with_name.json) which confuses HF's auto-schema inference.

Source: https://huggingface.co/datasets/Percena/locomo-mc10
LoCoMo origin: Maharana et al. 2024, https://snap-research.github.io/locomo/

Run:  python data/download_locomo.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "locomo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "Percena/locomo-mc10"
SOURCE_FILE = "data/locomo_mc10.json"


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is required. Run: "
            "pip install huggingface_hub --break-system-packages"
        )

    print(f"Downloading {SOURCE_FILE} from {REPO_ID} …")
    cached = hf_hub_download(
        repo_id=REPO_ID,
        filename=SOURCE_FILE,
        repo_type="dataset",
    )
    print(f"  cached at: {cached}")

    target = OUT_DIR / "locomo_mc10.json"
    shutil.copy(cached, target)
    print(f"  copied to: {target}  ({target.stat().st_size / 1e6:.1f} MB)")

    # Parse and inspect.
    print("\nInspecting schema …")
    data = _load_json_or_jsonl(target)

    if not isinstance(data, list):
        raise SystemExit(f"  expected a list of rows, got {type(data).__name__}")
    print(f"  rows: {len(data)}")
    if not data:
        raise SystemExit("  dataset is empty")

    first = data[0]
    if not isinstance(first, dict):
        raise SystemExit(f"  expected dict rows, got {type(first).__name__}")
    print(f"  columns: {sorted(first.keys())}")

    # Build a human-readable preview that's safe to commit.
    def _truncate(v, n=400):
        if isinstance(v, str) and len(v) > n:
            return v[:n] + "…[truncated]"
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return f"<list of {len(v)} dicts, first keys: {sorted(v[0].keys())}>"
        if isinstance(v, list) and len(v) > 6:
            return v[:3] + ["…", f"({len(v) - 6} more)", "…"] + v[-3:]
        return v

    preview = {k: _truncate(v) for k, v in first.items()}
    info = {
        "source": REPO_ID,
        "source_file": SOURCE_FILE,
        "rows": len(data),
        "columns": sorted(first.keys()),
        "first_row_preview": preview,
        "question_types_seen": sorted({r.get("question_type", "?") for r in data}),
        "num_choices_seen": sorted({r.get("num_choices") for r in data
                                    if r.get("num_choices") is not None}),
        "num_sessions_distribution": _session_distribution(data),
    }
    preview_path = OUT_DIR / "dataset_info.json"
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"  wrote {preview_path}")
    print(f"  question types: {info['question_types_seen']}")
    print(f"  choices: {info['num_choices_seen']}")
    print(f"  session counts: {info['num_sessions_distribution']}")
    print("\nDone.")


def _session_distribution(rows):
    from collections import Counter
    c = Counter(r.get("num_sessions") for r in rows
                if r.get("num_sessions") is not None)
    return dict(sorted(c.items()))


def _load_json_or_jsonl(path: Path) -> list:
    """Parse either a JSON array or JSONL (one object per line).

    Decides by sniffing the first non-whitespace byte: '[' -> JSON array,
    otherwise -> JSONL.
    """
    with open(path, encoding="utf-8") as f:
        # Find first non-whitespace char.
        while True:
            c = f.read(1)
            if c == "" or not c.isspace():
                break
        if c == "[":
            print("  format: JSON array")
            f.seek(0)
            return json.load(f)
        # JSONL.
        print("  format: JSONL (one record per line)")
        f.seek(0)
        rows = []
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"  parse error on line {i}: {e}")
        return rows


if __name__ == "__main__":
    main()