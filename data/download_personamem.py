"""Download the PersonaMem dataset from Hugging Face.

Adjust REPO_ID once you confirm the latest official release name.
"""
import os
import sys
from pathlib import Path

REPO_CANDIDATES = [
    # Try several candidate repo IDs — update the first match that works.
    "bowen-upenn/PersonaMem",
    "PersonaMem/PersonaMem-v2",
    "PersonaMem/PersonaMem",
]

OUT_DIR = Path("data/personamem")


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for repo in REPO_CANDIDATES:
        try:
            print(f"Trying {repo}…")
            snapshot_download(repo_id=repo, repo_type="dataset",
                              local_dir=str(OUT_DIR))
            print(f"✅ Downloaded {repo} → {OUT_DIR}")
            return
        except Exception as e:
            print(f"  ❌ {e}")
            continue
    print("\n⚠️  None of the candidate repos worked. Manually find PersonaMem on HF:")
    print("    https://huggingface.co/datasets?search=PersonaMem")
    print("Then update REPO_CANDIDATES in this script.")


if __name__ == "__main__":
    main()
