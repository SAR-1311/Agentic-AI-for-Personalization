"""Master evaluation runner — produces all dissertation numbers in one go.

Run: python -m evaluation.run_all --personamem-limit 50 --msc-limit 30
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personamem-data", default="data/personamem")
    ap.add_argument("--personamem-limit", type=int, default=None)
    ap.add_argument("--msc-limit", type=int, default=50)
    ap.add_argument("--skip-personamem", action="store_true")
    ap.add_argument("--skip-msc", action="store_true")
    ap.add_argument("--skip-life-transition", action="store_true")
    a = ap.parse_args()

    Path("evaluation/results").mkdir(parents=True, exist_ok=True)

    if not a.skip_personamem:
        log.info("=== PersonaMem ===")
        from evaluation.personamem_eval import evaluate as pm
        try:
            pm(Path(a.personamem_data),
               Path("evaluation/results/personamem.csv"),
               limit=a.personamem_limit)
        except Exception as e:
            log.warning(f"PersonaMem skipped: {e}")

    if not a.skip_msc:
        log.info("=== MSC ===")
        from evaluation.msc_eval import run as msc
        try:
            msc(Path("evaluation/results/msc.csv"), a.msc_limit)
        except Exception as e:
            log.warning(f"MSC skipped: {e}")

    if not a.skip_life_transition:
        log.info("=== Life-Transition ===")
        from evaluation.life_transition import run as lt
        try:
            lt(Path("evaluation/results/life_transition.json"))
        except Exception as e:
            log.warning(f"Life-Transition skipped: {e}")

    log.info("All evaluations complete. See evaluation/results/")


if __name__ == "__main__":
    main()
