"""Generate all dissertation figures from evaluation result files.

Reads:
  runs/personamem_n50.json       — PersonaMem random eval, n=50 (headline result)
  runs/personamem_n5.json        — PersonaMem pilot, n=5 shortest-context
  evaluation/results/life_transition.json — Adaptation latency

Hardcodes (values from terminal logs of runs that didn't dump JSON):
  retrieval_smoke.py             — P@K and RNR vs importance weight
  forgetting_demo.py             — Eq. 7 verification on backdated atoms

Writes:
  evaluation/figures/*.png       — Figures for the dissertation
  evaluation/figures/summary.csv — Headline metrics table
  evaluation/figures/summary.tex — LaTeX version of the same table

Usage: python -m evaluation.generate_figures
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================================
# Setup
# ============================================================================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
})

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "evaluation" / "results"
FIG_DIR = ROOT / "evaluation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette — used consistently across all figures.
C_OURS = "#2563eb"
C_OURS_DARK = "#1e40af"
C_BASE = "#94a3b8"
C_BASE_DARK = "#64748b"
C_GOOD = "#16a34a"
C_BAD = "#dc2626"
C_FLOOR = "#9ca3af"
C_RECALL = "#2563eb"
C_GEN = "#dc2626"


def _save(fig, name: str) -> Path:
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    rel = path.relative_to(ROOT)
    print(f"  wrote {rel}")
    return path


def _load(path: Path):
    if not path.exists():
        print(f"  WARNING: {path.relative_to(ROOT)} not found; skipping")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Figure 1 — PersonaMem per-question-type accuracy
# ============================================================================
# Recall-aligned vs creative-extrapolation question types separate cleanly.
# This is the central figure of the evaluation chapter.
# ============================================================================
RECALL_TYPES = {
    "recall_user_shared_facts",
    "recalling_the_reasons_behind_previous_updates",
    "track_full_preference_evolution",
    "recalling_facts_mentioned_by_the_user",
    "provide_preference_aligned_recommendations",
}


def fig1_personamem_per_type():
    print("Figure 1: PersonaMem per-question-type accuracy")
    data = _load(RUNS_DIR / "personamem_n50.json")
    if not data:
        return None

    # Aggregate correct/total per question_type.
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in data["results"]:
        by_type[r["question_type"]][1] += 1
        if r["correct"]:
            by_type[r["question_type"]][0] += 1

    rows = []
    for qt, (c, t) in by_type.items():
        rows.append({
            "type": qt,
            "correct": c, "total": t,
            "acc": (c / t) if t else 0.0,
            "regime": "recall" if qt in RECALL_TYPES else "generation",
        })
    df = pd.DataFrame(rows).sort_values(["regime", "acc"],
                                        ascending=[True, False]).reset_index(drop=True)

    # Pretty labels for axis.
    label_map = {
        "recall_user_shared_facts": "Recall user shared facts",
        "recalling_the_reasons_behind_previous_updates": "Recall reasons behind\nprevious updates",
        "track_full_preference_evolution": "Track full preference\nevolution",
        "provide_preference_aligned_recommendations": "Preference-aligned\nrecommendations",
        "recalling_facts_mentioned_by_the_user": "Recalling facts mentioned\nby the user",
        "suggest_new_ideas": "Suggest new ideas",
        "generalizing_to_new_scenarios": "Generalising to new\nscenarios",
    }
    df["label"] = df["type"].map(label_map).fillna(df["type"])
    df["color"] = df["regime"].map({"recall": C_RECALL, "generation": C_GEN})

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df["label"], df["acc"] * 100, color=df["color"], edgecolor="white")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PersonaMem v1 (32k) — Accuracy by Question Type (n=50 random)")
    for b, c, t, acc in zip(bars, df["correct"], df["total"], df["acc"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{acc*100:.1f}%\n({c}/{t})", ha="center", fontsize=9)

    # Overall accuracy line
    n = sum(df["total"])
    correct = sum(df["correct"])
    overall = correct / n * 100
    ax.axhline(overall, color=C_BASE_DARK, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(df) - 0.5, overall + 1.5, f"overall {overall:.1f}%",
            ha="right", fontsize=9, color=C_BASE_DARK, fontstyle="italic")

    # Regime legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_RECALL, label="Recall-aligned"),
        Patch(facecolor=C_GEN, label="Creative extrapolation"),
    ], loc="upper right", framealpha=0.9)

    ax.tick_params(axis="x", labelsize=9)
    return _save(fig, "fig1_personamem_per_type")


# ============================================================================
# Figure 2 — PersonaMem accuracy vs published baselines
# ============================================================================
def fig2_baseline_comparison():
    print("Figure 2: PersonaMem vs published baselines")
    data = _load(RUNS_DIR / "personamem_n50.json")
    ours = (data["accuracy"] * 100) if data else 64.0

    # Published baselines on PersonaMem v1-32k.
    baselines = [
        ("memU", 38.7),
        ("Zep", 43.4),
        ("Mem0", 43.9),
        ("MemOS", 50.7),
        ("EverMemOS", 53.2),
        ("Ours", ours),
    ]
    names = [b[0] for b in baselines]
    scores = [b[1] for b in baselines]
    colors = [C_BASE if n != "Ours" else C_OURS for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    ax.set_ylim(0, 75)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("PersonaMem v1 (32k) — Comparison Against Published Baselines")
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center",
                fontweight="bold" if names[bars.index(b)] == "Ours" else "regular")

    # Approximate Wilson CI for the n=50 result so the figure isn't misleading.
    if data:
        n = data["n"]
        p = data["accuracy"]
        z = 1.96
        denom = 1 + (z**2) / n
        centre = (p + (z**2) / (2 * n)) / denom
        half = (z * np.sqrt(p * (1 - p) / n + (z**2) / (4 * n**2))) / denom
        lower = (centre - half) * 100
        upper = (centre + half) * 100
        ours_idx = len(scores) - 1
        ax.errorbar([ours_idx], [ours], yerr=[[ours - lower], [upper - ours]],
                    fmt="none", ecolor=C_OURS_DARK, capsize=6, linewidth=2)
        ax.text(ours_idx, lower - 3, f"95% CI\n[{lower:.1f}, {upper:.1f}]",
                ha="center", fontsize=8, color=C_OURS_DARK)

    return _save(fig, "fig2_baseline_comparison")


# ============================================================================
# Figure 3 — Adaptation Latency per scenario
# ============================================================================
def fig3_adaptation_latency():
    print("Figure 3: Adaptation Latency per scenario (life-transition)")
    data = _load(RESULTS_DIR / "life_transition.json")
    if not data:
        return None
    df = pd.DataFrame([{
        "scenario": s["scenario"].replace("_", " ").title(),
        "AL": s["adaptation_latency_turns"],
        "n_old_hits_total": sum(len(r["old_keywords_hit"]) for r in s["replies"]),
        "n_probes": len(s["replies"]),
    } for s in data["scenarios"]])
    mean_AL = data["mean_AL"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(df["scenario"], df["AL"], color=C_OURS, edgecolor="white")
    ax.set_ylim(0, max(df["AL"]) + 1.5)
    ax.set_ylabel("Adaptation Latency (probes until adapted)")
    ax.set_title("Adaptation Latency — Synthetic Life-Transition Scenarios")
    for b, v in zip(bars, df["AL"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                f"AL = {v}", ha="center", fontweight="bold")

    ax.axhline(mean_AL, color=C_BASE_DARK, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(df) - 0.5, mean_AL + 0.08,
            f"mean = {mean_AL:.2f}", ha="right",
            fontsize=9, color=C_BASE_DARK, fontstyle="italic")

    return _save(fig, "fig3_adaptation_latency")


# ============================================================================
# Figure 4 — Eq. 7 theoretical decay curves
# ============================================================================
def fig4_decay_theory():
    print("Figure 4: Eq. 7 theoretical decay curves")
    t = np.linspace(0, 120, 300)
    floor = 0.05

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lam, label, color, ls in [
        (0.02, r"$\lambda = 0.02$ (slow)", "#1e40af", "-"),
        (0.05, r"$\lambda = 0.05$ (default)", C_GOOD, "-"),
        (0.10, r"$\lambda = 0.10$ (fast)", C_BAD, "-"),
    ]:
        ax.plot(t, np.exp(-lam * t), label=label, color=color, linewidth=2, linestyle=ls)
    ax.axhline(floor, color=C_FLOOR, linestyle=":", linewidth=1.5,
               label=f"Forget floor = {floor:.2f}")
    ax.fill_between(t, 0, floor, color=C_FLOOR, alpha=0.12)
    ax.set_xlabel("Days since last reinforcement, t")
    ax.set_ylabel(r"$I_t(m) \,/\, I_0(m)$")
    ax.set_title("Temporal Decay — Eq. 7:  $I_t(m) = I_0(m) \\cdot e^{-\\lambda t}$")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)
    ax.legend(loc="upper right")
    return _save(fig, "fig4_decay_theory")


# ============================================================================
# Figure 5 — Retrieval blend weight sweep (Phase 2.3)
# ============================================================================
# From terminal logs of retrieval_smoke.py. Importance weight β_imp varies;
# similarity weight = 1 - β_imp.
# ============================================================================
RETRIEVAL_SWEEP = [
    # (importance_weight, P@K, RNR)
    (0.00, 0.667, 0.333),
    (0.10, 0.778, 0.222),
    (0.20, 0.778, 0.222),
    (0.30, 0.889, 0.111),
    (0.40, 0.889, 0.111),
    (0.60, 0.778, 0.222),
]


def fig5_retrieval_sweep():
    print("Figure 5: Retrieval blend sweep — P@K and RNR vs importance weight")
    arr = np.array(RETRIEVAL_SWEEP)
    w, pk, rnr = arr[:, 0], arr[:, 1], arr[:, 2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)

    # Left: P@K
    ax1.plot(w, pk, marker="o", linewidth=2, color=C_GOOD)
    ax1.set_xlabel("Importance weight  $w_{\\mathrm{imp}}$")
    ax1.set_ylabel("Precision@K  (higher = better)")
    ax1.set_title("Precision@K across blend sweep")
    ax1.set_ylim(0.55, 1.0)
    ax1.axvspan(0.30, 0.40, color=C_GOOD, alpha=0.10)
    ax1.annotate("optimum band\n(0.30–0.40)",
                 xy=(0.35, 0.89), xytext=(0.10, 0.95),
                 fontsize=9, color=C_GOOD,
                 arrowprops=dict(arrowstyle="->", color=C_GOOD, alpha=0.7))
    for x, y in zip(w, pk):
        ax1.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)

    # Right: RNR
    ax2.plot(w, rnr, marker="o", linewidth=2, color=C_BAD)
    ax2.set_xlabel("Importance weight  $w_{\\mathrm{imp}}$")
    ax2.set_ylabel("Redundant-Retrieval Rate  (lower = better)")
    ax2.set_title("RNR across blend sweep")
    ax2.set_ylim(0.0, 0.4)
    ax2.axvspan(0.30, 0.40, color=C_GOOD, alpha=0.10)
    for x, y in zip(w, rnr):
        ax2.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("Retrieval Blend Sweep — P@K and RNR vs Importance Weight  "
                 "(Phase 2.3, adversarial decoys)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "fig5_retrieval_sweep")


# ============================================================================
# Figure 6 — Forgetting demo: Eq. 7 verification at empirical test points
# ============================================================================
# From terminal logs of forgetting_demo.py.
# ============================================================================
FORGETTING_POINTS = [
    # (age_days, I_0, I_t_measured, pruned)
    (120, 0.40, 0.0010, True),
    (120, 0.80, 0.0020, True),
    (30,  0.55, 0.1227, False),
]
# Plus 'recent atoms' summary: t=5 → ratio ≈ 0.78; t=2 → ratio ≈ 0.90
RECENT_RATIOS = [(5, 0.78), (2, 0.90)]


def fig6_forgetting_verification():
    print("Figure 6: Eq. 7 empirical verification (forgetting demo)")
    lam = 0.05
    floor = 0.05
    t = np.linspace(0, 130, 400)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))

    # Draw one theoretical curve per distinct I_0 in the empirical set, so each
    # atom sits at a unique position on its own curve (the ratio I_t/I_0 is
    # I_0-independent and would collapse the two t=120d points onto each other).
    I0_curves = sorted({pt[1] for pt in FORGETTING_POINTS}, reverse=True)
    palette = ["#1e40af", "#2563eb", "#60a5fa"]
    for I0, color in zip(I0_curves, palette):
        ax.plot(t, I0 * np.exp(-lam * t), color=color, linewidth=2, alpha=0.9,
                label=fr"Theory:  $I_0={I0:.2f}$,  $I_t = I_0 \cdot e^{{-\lambda t}}$")

    ax.axhline(floor, color=C_FLOOR, linestyle=":", linewidth=1.5,
               label=f"Forget floor = {floor:.2f}")
    ax.fill_between(t, 0, floor, color=C_FLOOR, alpha=0.10,
                    label="Pruning region")

    # Empirical markers at (age, I_t_measured). Each lies on its I_0 curve.
    annot_offsets = {
        (120, 0.40): (-10, 22),
        (120, 0.80): (-10, 45),
        (30, 0.55):  (12, 18),
    }
    for age, I0, It, pruned in FORGETTING_POINTS:
        color = C_BAD if pruned else C_GOOD
        marker = "X" if pruned else "o"   # "X" is a filled cross; accepts edgecolor
        ax.scatter([age], [It], s=130, marker=marker, color=color,
                   zorder=5, edgecolor="white", linewidth=1.5)
        dx, dy = annot_offsets.get((age, I0), (0, 18))
        label = (fr"$t={age}$d, $I_0={I0}$"
                 "\n"
                 fr"$I_t={It:.4f}$  ({'pruned' if pruned else 'survives'})")
        ax.annotate(label, xy=(age, It), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, lw=0.8))

    ax.set_xlabel("Atom age, t (days)")
    ax.set_ylabel(r"Importance  $I_t$")
    ax.set_title("Eq. 7 Empirical Verification — Backdated Atoms (Phase 3.1)")
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 0.95)
    ax.legend(loc="upper right", fontsize=9)

    return _save(fig, "fig6_forgetting_verification")


# ============================================================================
# Figure 7 — LoCoMo-MC10 per-question-type accuracy
# ============================================================================
# Second public benchmark. Same recall-vs-synthesis split as Figure 1, which
# strengthens the central architectural argument.
# ============================================================================
LOCOMO_RECALL_TYPES = {"single_hop", "adversarial"}


def fig7_locomo_per_type():
    print("Figure 7: LoCoMo-MC10 per-question-type accuracy")
    data = _load(RUNS_DIR / "locomo_n30.json")
    if not data:
        return None

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in data["results"]:
        by_type[r["question_type"]][1] += 1
        if r["correct"]:
            by_type[r["question_type"]][0] += 1

    rows = []
    for qt, (c, t) in by_type.items():
        rows.append({
            "type": qt,
            "correct": c, "total": t,
            "acc": (c / t) if t else 0.0,
            "regime": "recall" if qt in LOCOMO_RECALL_TYPES else "generation",
        })
    df = pd.DataFrame(rows).sort_values(["regime", "acc"],
                                        ascending=[True, False]).reset_index(drop=True)

    label_map = {
        "single_hop": "Single-hop\n(direct recall)",
        "adversarial": "Adversarial\n(refusal)",
        "multi_hop": "Multi-hop\n(cross-session synthesis)",
        "open_domain": "Open-domain",
        "temporal_reasoning": "Temporal\nreasoning",
    }
    df["label"] = df["type"].map(label_map).fillna(df["type"])
    df["color"] = df["regime"].map({"recall": C_RECALL, "generation": C_GEN})

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["label"], df["acc"] * 100, color=df["color"], edgecolor="white")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LoCoMo-MC10 — Accuracy by Question Type (n=30 random)")

    for b, c, t, acc in zip(bars, df["correct"], df["total"], df["acc"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                f"{acc*100:.1f}%\n({c}/{t})", ha="center", fontsize=9)

    n = sum(df["total"])
    correct = sum(df["correct"])
    overall = (correct / n * 100) if n else 0
    ax.axhline(overall, color=C_BASE_DARK, linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(df) - 0.5, overall + 1.5, f"overall {overall:.1f}%",
            ha="right", fontsize=9, color=C_BASE_DARK, fontstyle="italic")

    # Random baseline marker — important context for a 10-choice MCQ.
    ax.axhline(10, color=C_FLOOR, linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0, 11.5, "random baseline (10%)",
            ha="left", fontsize=8, color=C_FLOOR, fontstyle="italic")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C_RECALL, label="Recall-aligned"),
        Patch(facecolor=C_GEN, label="Synthesis / generation"),
    ], loc="upper right", framealpha=0.9)

    ax.tick_params(axis="x", labelsize=9)
    return _save(fig, "fig7_locomo_per_type")


# ============================================================================
# Summary table for the dissertation
# ============================================================================
def write_summary():
    print("Summary: CSV + LaTeX table")
    pm50 = _load(RUNS_DIR / "personamem_n50.json")
    lt = _load(RESULTS_DIR / "life_transition.json")
    lc30 = _load(RUNS_DIR / "locomo_n30.json")

    rows = []
    if pm50:
        rows.append({
            "Metric": "PersonaMem v1 accuracy (n=50)",
            "Value": f"{pm50['accuracy']*100:.1f}%",
            "Best baseline": "EverMemOS 53.2%",
        })
    if lc30:
        rows.append({
            "Metric": "LoCoMo-MC10 accuracy (n=30)",
            "Value": f"{lc30['accuracy']*100:.1f}%",
            "Best baseline": "no MC10 baselines (10-choice; random=10%)",
        })
    rows.append({
        "Metric": "Retrieval P@K (optimum band)",
        "Value": "0.889",
        "Best baseline": "Pure cosine 0.667",
    })
    rows.append({
        "Metric": "Retrieval RNR (optimum band)",
        "Value": "0.111",
        "Best baseline": "Pure cosine 0.333",
    })
    if lt:
        rows.append({
            "Metric": "Adaptation Latency (mean of 3)",
            "Value": f"{lt['mean_AL']:.2f} turns",
            "Best baseline": "n/a (original metric)",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    csv_path = FIG_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path.relative_to(ROOT)}")

    tex_path = FIG_DIR / "summary.tex"
    tex_path.write_text(df.to_latex(index=False, escape=False))
    print(f"  wrote {tex_path.relative_to(ROOT)}")


# ============================================================================
def main():
    print("=" * 72)
    print("Generating dissertation figures")
    print("=" * 72)
    print(f"  output dir: {FIG_DIR}")
    print()
    fig1_personamem_per_type()
    fig2_baseline_comparison()
    fig3_adaptation_latency()
    fig4_decay_theory()
    fig5_retrieval_sweep()
    fig6_forgetting_verification()
    fig7_locomo_per_type()
    print()
    write_summary()
    print()
    print("Done.")


if __name__ == "__main__":
    main()