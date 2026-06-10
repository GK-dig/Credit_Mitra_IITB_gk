"""
Fig 2B — Error Breakdown per Model
Shows exact match, partial match, and complete failure rates.

Run in Codespace:
    python fig2b_error_breakdown.py --project_root .

Reads:
    <root>/outputs_8/outputs/eval/predictions.jsonl   (LoRA)
    <root>/outputs_1/eval-dp/predictions.jsonl        (DP-LoRA)
    <root>/outputs_1/plots/benchmark_summary.json     (Base Qwen + baselines)

Saves:
    <root>/outputs_1/plots/figures/fig2b_error_breakdown.pdf
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    return p.parse_args()

def load_preds(path):
    char_sims, ems = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            char_sims.append(float(obj["char_similarity"]))
            ems.append(int(obj["exact_match"]))
    return np.array(char_sims), np.array(ems)

def compute_breakdown(sims, ems, n):
    """
    Three categories:
      Exact Match   : exact_match == 1                  (sim = 1.0)
      Partial Match : exact_match == 0 AND sim >= 0.5   (got close, wrong)
      Complete Fail : exact_match == 0 AND sim <  0.5   (far off)
    """
    exact   = ems.sum()
    partial = ((ems == 0) & (sims >= 0.5)).sum()
    fail    = ((ems == 0) & (sims <  0.5)).sum()
    return (exact / n * 100,
            partial / n * 100,
            fail    / n * 100)

def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_8/outputs/plots/figures"
    out.mkdir(parents=True, exist_ok=True)

    # Load per-sample predictions for LoRA and DP-LoRA
    lora_sims, lora_ems = load_preds(root / "outputs_8/outputs/eval/predictions.jsonl")
    dp_sims,   dp_ems   = load_preds(root / "outputs_8/outputs/eval-dp/predictions.jsonl")
    n = len(lora_ems)

    # Load benchmark summary for Base Qwen and baselines
    with open(root / "outputs_8/outputs/plots/benchmark_summary.json") as f:
        bench = json.load(f)

    # For models without per-sample files, approximate from EM and F1:
    # partial ≈ (F1/100 - EM/100) * n  (got some tokens right but not exact)
    # fail    ≈ rest
    def bench_breakdown(name):
        em = bench[name]["exact_match"] / 100
        f1 = bench[name]["f1"] / 100
        exact   = em * 100
        partial = max(0, (f1 - em)) * 100
        fail    = 100 - exact - partial
        return exact, partial, fail

    # Build data for all models
    lora_exact, lora_partial, lora_fail = compute_breakdown(lora_sims, lora_ems, n)
    dp_exact,   dp_partial,   dp_fail   = compute_breakdown(dp_sims,   dp_ems,   n)

    models = [
        "Base Qwen\n2.5-1.5B",
        "Llama-3.2-1B\n(Meta)",
        "Gemma-2-2B\n(Google)",
        "QLoRA\n(non-private)",
        "DP-QLoRA\n(ε≈0.994)",
    ]

    exact_vals   = [
        bench["Base Qwen 2.5-1.5B"]["exact_match"],
        bench["Llama-3.2-1B (Meta)"]["exact_match"],
        bench["Gemma-2-2B (Google)"]["exact_match"],
        lora_exact,
        dp_exact,
    ]
    partial_vals = [
        *[max(0, bench[k]["f1"] - bench[k]["exact_match"])
          for k in ["Base Qwen 2.5-1.5B",
                    "Llama-3.2-1B (Meta)",
                    "Gemma-2-2B (Google)"]],
        lora_partial,
        dp_partial,
    ]
    fail_vals = [100 - e - p for e, p in zip(exact_vals, partial_vals)]

    # ── Plot ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F9F9F9",
    })

    x     = np.arange(len(models))
    width = 0.22
    fig, ax = plt.subplots(figsize=(13, 6))

    bars_exact   = ax.bar(x - width, exact_vals,   width,
                          label="Exact Match",
                          color="#82B366", edgecolor="white", lw=0.8, zorder=3)
    bars_partial = ax.bar(x,          partial_vals, width,
                          label="Partial Match  (sim ≥ 0.5, EM = 0)",
                          color="#F0A500", edgecolor="white", lw=0.8, zorder=3)
    bars_fail    = ax.bar(x + width,  fail_vals,    width,
                          label="Complete Failure  (sim < 0.5)",
                          color="#AE4132", edgecolor="white", lw=0.8, zorder=3)

    # Value labels on bars
    for bars in [bars_exact, bars_partial, bars_fail]:
        for bar in bars:
            h = bar.get_height()
            if h > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.5, f"{h:.1f}%",
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333")

    # Divider between baselines and your models
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, lw=1.2)
    ax.text(1.0,  102, "Baselines",  fontsize=9, color="gray", ha="center")
    ax.text(3.5,  102, "Your Models", fontsize=9, color="gray", ha="center")

    # Highlight DP cost with a bracket annotation
    ax.annotate("",
                xy=(3 - width, dp_exact + 1),
                xytext=(4 - width, lora_exact + 1),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.2))
    ax.text(3.5, max(dp_exact, lora_exact) + 4,
            f"DP cost\n−{lora_exact - dp_exact:.1f}% EM",
            ha="center", fontsize=8.5, color="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Percentage of Test Samples (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title(
        "Prediction Error Breakdown — All Models\n"
        "Exact Match vs Partial Match vs Complete Failure  (n=486)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    path = out / "fig2b_error_breakdown.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()