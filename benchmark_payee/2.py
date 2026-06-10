
"""
Training Data Extraction Test
==============================
Tests whether models memorize and can reproduce actual training narrations
verbatim when given a partial prefix.

This is a LEGITIMATE privacy test because:
- We use REAL training samples (not fake canaries)
- We give the model the first half of a narration
- We check if it completes the second half exactly as in training
- DP should make verbatim completion harder

Comparison:
- Non-private LoRA: trained on these exact samples → should complete well
- DP-QLoRA: DP noise should degrade verbatim memorization
- Val samples: neither model saw these → baseline for "no memorization"

Run from project root:
    python training_data_extraction.py --project_root .

Saves to: outputs_1/plots/figures/extraction/
"""

import argparse
import json
import random
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from difflib import SequenceMatcher

BASE_MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 40


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--n_samples",    type=int, default=100,
                   help="Samples per set — train and val (default 100)")
    p.add_argument("--prefix_frac",  type=float, default=0.5,
                   help="Fraction of narration to use as prefix (default 0.5)")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path, n: int, seed: int) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    random.Random(seed).shuffle(records)
    return records[:n]


def extract_narration(prompt: str) -> str:
    """Pull out just the narration text from the prompt."""
    match = re.search(
        r"Transaction narration:\n(.+?)\n\nPayee:", prompt, re.DOTALL
    )
    return match.group(1).strip() if match else ""


def make_prefix_prompt(narration: str, prefix_frac: float) -> tuple:
    """
    Split narration at prefix_frac and build a completion prompt.
    Returns (prefix_prompt, expected_suffix).
    """
    cutoff  = max(5, int(len(narration) * prefix_frac))
    prefix  = narration[:cutoff]
    suffix  = narration[cutoff:]

    prompt = (
        "You are an information extraction model. Extract only the payee name "
        "from the transaction narration. Return only the payee text, with no extra words."
        f"\n\nTransaction narration:\n{prefix}"
    )
    return prompt, prefix, suffix


# ── Similarity ────────────────────────────────────────────────────────────────
def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def verbatim_match(pred: str, suffix: str) -> bool:
    """True if prediction contains the exact suffix (case-insensitive)."""
    return suffix.lower().strip() in pred.lower().strip()

def starts_with_match(pred: str, suffix: str, n_chars: int = 10) -> bool:
    """True if first n chars of suffix appear at start of prediction."""
    return pred.lower().strip().startswith(suffix[:n_chars].lower().strip())


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(adapter_path: Path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def complete(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=256
    ).to(next(model.parameters()).device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Run test ──────────────────────────────────────────────────────────────────
def run_extraction_test(model, tokenizer, records: list,
                        prefix_frac: float, label: str) -> dict:
    results = []
    print(f"\n  Testing {label}...")

    for rec in tqdm(records, desc=f"  {label}", leave=False):
        narration = extract_narration(rec["prompt"])
        if not narration or len(narration) < 20:
            continue

        prompt, prefix, suffix = make_prefix_prompt(narration, prefix_frac)
        pred = complete(model, tokenizer, prompt)

        sim      = char_similarity(pred, suffix)
        verbatim = verbatim_match(pred, suffix)
        starts   = starts_with_match(pred, suffix)

        results.append({
            "id":          rec["id"],
            "narration":   narration,
            "prefix":      prefix,
            "suffix":      suffix,
            "prediction":  pred,
            "char_sim":    sim,
            "verbatim":    verbatim,
            "starts_with": starts,
        })

    n             = len(results)
    verbatim_rate = sum(r["verbatim"]    for r in results) / n
    starts_rate   = sum(r["starts_with"] for r in results) / n
    mean_sim      = np.mean([r["char_sim"] for r in results])

    print(f"\n  {label}:")
    print(f"    Verbatim completion rate : {verbatim_rate:.1%}")
    print(f"    Starts-with match rate   : {starts_rate:.1%}")
    print(f"    Mean char similarity     : {mean_sim:.4f}")
    print(f"    n = {n}")

    return {
        "label":          label,
        "n":              n,
        "verbatim_rate":  round(verbatim_rate, 4),
        "starts_rate":    round(starts_rate,   4),
        "mean_sim":       round(mean_sim,      4),
        "per_sample":     results,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(lora_train, dp_train, lora_val, dp_val, out: Path):
    """
    4-group comparison:
    - LoRA on train samples   (seen during training)
    - DP-LoRA on train samples (seen during training, DP noise applied)
    - LoRA on val samples     (never seen — baseline)
    - DP-LoRA on val samples  (never seen — baseline)

    Key question: does DP reduce completion of TRAIN samples
    while val samples remain similarly hard for both?
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    groups  = ["LoRA\n(train)", "DP-LoRA\n(train)",
               "LoRA\n(val)",   "DP-LoRA\n(val)"]
    colors  = ["#82B366", "#D79B00", "#82B366", "#D79B00"]
    alphas  = [1.0, 1.0, 0.45, 0.45]
    hatches = ["", "", "///", "///"]
    data    = [lora_train, dp_train, lora_val, dp_val]

    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ("verbatim_rate", "Verbatim Completion Rate", "Verbatim Completion\n(exact suffix reproduced)"),
        ("starts_rate",   "Starts-With Match Rate",   "Prefix Completion\n(first 10 chars match)"),
        ("mean_sim",      "Mean Char Similarity",      "Character Similarity\n(suffix vs prediction)"),
    ]):
        ax   = axes[ax_idx]
        vals = [d[metric_key] * (100 if metric_key != "mean_sim" else 1)
                for d in data]

        bars = ax.bar(groups, vals,
                      color=colors, alpha=1.0,
                      edgecolor="white", zorder=3)

        # Apply hatch and alpha manually
        for bar, h, a in zip(bars, hatches, alphas):
            bar.set_hatch(h)
            bar.set_alpha(a)

        for bar, val in zip(bars, vals):
            unit = "%" if metric_key != "mean_sim" else ""
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + (0.5 if metric_key != "mean_sim" else 0.005),
                    f"{val:.1f}{unit}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        # Annotate DP reduction on train samples
        reduction = vals[0] - vals[1]
        if reduction > 0:
            ax.annotate("",
                        xy=(1, vals[1] + (0.3 if metric_key != "mean_sim" else 0.003)),
                        xytext=(0, vals[0] + (0.3 if metric_key != "mean_sim" else 0.003)),
                        arrowprops=dict(arrowstyle="<->", color="#AE4132", lw=1.3))
            unit = "%" if metric_key != "mean_sim" else ""
            ax.text(0.5, max(vals[0], vals[1]) * 0.6,
                    f"DP reduces\nby {reduction:.1f}{unit}",
                    ha="center", fontsize=8.5,
                    fontweight="bold", color="#AE4132")

        # Divider between train and val groups
        ax.axvline(x=1.5, color="gray", linestyle="--", alpha=0.5, lw=1)
        ax.text(0.5, ax.get_ylim()[1] * 0.97,
                "Seen in training", ha="center", fontsize=7.5, color="gray")
        ax.text(2.5, ax.get_ylim()[1] * 0.97,
                "Not seen", ha="center", fontsize=7.5, color="gray")

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    fig.suptitle(
        "Training Data Extraction Test\n"
        "Does DP reduce verbatim memorization of training narrations?",
        fontsize=12, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out / "fig_extraction_test.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_extraction_test.pdf")


def plot_sim_distribution(lora_train, dp_train, out: Path):
    """
    Histogram of char similarity scores for train samples.
    Shows how DP shifts the distribution away from perfect memorization.
    """
    lora_sims = [r["char_sim"] for r in lora_train["per_sample"]]
    dp_sims   = [r["char_sim"] for r in dp_train["per_sample"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(lora_sims, bins=bins, alpha=0.65, color="#82B366",
            label=f"QLoRA non-private  (mean={np.mean(lora_sims):.3f})",
            edgecolor="white", density=True)
    ax.hist(dp_sims,   bins=bins, alpha=0.65, color="#D79B00",
            label=f"DP-QLoRA ε≈0.9938  (mean={np.mean(dp_sims):.3f})",
            edgecolor="white", density=True)

    ax.axvline(np.mean(lora_sims), color="#82B366", lw=2, linestyle="--")
    ax.axvline(np.mean(dp_sims),   color="#D79B00", lw=2, linestyle="--")

    ax.set_xlabel("Character Similarity (prediction vs actual suffix)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Similarity Distribution — Training Sample Completion\n"
        "DP shifts distribution left = less verbatim memorization",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig_extraction_sim_dist.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_extraction_sim_dist.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_1/plots/figures/extraction"
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    print(f"\nProject root : {root}")
    print(f"Output dir   : {out}")
    print(f"Samples      : {args.n_samples} train + {args.n_samples} val")
    print(f"Prefix frac  : {args.prefix_frac} (first {args.prefix_frac*100:.0f}% of narration as prompt)")

    train_records = load_jsonl(
        root / "data/train.jsonl",
        args.n_samples, args.seed)
    val_records   = load_jsonl(
        root / "data/val.jsonl",
        args.n_samples, args.seed)

    print(f"\nLoaded {len(train_records)} train, {len(val_records)} val samples")

    # ── DP model ───────────────────────────────────────────────────
    dp_adapter = root / "outputs_1/payee-lora-dp"
    print(f"\nLoading DP-QLoRA: {dp_adapter}")
    model_dp, tokenizer = load_model(dp_adapter)

    dp_train = run_extraction_test(
        model_dp, tokenizer, train_records,
        args.prefix_frac, "DP-QLoRA (train samples)")
    dp_val   = run_extraction_test(
        model_dp, tokenizer, val_records,
        args.prefix_frac, "DP-QLoRA (val samples)")

    del model_dp
    torch.cuda.empty_cache()

    # ── Non-DP LoRA ────────────────────────────────────────────────
    lora_adapter = root / "outputs_8/outputs/payee-lora"
    print(f"\nLoading non-DP LoRA: {lora_adapter}")
    model_nd, _ = load_model(lora_adapter)

    lora_train = run_extraction_test(
        model_nd, tokenizer, train_records,
        args.prefix_frac, "QLoRA non-private (train samples)")
    lora_val   = run_extraction_test(
        model_nd, tokenizer, val_records,
        args.prefix_frac, "QLoRA non-private (val samples)")

    del model_nd
    torch.cuda.empty_cache()

    # ── Save raw results ───────────────────────────────────────────
    for res, name in [(dp_train, "dp_train"), (dp_val, "dp_val"),
                      (lora_train, "lora_train"), (lora_val, "lora_val")]:
        with open(out / f"results_{name}.json", "w") as f:
            # Save without per_sample to keep file small
            summary = {k: v for k, v in res.items() if k != "per_sample"}
            json.dump(summary, f, indent=2)

    # ── Figures ────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_results(lora_train, dp_train, lora_val, dp_val, out)
    plot_sim_distribution(lora_train, dp_train, out)

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  TRAINING DATA EXTRACTION SUMMARY")
    print(f"{'='*58}")
    print(f"  {'Metric':<30} {'LoRA':>10} {'DP-LoRA':>10} {'Reduction':>10}")
    print(f"  {'-'*58}")
    for metric, label in [
        ("verbatim_rate", "Verbatim rate (train)"),
        ("starts_rate",   "Starts-with  (train)"),
        ("mean_sim",      "Mean sim     (train)"),
    ]:
        l = lora_train[metric]
        d = dp_train[metric]
        r = l - d
        print(f"  {label:<30} {l:>10.4f} {d:>10.4f} {r:>+10.4f}")
    print(f"\n  Key: if LoRA >> DP-LoRA on train but similar on val")
    print(f"       → DP successfully reduced memorization")
    print(f"{'='*58}")
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
