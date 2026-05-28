"""
Payee Extraction Benchmarking Pipeline
- Base Qwen/Qwen2.5-1.5B-Instruct : pulled from HuggingFace Hub, inference run fresh
- Fine-tuned LoRA (payee-lora/checkpoint-414) : predictions reused from eval/predictions.jsonl
- DP Fine-tuned LoRA (payee-lora-dp)          : predictions reused from eval-dp/predictions.jsonl

Usage:
    python benchmark_payee.py \
        --test_file   ./test.jsonl \
        --eval_dir    ./outputs/eval \
        --eval_dp_dir ./outputs/eval-dp \
        --output_dir  ./outputs/plots

Optional:
    --base_model  Qwen/Qwen2.5-1.5B-Instruct   (default)
    --batch_size  8
    --max_new_tokens 30
    --limit 100    (quick smoke-test)
"""

import argparse, json, time, os, re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────────────────
# 0. ARGS
# ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file",    required=True,
                   help="Path to test.jsonl  (fields: prompt, response)")
    p.add_argument("--eval_dir",     required=True,
                   help="outputs/eval/  — must contain predictions.jsonl")
    p.add_argument("--eval_dp_dir",  required=True,
                   help="outputs/eval-dp/ — must contain predictions.jsonl")
    p.add_argument("--output_dir",   default="./outputs/plots",
                   help="Where to write graphs + summary")
    p.add_argument("--base_model",   default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HuggingFace Hub ID for base model")
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--limit",          type=int, default=None,
                   help="Cap test samples for a quick run")
    return p.parse_args()


# ──────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────

def load_test_data(filepath: str, limit=None) -> List[Dict]:
    """Load JSONL test set. Each line: {prompt, response}."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append({
                "prompt":       obj["prompt"],
                "ground_truth": obj["response"].strip()
            })
    if limit:
        samples = samples[:limit]
    print(f"  Loaded {len(samples)} test samples from {filepath}")
    return samples


def load_existing_predictions(pred_dir: str, n_expected: int) -> List[str]:
    """
    Load predictions.jsonl from eval/ or eval-dp/.
    Supports keys: prediction | response | output
    Truncates / warns if length mismatches.
    """
    pred_file = os.path.join(pred_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(
            f"predictions.jsonl not found in {pred_dir}\n"
            f"Expected path: {pred_file}"
        )
    preds = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("prediction",
                   obj.get("response",
                   obj.get("output", ""))).strip()
            preds.append(pred)

    if len(preds) != n_expected:
        print(f"  WARNING: {pred_dir} has {len(preds)} predictions "
              f"but test set has {n_expected}. Using first {min(len(preds), n_expected)}.")
        preds = preds[:n_expected]

    print(f"  Loaded {len(preds)} cached predictions from {pred_dir}")
    return preds


# ──────────────────────────────────────────────────────────
# 2. BASE MODEL INFERENCE (Hub → no local path needed)
# ──────────────────────────────────────────────────────────

def load_base_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Pulling base model '{model_id}' onto {device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    return tokenizer, model, device


def clean_output(raw: str) -> str:
    """Keep only the first line and strip trailing junk tokens."""
    raw = raw.strip().split("\n")[0].strip()
    raw = re.sub(r"[<\|].*", "", raw).strip()
    return raw


def run_base_inference(
    model_id: str,
    samples: List[Dict],
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[List[str], float]:
    print(f"\n{'='*55}\n  Base model inference: {model_id}\n{'='*55}")
    tokenizer, model, device = load_base_model(model_id)
    prompts = [s["prompt"] for s in samples]
    predictions, total_time = [], 0.0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        total_time += time.perf_counter() - t0

        # Decode only newly generated tokens
        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend([clean_output(d) for d in decoded])

        done = min(i + batch_size, len(prompts))
        if (i // batch_size + 1) % 10 == 0:
            print(f"    [{done}/{len(prompts)}] done …")

    avg_lat = (total_time / len(samples)) * 1000
    print(f"  Done. Avg latency: {avg_lat:.1f} ms/sample")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, avg_lat


# ──────────────────────────────────────────────────────────
# 3. METRICS
# ──────────────────────────────────────────────────────────

MODEL_NAMES = [
    "Base Qwen 2.5-1.5B",
    "Fine-tuned (LoRA)",
    "DP Fine-tuned (LoRA)",
]

COLORS = {
    "Base Qwen 2.5-1.5B":   "#6C8EBF",
    "Fine-tuned (LoRA)":    "#82B366",
    "DP Fine-tuned (LoRA)": "#D79B00",
}


def normalize(t: str) -> str:
    return t.lower().strip()


def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    pt = normalize(pred).split()
    gt = normalize(gold).split()
    if not pt and not gt: return 1.0, 1.0, 1.0
    if not pt or  not gt: return 0.0, 0.0, 0.0
    ps = {t: pt.count(t) for t in pt}
    gs = {t: gt.count(t) for t in gt}
    common = sum(min(ps.get(t, 0), gs.get(t, 0)) for t in gs)
    prec = common / len(pt)
    rec  = common / len(gt)
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    em, precs, recs, f1s = 0, [], [], []
    for pred, gold in zip(predictions, ground_truths):
        if normalize(pred) == normalize(gold):
            em += 1
        p, r, f = token_f1(pred, gold)
        precs.append(p); recs.append(r); f1s.append(f)
    n = len(predictions)
    return {
        "exact_match": round(em / n * 100, 2),
        "precision":   round(np.mean(precs) * 100, 2),
        "recall":      round(np.mean(recs)  * 100, 2),
        "f1":          round(np.mean(f1s)   * 100, 2),
        "n_samples":   n,
    }


# ──────────────────────────────────────────────────────────
# 4. GRAPHS  (4 plots)
# ──────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "figure.facecolor":   "white",
        "axes.facecolor":     "#F8F8F8",
    })


# Graph 1 — grouped bar: all 4 metrics × 3 models
def plot_accuracy(all_metrics: Dict, out: str):
    _style()
    metric_keys   = ["exact_match", "precision", "recall", "f1"]
    metric_labels = ["Exact Match", "Precision", "Recall", "F1"]
    x      = np.arange(len(metric_labels))
    width  = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, offset in zip(MODEL_NAMES, offsets):
        vals = [all_metrics[name][k] for k in metric_keys]
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=COLORS[name], edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.7,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Payee Extraction — Model Accuracy Comparison",
                 fontsize=14, fontweight="bold", pad=16)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    path = os.path.join(out, "graph1_accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Graph 2 — horizontal bar: latency (base model only, cached have no timing)
def plot_latency(latencies: Dict, out: str):
    _style()
    names  = [n for n in MODEL_NAMES if latencies[n] > 0]
    vals   = [latencies[n] for n in names]
    colors = [COLORS[n]    for n in names]

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 1.4)))
    bars = ax.barh(names, vals, color=colors, edgecolor="white",
                   linewidth=0.8, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} ms", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Avg Latency per Sample (ms)", fontsize=11)
    ax.set_title("Inference Latency — Base Model",
                 fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    fig.text(0.5, -0.04,
             "Note: latency shown for base model only "
             "(fine-tuned results loaded from saved predictions)",
             ha="center", fontsize=8, color="gray", style="italic")
    fig.tight_layout()
    path = os.path.join(out, "graph2_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Graph 3 — radar chart: all 4 metrics × 3 models
def plot_radar(all_metrics: Dict, out: str):
    _style()
    categories = ["Exact Match", "Precision", "Recall", "F1"]
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name in MODEL_NAMES:
        m      = all_metrics[name]
        values = [m["exact_match"], m["precision"], m["recall"], m["f1"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=name, color=COLORS[name])
        ax.fill(angles, values, alpha=0.08, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.set_title("Model Performance Radar\nBase vs Fine-tuned vs DP Fine-tuned",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    path = os.path.join(out, "graph3_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Graph 4 — clean single-metric bar: Exact Match headline figure
def plot_exact_match(all_metrics: Dict, out: str):
    _style()
    vals   = [all_metrics[n]["exact_match"] for n in MODEL_NAMES]
    colors = [COLORS[n] for n in MODEL_NAMES]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(MODEL_NAMES, vals, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.5, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Exact Match Accuracy — All Models",
                 fontsize=13, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticklabels(MODEL_NAMES, fontsize=10)
    fig.tight_layout()
    path = os.path.join(out, "graph4_exact_match.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# 5. SAVE + PRINT
# ──────────────────────────────────────────────────────────

def print_table(all_metrics: Dict, latencies: Dict):
    print("\n" + "=" * 75)
    print(f"{'Model':<26} {'EM%':>8} {'Prec%':>8} {'Rec%':>8} "
          f"{'F1%':>8} {'Latency':>12}")
    print("=" * 75)
    for name in MODEL_NAMES:
        m   = all_metrics[name]
        lat = f"{latencies[name]:.1f} ms" if latencies[name] > 0 else "N/A (cached)"
        print(f"{name:<26} {m['exact_match']:>7.1f}% {m['precision']:>7.1f}% "
              f"{m['recall']:>7.1f}% {m['f1']:>7.1f}% {lat:>12}")
    print("=" * 75)


def save_summary(all_metrics: Dict, latencies: Dict, out: str):
    summary = {
        n: {**all_metrics[n], "avg_latency_ms": latencies[n]}
        for n in MODEL_NAMES
    }
    path = os.path.join(out, "benchmark_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


def save_csv(all_preds: Dict, samples: List[Dict], out: str):
    path = os.path.join(out, "per_sample_predictions.csv")
    with open(path, "w", encoding="utf-8") as f:
        cols = (["sample_id", "ground_truth"]
                + [f"pred_{n.lower().replace(' ','_').replace('(','').replace(')','')}"
                   for n in MODEL_NAMES]
                + [f"exact_{n.lower().replace(' ','_').replace('(','').replace(')','')}"
                   for n in MODEL_NAMES])
        f.write(",".join(cols) + "\n")
        for i, sample in enumerate(samples):
            gt     = sample["ground_truth"]
            preds  = [all_preds[n][i] for n in MODEL_NAMES]
            exacts = ["1" if normalize(p) == normalize(gt) else "0" for p in preds]
            f.write(f"{i},{gt}," + ",".join(preds) + "," + ",".join(exacts) + "\n")
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n── Loading test data ──")
    samples       = load_test_data(args.test_file, limit=args.limit)
    ground_truths = [s["ground_truth"] for s in samples]
    n             = len(samples)

    all_predictions: Dict[str, List[str]] = {}
    all_metrics:     Dict[str, Dict]      = {}
    latencies:       Dict[str, float]     = {}

    # ── 1. Base model: run inference fresh ────────────────
    print("\n── Base Model (HuggingFace Hub) ──")
    base_preds, base_lat = run_base_inference(
        args.base_model, samples, args.max_new_tokens, args.batch_size
    )
    all_predictions["Base Qwen 2.5-1.5B"] = base_preds
    all_metrics["Base Qwen 2.5-1.5B"]     = compute_metrics(base_preds, ground_truths)
    latencies["Base Qwen 2.5-1.5B"]       = base_lat

    # ── 2. Fine-tuned LoRA: reuse eval/predictions.jsonl ──
    print("\n── Fine-tuned LoRA (cached from eval/) ──")
    ft_preds = load_existing_predictions(args.eval_dir, n)
    all_predictions["Fine-tuned (LoRA)"] = ft_preds
    all_metrics["Fine-tuned (LoRA)"]     = compute_metrics(ft_preds, ground_truths)
    latencies["Fine-tuned (LoRA)"]       = 0.0

    # ── 3. DP Fine-tuned LoRA: reuse eval-dp/predictions.jsonl
    print("\n── DP Fine-tuned LoRA (cached from eval-dp/) ──")
    dp_preds = load_existing_predictions(args.eval_dp_dir, n)
    all_predictions["DP Fine-tuned (LoRA)"] = dp_preds
    all_metrics["DP Fine-tuned (LoRA)"]     = compute_metrics(dp_preds, ground_truths)
    latencies["DP Fine-tuned (LoRA)"]       = 0.0

    # ── Results table ──────────────────────────────────────
    print_table(all_metrics, latencies)

    # ── Graphs ─────────────────────────────────────────────
    print("\n── Generating graphs ──")
    plot_accuracy(all_metrics,    args.output_dir)
    plot_latency(latencies,       args.output_dir)
    plot_radar(all_metrics,       args.output_dir)
    plot_exact_match(all_metrics, args.output_dir)

    # ── Save files ─────────────────────────────────────────
    print("\n── Saving output files ──")
    save_summary(all_metrics, latencies, args.output_dir)
    save_csv(all_predictions, samples,   args.output_dir)

    print(f"\n✓  All done. Outputs written to: {args.output_dir}/")
    print("   graph1_accuracy_comparison.png  — grouped bar (EM / Prec / Rec / F1)")
    print("   graph2_latency.png              — latency bar (base model)")
    print("   graph3_radar.png                — radar chart (all 3 models)")
    print("   graph4_exact_match.png          — headline exact-match bar")
    print("   benchmark_summary.json          — all metrics as JSON")
    print("   per_sample_predictions.csv      — per-sample breakdown")


if __name__ == "__main__":
    main()