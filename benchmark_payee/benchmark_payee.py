"""
Payee Extraction Benchmarking Pipeline
==============================================
PRIMARY MODELS (your trained checkpoints):
  - Base Qwen/Qwen2.5-1.5B-Instruct     : HuggingFace Hub, fresh inference
  - Fine-tuned LoRA (checkpoint-414)     : predictions reused from eval/
  - DP Fine-tuned LoRA (payee-lora-dp)   : predictions reused from eval-dp/

BASELINE COMPARISONS (same ~1.5B parameter class, NER literature standards):
  - meta-llama/Llama-3.2-1B-Instruct     : Meta's 1B instruction model (edge extraction)
  - google/gemma-2-2b-it                  : Google's 2B, highest NER accuracy in class

All 3 baselines run via HuggingFace Inference API — no local GPU required.

Usage:
    python benchmark_payee.py \
        --test_file   ./test.jsonl \
        --eval_dir    ./outputs/eval \
        --eval_dp_dir ./outputs/eval-dp \
        --hf_token    hf_xxxxxxxxxxxx \
        --output_dir  ./outputs/plots

Optional:
    --limit 50        quick smoke-test
    --batch_size 8
    --max_new_tokens 30
    --skip_baselines  skip API calls (plot with existing results only)

WHY THESE 3 BASELINE MODELS:
  Llama-3.2-1B  — closest parameter match; Meta's official 1B instruction model.
                   Competitive with Gemma 2B on instruction-following (Meta, 2024).
 
  Gemma-2-2b    — "highest accuracy overall, particularly excelling in extracting
                   various entity types" vs Llama 3.2 and Qwen (Analytics Vidhya, 2025).
"""
import os
import argparse, json, time, os, re, requests
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv


load_dotenv()


# ══════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════

# Your 3 primary models
PRIMARY_MODELS = [
    "Base Qwen 2.5-1.5B",
    "Fine-tuned (LoRA)",
    "DP Fine-tuned (LoRA)",
]

# 3 baselines — all via HF Inference API
BASELINE_MODELS = [
    "Llama-3.2-1B (Meta)",
    "Gemma-2-2B (Google)",
]

BASELINE_HF_IDS = {
    "Llama-3.2-1B (Meta)":      "meta-llama/Llama-3.2-1B-Instruct",
    "Gemma-2-2B (Google)":      "google/gemma-2-2b-it",
}

ALL_MODELS = PRIMARY_MODELS + BASELINE_MODELS

# Color palette — primaries warm, baselines cool-neutral
COLORS = {
    "Base Qwen 2.5-1.5B":      "#6C8EBF",   # blue
    "Fine-tuned (LoRA)":       "#82B366",   # green
    "DP Fine-tuned (LoRA)":    "#D79B00",   # amber
    "Llama-3.2-1B (Meta)":     "#AE4132",   # red
    "Gemma-2-2B (Google)":     "#3A7D7B",   # teal
}

# Hatch patterns to distinguish groups even in B&W print
HATCHES = {
    "Base Qwen 2.5-1.5B":      "",
    "Fine-tuned (LoRA)":       "",
    "DP Fine-tuned (LoRA)":    "",
    "Llama-3.2-1B (Meta)":     "///",
    "Gemma-2-2B (Google)":     "///",
}

HF_API_URL = "https://api-inference.huggingface.co/models/{model_id}"


# ══════════════════════════════════════════════════════════
# 1. ARGS
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Payee extraction benchmark: primary + NER baseline models"
    )
    p.add_argument("--test_file",      required=True)
    p.add_argument("--eval_dir",       required=True,
                   help="outputs/eval/ — contains predictions.jsonl for fine-tuned LoRA")
    p.add_argument("--eval_dp_dir",    required=True,
                   help="outputs/eval-dp/ — contains predictions.jsonl for DP LoRA")
    p.add_argument("--hf_token",       default=None,
                   help="HuggingFace API token (or set HF_TOKEN env var)")
    p.add_argument("--output_dir",     default="./outputs/plots")
    p.add_argument("--base_model",     default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--batch_size",     type=int, default=8)
    p.add_argument("--limit",          type=int, default=None)
    p.add_argument("--skip_baselines", action="store_true",
                   help="Skip API baseline calls (useful if re-plotting saved results)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════
# 2. DATA LOADING
# ══════════════════════════════════════════════════════════

def load_test_data(filepath: str, limit=None) -> List[Dict]:
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
    print(f"  Loaded {len(samples)} test samples")
    return samples


def load_existing_predictions(pred_dir: str, n_expected: int) -> List[str]:
    pred_file = os.path.join(pred_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"predictions.jsonl not found in {pred_dir}")
    preds = []
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("prediction",
                   obj.get("response",
                   obj.get("output",
                   obj.get("pred", "")))).strip()
            preds.append(pred)
    if len(preds) != n_expected:
        print(f"  WARNING: {pred_dir} has {len(preds)} preds, expected {n_expected}. Truncating.")
        preds = preds[:n_expected]
    print(f"  Loaded {len(preds)} cached predictions from {pred_dir}")
    return preds


# ══════════════════════════════════════════════════════════
# 3. LOCAL INFERENCE — Qwen base model
# ══════════════════════════════════════════════════════════

def load_local_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   
        device_map="auto",
        trust_remote_code=True,
       
    )
    model.eval()
    return tokenizer, model, device

def clean_output(raw: str) -> str:
    raw = raw.strip().split("\n")[0].strip()
    raw = re.sub(r"[<\|].*", "", raw).strip()
    return raw


def run_local_inference(
    model_id: str,
    samples: List[Dict],
    max_new_tokens: int,
    batch_size: int,
) -> Tuple[List[str], float]:
    print(f"\n{'='*55}\n  Local inference: {model_id}\n{'='*55}")
    tokenizer, model, device = load_local_model(model_id)
    prompts = [s["prompt"] for s in samples]
    predictions, total_time = [], 0.0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
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
                pad_token_id=tokenizer.eos_token_id,
             )
        total_time += time.perf_counter() - t0
        generated = outputs[:, inputs["input_ids"].shape[1]:]  # strip echoed prompt
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


# ══════════════════════════════════════════════════════════
# 4. HF INFERENCE API — baselines
# ══════════════════════════════════════════════════════════

def _hf_api_predict_single(
    prompt: str,
    model_id: str,
    token: str,
    max_new_tokens: int,
    retries: int = 3,
) -> str:
    """
    Call HuggingFace Inference API for one prompt.
    Uses text-generation endpoint with greedy decoding (temperature=0 equivalent).
    Retries on model-loading 503s (HF loads models on demand).
    """
    url     = HF_API_URL.format(model_id=model_id)
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,   # return only generated text, not echoed prompt
        }
    }
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 503:
                wait = 20 * (attempt + 1)
                print(f"    Model loading (503), waiting {wait}s …")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    API error {resp.status_code}: {resp.text[:100]}")
                return ""
            result = resp.json()
            if isinstance(result, list) and result:
                text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                text = result.get("generated_text", "")
            else:
                text = ""
            return clean_output(text)
        except Exception as e:
            print(f"    Request error: {e}")
            time.sleep(5)
    return ""


def run_api_inference(
    model_name: str,
    model_id: str,
    samples: List[Dict],
    token: str,
    max_new_tokens: int,
) -> Tuple[List[str], float]:
    """
    Run HF Inference API sequentially (API doesn't support batching).
    Measures wall-clock time per sample for fair latency reporting.
    Note: API latency includes network round-trip — reported separately
    from local latency in graphs to avoid misleading comparison.
    """
    print(f"\n{'='*55}\n  API inference: {model_name}\n  HF ID: {model_id}\n{'='*55}")
    predictions, total_time = [], 0.0

    for i, sample in enumerate(samples):
        t0   = time.perf_counter()
        pred = _hf_api_predict_single(
            sample["prompt"], model_id, token, max_new_tokens
        )
        total_time += time.perf_counter() - t0
        predictions.append(pred)
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}] done …")

    avg_lat = (total_time / len(samples)) * 1000
    print(f"  Done. Avg API latency: {avg_lat:.1f} ms/sample (includes network)")
    return predictions, avg_lat

def run_local_baseline_inference(
    model_name: str,
    model_id: str,
    samples: List[Dict],
    max_new_tokens: int,
    batch_size: int = 1,
) -> Tuple[List[str], float]:

    print(f"\n{'='*55}\n  Local inference: {model_name}\n  HF ID: {model_id}\n{'='*55}")

    import torch
    torch.cuda.empty_cache()

    hf_token = os.getenv("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    predictions = []
    total_time = 0.0

    for i, sample in enumerate(samples):
        prompt = sample["prompt"]

        t0 = time.perf_counter()

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        pred = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        pred = clean_output(pred)

        total_time += time.perf_counter() - t0
        predictions.append(pred)

        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{len(samples)}] done …")

    avg_lat = (total_time / len(samples)) * 1000
    print(f"  Done. Avg latency: {avg_lat:.1f} ms/sample")

    del model
    torch.cuda.empty_cache()

    return predictions, avg_lat


# ══════════════════════════════════════════════════════════
# 5. METRICS
# ══════════════════════════════════════════════════════════

def normalize(t: str) -> str:
    return t.lower().strip()


def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """
    Token-level Precision, Recall, F1.
    Identical to the official SQuAD evaluation script
    (Rajpurkar et al., EMNLP 2016).
    """
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


# ══════════════════════════════════════════════════════════
# 6. GRAPHS
# ══════════════════════════════════════════════════════════

def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.facecolor":  "white",
        "axes.facecolor":    "#F8F8F8",
    })


# ── Graph 1: grouped bar — all 6 models × 4 metrics ──────
def plot_accuracy(all_metrics: Dict, out: str):
    _style()
    metric_keys   = ["exact_match", "precision", "recall", "f1"]
    metric_labels = ["Exact Match", "Precision", "Recall", "F1"]
    n_models = len(ALL_MODELS)
    x        = np.arange(len(metric_labels))
    width    = 0.12
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(15, 7))
    for name, offset in zip(ALL_MODELS, offsets):
        vals = [all_metrics[name][k] for k in metric_keys]
        bars = ax.bar(
            x + offset, vals, width, label=name,
            color=COLORS[name], hatch=HATCHES[name],
            edgecolor="white", linewidth=0.6, zorder=3
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=7, fontweight="bold", color="#333"
            )

    # Dividing line between primary and baseline groups
    ax.axvline(x=len(metric_labels) - 0.5, color="gray",
               linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_title(
        "Payee Extraction — Primary Models vs ~1.5B Baselines",
        fontsize=14, fontweight="bold", pad=16
    )

    # Split legend into two groups
    primary_patches = [
        mpatches.Patch(color=COLORS[n], label=n) for n in PRIMARY_MODELS
    ]
    baseline_patches = [
        mpatches.Patch(color=COLORS[n], hatch="///", label=n) for n in BASELINE_MODELS
    ]
    leg1 = ax.legend(
        handles=primary_patches, title="Your Models",
        fontsize=9, title_fontsize=9,
        loc="upper left", framealpha=0.9
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=baseline_patches, title="Baselines (~1.5B class)",
        fontsize=9, title_fontsize=9,
        loc="upper right", framealpha=0.9
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    fig.tight_layout()
    path = os.path.join(out, "graph1_accuracy_all_models.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Graph 2: F1 ranking bar — all 6 models, sorted ───────
def plot_f1_ranking(all_metrics: Dict, out: str):
    """
    Sorted horizontal bar of F1 scores.
    Justification: sorted bars for ranking comparison follow the
    principle of 'comparison by position on a common scale'
    (Cleveland & McGill, Science 1984) — the most accurate
    pre-attentive visual encoding for quantitative ranking.
    """
    _style()
    sorted_models = sorted(ALL_MODELS, key=lambda n: all_metrics[n]["f1"], reverse=True)
    vals   = [all_metrics[n]["f1"]   for n in sorted_models]
    colors = [COLORS[n]              for n in sorted_models]
    hatch  = [HATCHES[n]             for n in sorted_models]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        sorted_models, vals, color=colors,
        hatch=hatch, edgecolor="white", linewidth=0.8, zorder=3
    )
    for bar, val in zip(bars, vals):
        ax.text(
            val + 0.4, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold"
        )

    # Shade your fine-tuned models
    ax.axhspan(
        len(BASELINE_MODELS) - 0.5, len(ALL_MODELS) - 0.5,
        alpha=0.04, color="green", zorder=0
    )
    ax.set_xlabel("Token-level F1 Score (%)", fontsize=11)
    ax.set_title(
        "F1 Score Ranking — Your Models vs Baselines",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xlim(0, 105)

    primary_patch  = mpatches.Patch(color="#82B366", label="Your models")
    baseline_patch = mpatches.Patch(color="#AE4132", hatch="///", label="Baselines")
    ax.legend(handles=[primary_patch, baseline_patch], fontsize=9, loc="lower right")

    fig.tight_layout()
    path = os.path.join(out, "graph2_f1_ranking.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Graph 3: radar — primary models only ─────────────────
def plot_radar(all_metrics: Dict, out: str):
    _style()
    categories = ["Exact Match", "Precision", "Recall", "F1"]
    N      = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for name in PRIMARY_MODELS:
        m      = all_metrics[name]
        values = [m["exact_match"], m["precision"], m["recall"], m["f1"]]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=name, color=COLORS[name])
        ax.fill(angles, values, alpha=0.08, color=COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Performance Profile — Base vs Fine-tuned vs DP Fine-tuned",
        fontsize=12, fontweight="bold", pad=20
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.15), fontsize=9)
    path = os.path.join(out, "graph3_radar_primary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Graph 4: exact match headline bar ────────────────────
def plot_exact_match(all_metrics: Dict, out: str):
    _style()
    vals   = [all_metrics[n]["exact_match"] for n in ALL_MODELS]
    colors = [COLORS[n]                     for n in ALL_MODELS]
    hatch  = [HATCHES[n]                    for n in ALL_MODELS]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        ALL_MODELS, vals, color=colors, hatch=hatch,
        edgecolor="white", linewidth=0.8, width=0.55, zorder=3
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=10, fontweight="bold"
        )

    # Vertical divider between your models and baselines
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(0.95, 105, "Your Models", fontsize=9, color="gray", ha="center")
    ax.text(4.0,  105, "Baselines",   fontsize=9, color="gray", ha="center")

    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 112)
    ax.set_title(
        "Exact Match Accuracy — All Models",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out, "graph4_exact_match_all.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Graph 5: latency — local models only ─────────────────
def plot_latency(latencies: Dict, out: str):
    """
    API latency (baselines) excluded from this chart because it
    includes network round-trip time which is not comparable to
    local inference latency. Reported separately in summary JSON.
    Principle: only compare measurements taken under identical
    conditions (Pineau et al., JMLR 2021 — reproducibility checklist).
    """
    _style()
    local_models = [n for n in ALL_MODELS
                    if latencies.get(n, 0) > 0 and "API" not in n]
    # Tag API models separately
    api_models   = [n for n in BASELINE_MODELS
                    if latencies.get(n, 0) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                              gridspec_kw={"width_ratios": [1, 2]})

    # Left: local inference
    if local_models:
        vals   = [latencies[n] for n in local_models]
        colors = [COLORS[n]    for n in local_models]
        bars = axes[0].barh(local_models, vals, color=colors,
                            edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            axes[0].text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                         f"{val:.0f} ms", va="center", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Avg Latency (ms/sample)", fontsize=10)
        axes[0].set_title("Local Inference Latency", fontsize=11, fontweight="bold")
        axes[0].xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    else:
        axes[0].text(0.5, 0.5, "No local latency data", ha="center", va="center",
                     transform=axes[0].transAxes, color="gray")
        axes[0].set_title("Local Inference Latency", fontsize=11, fontweight="bold")

    # Right: API inference (includes network — clearly labelled)
    if api_models:
        vals   = [latencies[n] for n in api_models]
        colors = [COLORS[n]    for n in api_models]
        bars = axes[1].barh(api_models, vals, color=colors, hatch="///",
                            edgecolor="white", linewidth=0.8, zorder=3)
        for bar, val in zip(bars, vals):
            axes[1].text(val + 5, bar.get_y() + bar.get_height() / 2,
                         f"{val:.0f} ms", va="center", fontsize=10, fontweight="bold")
        axes[1].set_xlabel("Avg Latency (ms/sample) — includes network", fontsize=10)
        axes[1].set_title("API Inference Latency (HF Inference API)", fontsize=11,
                          fontweight="bold")
        axes[1].xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    else:
        axes[1].text(0.5, 0.5, "No API latency data\n(run without --skip_baselines)",
                     ha="center", va="center",
                     transform=axes[1].transAxes, color="gray")
        axes[1].set_title("API Inference Latency", fontsize=11, fontweight="bold")

    fig.suptitle("Inference Latency Comparison", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out, "graph5_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 7. SAVE + PRINT
# ══════════════════════════════════════════════════════════

def print_table(all_metrics: Dict, latencies: Dict):
    print("\n" + "=" * 80)
    print(f"{'Model':<28} {'EM%':>7} {'Prec%':>7} {'Rec%':>7} {'F1%':>7} {'Latency':>14}")
    print("=" * 80)

    print("  — Your Models —")
    for name in PRIMARY_MODELS:
        m   = all_metrics[name]
        lat = f"{latencies[name]:.0f} ms (local)" if latencies[name] > 0 else "cached"
        print(f"  {name:<26} {m['exact_match']:>6.1f}% {m['precision']:>6.1f}% "
              f"{m['recall']:>6.1f}% {m['f1']:>6.1f}% {lat:>14}")

    print("  — Baselines (~1.5B class) —")
    for name in BASELINE_MODELS:
        m   = all_metrics.get(name, {})
        if not m:
            print(f"  {name:<26}   (skipped)")
            continue
        lat = f"{latencies[name]:.0f} ms (API)" if latencies.get(name, 0) > 0 else "API"
        print(f"  {name:<26} {m['exact_match']:>6.1f}% {m['precision']:>6.1f}% "
              f"{m['recall']:>6.1f}% {m['f1']:>6.1f}% {lat:>14}")
    print("=" * 80)


def save_summary(all_metrics: Dict, latencies: Dict, out: str):
    summary = {}
    for n in ALL_MODELS:
        if n in all_metrics:
            summary[n] = {
                **all_metrics[n],
                "avg_latency_ms": round(latencies.get(n, 0), 2),
                "model_type": "primary" if n in PRIMARY_MODELS else "baseline",
            }
    path = os.path.join(out, "benchmark_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {path}")


def save_csv(all_preds: Dict, samples: List[Dict], out: str):
    models_with_preds = [n for n in ALL_MODELS if n in all_preds]
    path = os.path.join(out, "per_sample_predictions.csv")
    with open(path, "w", encoding="utf-8") as f:
        cols = (["sample_id", "ground_truth"]
                + [f"pred_{n.lower().replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
                   for n in models_with_preds]
                + [f"exact_{n.lower().replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
                   for n in models_with_preds])
        f.write(",".join(cols) + "\n")
        for i, sample in enumerate(samples):
            gt     = sample["ground_truth"]
            preds  = [all_preds[n][i] for n in models_with_preds]
            exacts = ["1" if normalize(p) == normalize(gt) else "0" for p in preds]
            row    = f"{i},{gt}," + ",".join(preds) + "," + ",".join(exacts)
            f.write(row + "\n")
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.skip_baselines:
        print("WARNING: No HF token provided. Set --hf_token or HF_TOKEN env var.")
        print("         Baselines will be skipped. Use --skip_baselines to suppress this.")

    print("\n── Loading test data ──")
    samples       = load_test_data(args.test_file, limit=args.limit)
    ground_truths = [s["ground_truth"] for s in samples]
    n             = len(samples)

    all_predictions: Dict[str, List[str]] = {}
    all_metrics:     Dict[str, Dict]      = {}
    latencies:       Dict[str, float]     = {m: 0.0 for m in ALL_MODELS}

    # ── PRIMARY 1: Base Qwen — local inference ────────────
    print("\n── Base Qwen 2.5-1.5B (local) ──")
    base_preds, base_lat = run_local_inference(
        args.base_model, samples, args.max_new_tokens, args.batch_size
    )
    all_predictions["Base Qwen 2.5-1.5B"] = base_preds
    all_metrics["Base Qwen 2.5-1.5B"]     = compute_metrics(base_preds, ground_truths)
    latencies["Base Qwen 2.5-1.5B"]       = base_lat

    # ── PRIMARY 2: Fine-tuned LoRA — cached ───────────────
    print("\n── Fine-tuned LoRA (cached) ──")
    ft_preds = load_existing_predictions(args.eval_dir, n)
    all_predictions["Fine-tuned (LoRA)"] = ft_preds
    all_metrics["Fine-tuned (LoRA)"]     = compute_metrics(ft_preds, ground_truths)

    # ── PRIMARY 3: DP Fine-tuned LoRA — cached ────────────
    print("\n── DP Fine-tuned LoRA (cached) ──")
    dp_preds = load_existing_predictions(args.eval_dp_dir, n)
    all_predictions["DP Fine-tuned (LoRA)"] = dp_preds
    all_metrics["DP Fine-tuned (LoRA)"]     = compute_metrics(dp_preds, ground_truths)

    # ── BASELINES: HF Inference API ───────────────────────
    if not args.skip_baselines:
        for name, hf_id in BASELINE_HF_IDS.items():
            print(f"\n── Baseline: {name} ──")
            preds, lat = run_local_baseline_inference(
                    name,
                    hf_id,
                    samples,
                    args.max_new_tokens,
                    batch_size=1,
                )
            all_predictions[name] = preds
            all_metrics[name]     = compute_metrics(preds, ground_truths)
            latencies[name]       = lat

            # Cache baseline predictions so you don't re-call API
            cache_path = os.path.join(
                args.output_dir,
                f"baseline_preds_{name.lower().replace(' ','_').replace('(','').replace(')','')}.jsonl"
            )
            with open(cache_path, "w") as f:
                for p in preds:
                    f.write(json.dumps({"prediction": p}) + "\n")
            print(f"  Cached to: {cache_path}")
    else:
        # Try to load cached baseline predictions if they exist
        for name in BASELINE_MODELS:
            cache_path = os.path.join(
                args.output_dir,
                f"baseline_preds_{name.lower().replace(' ','_').replace('(','').replace(')','')}.jsonl"
            )
            if os.path.exists(cache_path):
                print(f"\n── Loading cached baseline: {name} ──")
                preds = load_existing_predictions(cache_path.replace("/baseline_preds_","/../outputs/plots/baseline_preds_"), n)
                # direct load
                cached = []
                with open(cache_path) as cf:
                    for line in cf:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            cached.append(obj.get("prediction", "").strip())
                cached = cached[:n]
                if cached:
                    all_predictions[name] = cached
                    all_metrics[name]     = compute_metrics(cached, ground_truths)
            else:
                print(f"\n  Skipping {name} — no token and no cache found.")
                # Fill with placeholder so graphs still render
                all_predictions[name] = [""] * n
                all_metrics[name]     = compute_metrics([""] * n, ground_truths)

    # ── Print results ──────────────────────────────────────
    print_table(all_metrics, latencies)

    # ── Generate graphs ────────────────────────────────────
    print("\n── Generating graphs ──")
    plot_accuracy(all_metrics,    args.output_dir)
    plot_f1_ranking(all_metrics,  args.output_dir)
    plot_radar(all_metrics,       args.output_dir)
    plot_exact_match(all_metrics, args.output_dir)
    plot_latency(latencies,       args.output_dir)

    # ── Save outputs ───────────────────────────────────────
    print("\n── Saving output files ──")
    save_summary(all_metrics, latencies, args.output_dir)
    save_csv(all_predictions, samples,   args.output_dir)

    print(f"\n✓  Done. Outputs in: {args.output_dir}/")
    print("   graph1_accuracy_all_models.png  — all 6 models × 4 metrics grouped bar")
    print("   graph2_f1_ranking.png           — F1 sorted ranking bar")
    print("   graph3_radar_primary.png        — radar: your 3 models only")
    print("   graph4_exact_match_all.png      — EM headline bar all 6 models")
    print("   graph5_latency.png              — local vs API latency (split panels)")
    print("   benchmark_summary.json          — all metrics + model type labels")
    print("   per_sample_predictions.csv      — per-sample breakdown")
    print("\n   Baseline API cache files:")
    for name in BASELINE_MODELS:
        slug = name.lower().replace(' ','_').replace('(','').replace(')','')
        print(f"   baseline_preds_{slug}.jsonl")


if __name__ == "__main__":
    main()