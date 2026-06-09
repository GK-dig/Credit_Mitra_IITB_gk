"""
Membership Inference Attack (MIA) — DP-QLoRA Payee Extraction
==============================================================
Repo layout expected (matches gk-dig-credit_mitra_iitb_gk):

    benchmark_payee/
        benchmark_payee.py            ← dataset loader reused here
        outputs/
            payee-lora/               ← non-private LoRA adapter
            payee-lora-dp/            ← DP-trained adapter  ← PRIMARY target
            eval-dp/metrics.json      ← existing DP eval (for comparison)

    dp/
        budget_tracker.py
        mechanisms.py
        stage2_payee.py               ← PayeeExtractionStage used for inference

    Fine-tuning/scripts/
        evaluate.py                   ← metric helpers reused

Run from repo root:
    python mia_dp_qlora.py \
        --adapter   benchmark_payee/outputs/payee-lora-dp \
        --train_data  <path-to-train.jsonl> \
        --val_data    <path-to-val.jsonl>   \
        --output_dir  benchmark_payee/outputs/eval-mia

Requirements (already in Fine-tuning/requirements.txt + dp extras):
    pip install torch transformers peft bitsandbytes scikit-learn matplotlib tqdm opacus
"""

import argparse, json, os, random, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report,
    precision_recall_curve, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════
#  Paths — mirrors benchmark_payee/outputs structure
# ══════════════════════════════════════════════════════════════════

REPO_ROOT    = Path(__file__).resolve().parent          # repo root when run from there
ADAPTER_DP   = REPO_ROOT / "benchmark_payee/outputs/payee-lora-dp"
ADAPTER_LORA = REPO_ROOT / "benchmark_payee/outputs/payee-lora"
DP_LOGS      = ADAPTER_DP / "dp_training_logs.json"
EVAL_DP_JSON = REPO_ROOT / "benchmark_payee/outputs/eval-dp/metrics.json"
PLOTS_DIR    = REPO_ROOT / "benchmark_payee/outputs/plots/evaluation"

# Base model: Qwen2-1.5B (from adapter_config.json in payee-lora-dp)
BASE_MODEL_ID = "Qwen/Qwen2-1.5B"
MAX_LENGTH    = 64       # paper §IV-A: sequences truncated to 64 tokens
LORA_RANK     = 8        # paper §III-B
LORA_ALPHA    = 16


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MIA for payee-lora-dp adapter")
    p.add_argument("--adapter",     default=str(ADAPTER_DP),
                   help="Path to DP-trained LoRA adapter dir")
    p.add_argument("--base_model",  default=BASE_MODEL_ID)
    p.add_argument("--train_data",  required=True,
                   help="JSONL used during fine-tuning  → member set")
    p.add_argument("--val_data",    required=True,
                   help="JSONL NOT seen during training → non-member set")
    p.add_argument("--max_samples", type=int, default=500,
                   help="Samples per class (balanced)")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--output_dir",  default="benchmark_payee/outputs/eval-mia")
    p.add_argument("--also_eval_nondp", action="store_true",
                   help="Run MIA on the non-private payee-lora adapter too (baseline)")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
#  Dataset  (mirrors benchmark_payee.py data format)
# ══════════════════════════════════════════════════════════════════

def load_jsonl(path: str, max_n: int, seed: int) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:max_n]


def fmt_record(rec: dict) -> str:
    """
    Mirrors the prompt template used in Fine-tuning/scripts/prepare_dataset.py.
    Falls back gracefully for both instruction-response and raw-text records.
    """
    if "instruction" in rec and "response" in rec:
        return rec["instruction"].strip() + "\n" + rec["response"].strip()
    if "input" in rec and "output" in rec:
        return rec["input"].strip() + "\n" + rec["output"].strip()
    if "text" in rec:
        return rec["text"].strip()
    # last resort: concatenate all string values
    return " ".join(str(v) for v in rec.values() if isinstance(v, str))


class PayeeDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.encodings = []
        for rec in records:
            text = fmt_record(rec)
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            self.encodings.append(enc["input_ids"].squeeze(0))

    def __len__(self):  return len(self.encodings)
    def __getitem__(self, i): return self.encodings[i]


# ══════════════════════════════════════════════════════════════════
#  Model loader  (NF4 quant + LoRA, exact paper setup)
# ══════════════════════════════════════════════════════════════════

def load_qlora_model(base_id: str, adapter_path: str):
    """
    Load Qwen2-1.5B in NF4 4-bit with double-quantization (paper §III-A)
    then attach LoRA adapter weights.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_id, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


# ══════════════════════════════════════════════════════════════════
#  Per-sample cross-entropy loss  (MIA signal)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def per_sample_loss(model, loader, device, pad_id: int) -> np.ndarray:
    """
    Token-averaged CE loss per sample.
    Members → lower loss (model memorised them).
    Non-members → higher loss.
    """
    losses = []
    for batch in tqdm(loader, desc="  loss", leave=False):
        ids = batch.to(device)                      # (B, L)
        labels = ids.clone()
        labels[labels == pad_id] = -100             # ignore padding

        out = model(input_ids=ids, labels=labels)
        logits = out.logits                         # (B, L, V)

        # Per-sample loss (model.forward returns mean; recompute manually)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)                  # (B, L-1)

        mask = (shift_labels != -100).float()
        seq_loss = (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        losses.extend(seq_loss.cpu().tolist())
    return np.array(losses)


# ══════════════════════════════════════════════════════════════════
#  MIA evaluation + plots  (output goes into eval-mia/ like eval-dp/)
# ══════════════════════════════════════════════════════════════════

def evaluate_mia(
    member_loss: np.ndarray,
    nonmember_loss: np.ndarray,
    tag: str,           # "dp" or "nondp"
    out_dir: Path,
    existing_dp_metrics: dict | None = None,
) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    y_true  = np.concatenate([np.ones(len(member_loss)), np.zeros(len(nonmember_loss))])
    scores  = np.concatenate([-member_loss, -nonmember_loss])   # higher = more likely member

    auc     = roc_auc_score(y_true, scores)
    ap      = average_precision_score(y_true, scores)
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, scores)

    # Optimal threshold (Youden's J)
    j       = tpr_arr - fpr_arr
    best_i  = int(np.argmax(j))
    best_thr = thresholds[best_i]
    y_pred  = (scores >= best_thr).astype(int)
    acc     = accuracy_score(y_true, y_pred)

    # TPR @ FPR ≤ 0.1 — standard privacy operating point
    mask_low = fpr_arr <= 0.10
    tpr_at_10 = float(tpr_arr[mask_low][-1]) if mask_low.any() else 0.0

    # ── console ──────────────────────────────────────────────────
    label = "DP-QLoRA  (payee-lora-dp)" if tag == "dp" else "LoRA baseline (payee-lora)"
    print(f"\n{'═'*58}")
    print(f"  MIA Results — {label}")
    print(f"{'═'*58}")
    print(f"  MIA-AUC              : {auc:.4f}")
    print(f"  Avg Precision (AP)   : {ap:.4f}")
    print(f"  Accuracy @ opt.thr   : {acc:.4f}  (loss thr = {-best_thr:.4f})")
    print(f"  TPR @ FPR ≤ 0.10     : {tpr_at_10:.4f}")
    if existing_dp_metrics and tag == "dp":
        ref = existing_dp_metrics.get("mia_auc") or existing_dp_metrics.get("roc_auc")
        if ref:
            print(f"  eval-dp/metrics ref  : {ref:.4f}  ({'✅ match' if abs(auc-ref)<0.05 else '⚠ diff'})")
    print(f"{'═'*58}")
    print(f"  Member   loss: mean={member_loss.mean():.4f}  std={member_loss.std():.4f}")
    print(f"  Non-mbr  loss: mean={nonmember_loss.mean():.4f}  std={nonmember_loss.std():.4f}")
    print(classification_report(y_true, y_pred, target_names=["Non-Member", "Member"]))

    # ── save metrics.json  (same schema as eval-dp/metrics.json) ─
    metrics = {
        "tag": tag,
        "mia_auc": round(float(auc), 6),
        "avg_precision": round(float(ap), 6),
        "accuracy": round(float(acc), 6),
        "tpr_at_fpr_0.10": round(tpr_at_10, 6),
        "optimal_loss_threshold": round(float(-best_thr), 6),
        "n_members": int(len(member_loss)),
        "n_nonmembers": int(len(nonmember_loss)),
        "member_loss_mean": round(float(member_loss.mean()), 6),
        "member_loss_std":  round(float(member_loss.std()),  6),
        "nonmember_loss_mean": round(float(nonmember_loss.mean()), 6),
        "nonmember_loss_std":  round(float(nonmember_loss.std()),  6),
        "dp_interpretation": _interpret(auc),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── ROC curve  (naming: 12_roc_*.eps style from plots/evaluation/) ───
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_arr, tpr_arr, lw=2, color="royalblue",
            label=f"{label}\nAUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
    ax.scatter(fpr_arr[best_i], tpr_arr[best_i], s=80, color="red", zorder=5,
               label=f"Opt. threshold (loss={-best_thr:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"MIA ROC — {label}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / f"roc_mia_{tag}.eps"
    fig.savefig(roc_path, format="eps"); fig.savefig(str(roc_path).replace(".eps",".png"), dpi=150)
    plt.close()

    # ── PR curve ──────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"MIA Precision-Recall — {label}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = out_dir / f"pr_mia_{tag}.eps"
    fig.savefig(pr_path, format="eps"); fig.savefig(str(pr_path).replace(".eps",".png"), dpi=150)
    plt.close()

    # ── Loss distribution histogram ───────────────────────────────
    all_min = min(member_loss.min(), nonmember_loss.min())
    all_max = max(member_loss.max(), nonmember_loss.max())
    bins = np.linspace(all_min, all_max, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(member_loss,    bins=bins, alpha=0.6, color="steelblue",
            density=True, label=f"Members (train) n={len(member_loss)}")
    ax.hist(nonmember_loss, bins=bins, alpha=0.6, color="coral",
            density=True, label=f"Non-members (val) n={len(nonmember_loss)}")
    ax.axvline(-best_thr, ls="--", color="black", lw=1.5,
               label=f"Decision threshold={-best_thr:.3f}")
    ax.set_xlabel("Cross-Entropy Loss"); ax.set_ylabel("Density")
    ax.set_title(f"Loss Distribution — {label}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    dist_path = out_dir / f"loss_dist_{tag}.png"
    fig.savefig(dist_path, dpi=150)
    plt.close()

    print(f"\n  Saved → {out_dir}/metrics.json")
    print(f"          {roc_path.name}  |  {pr_path.name}  |  {dist_path.name}")
    return metrics


def _interpret(auc: float) -> str:
    if auc < 0.55:
        return "✅ Strong DP — attack barely above random"
    if auc < 0.65:
        return "✅ Good DP protection (paper DP-LoRA target: 0.65)"
    if auc < 0.75:
        return "⚠ Moderate — consider reducing ε or increasing noise multiplier"
    if auc < 0.85:
        return "❌ Weak DP — MIA succeeds; model memorising training data"
    return "❌ Very weak — equivalent to no privacy protection"


# ══════════════════════════════════════════════════════════════════
#  Optional: read dp_training_logs.json for context
# ══════════════════════════════════════════════════════════════════

def print_dp_logs(logs_path: Path):
    if not logs_path.exists():
        return
    try:
        logs = json.loads(logs_path.read_text())
        epsilon = logs.get("epsilon") or logs.get("final_epsilon") or logs.get("eps")
        delta   = logs.get("delta")
        noise   = logs.get("noise_multiplier") or logs.get("sigma")
        clip    = logs.get("max_grad_norm") or logs.get("clipping_threshold")
        print(f"\n  DP training params from dp_training_logs.json:")
        print(f"    ε = {epsilon}  |  δ = {delta}  |  σ = {noise}  |  C = {clip}")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    out_dir = Path(args.output_dir)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Adapter: {args.adapter}")

    # Print DP training context if available
    print_dp_logs(Path(args.adapter) / "dp_training_logs.json")

    # Load existing DP eval metrics for reference comparison
    existing_dp = None
    if EVAL_DP_JSON.exists():
        try:
            existing_dp = json.loads(EVAL_DP_JSON.read_text())
            print(f"\n  Reference eval-dp/metrics.json: {existing_dp}")
        except Exception:
            pass

    # ── Load data ──────────────────────────────────────────────────
    print(f"\nLoading member data    : {args.train_data}")
    members     = load_jsonl(args.train_data, args.max_samples, args.seed)
    print(f"Loading non-member data: {args.val_data}")
    nonmembers  = load_jsonl(args.val_data,   args.max_samples, args.seed)
    n = min(len(members), len(nonmembers))
    members, nonmembers = members[:n], nonmembers[:n]
    print(f"Balanced n = {n} per class")

    # ── Load DP model ──────────────────────────────────────────────
    print(f"\nLoading DP-QLoRA model ({BASE_MODEL_ID} + {args.adapter})…")
    model_dp, tokenizer = load_qlora_model(args.base_model, args.adapter)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds_m  = PayeeDataset(members,    tokenizer, MAX_LENGTH)
    ds_nm = PayeeDataset(nonmembers, tokenizer, MAX_LENGTH)
    dl_m  = DataLoader(ds_m,  batch_size=args.batch_size, shuffle=False)
    dl_nm = DataLoader(ds_nm, batch_size=args.batch_size, shuffle=False)

    print("\nComputing member losses (DP model)…")
    loss_m_dp  = per_sample_loss(model_dp, dl_m,  device, pad_id)
    print("Computing non-member losses (DP model)…")
    loss_nm_dp = per_sample_loss(model_dp, dl_nm, device, pad_id)

    metrics_dp = evaluate_mia(loss_m_dp, loss_nm_dp, "dp", out_dir / "dp", existing_dp)

    # ── Optional: baseline non-DP LoRA ────────────────────────────
    if args.also_eval_nondp and Path(ADAPTER_LORA).exists():
        print(f"\nLoading non-DP LoRA model ({ADAPTER_LORA})…")
        model_nd, _ = load_qlora_model(args.base_model, str(ADAPTER_LORA))
        ds_m2  = PayeeDataset(members,    tokenizer, MAX_LENGTH)
        ds_nm2 = PayeeDataset(nonmembers, tokenizer, MAX_LENGTH)
        dl_m2  = DataLoader(ds_m2,  batch_size=args.batch_size, shuffle=False)
        dl_nm2 = DataLoader(ds_nm2, batch_size=args.batch_size, shuffle=False)

        print("\nComputing member losses (non-DP LoRA)…")
        loss_m_nd  = per_sample_loss(model_nd, dl_m2,  device, pad_id)
        print("Computing non-member losses (non-DP LoRA)…")
        loss_nm_nd = per_sample_loss(model_nd, dl_nm2, device, pad_id)

        metrics_nd = evaluate_mia(loss_m_nd, loss_nm_nd, "nondp", out_dir / "nondp")

        # ── Side-by-side comparison plot ──────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        for tag, ml, nml, col, lbl in [
            ("dp",    loss_m_dp, loss_nm_dp, "steelblue",  f"DP-QLoRA  (AUC={metrics_dp['mia_auc']:.3f})"),
            ("nondp", loss_m_nd, loss_nm_nd, "darkorange", f"LoRA base (AUC={metrics_nd['mia_auc']:.3f})"),
        ]:
            y_true = np.concatenate([np.ones(len(ml)), np.zeros(len(nml))])
            scores = np.concatenate([-ml, -nml])
            fpr, tpr, _ = roc_curve(y_true, scores)
            ax.plot(fpr, tpr, lw=2, color=col, label=lbl)
        ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("MIA ROC — DP-QLoRA vs LoRA baseline\n(payee-lora-dp vs payee-lora)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "roc_comparison_dp_vs_nondp.png", dpi=150)
        plt.close()
        print(f"\n  Comparison plot → {out_dir}/roc_comparison_dp_vs_nondp.png")

    print("\n✅  MIA complete.  Results in:", out_dir)


if __name__ == "__main__":
    main()