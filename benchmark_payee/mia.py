"""
Advanced Membership Inference Attack (MIA) for DP-QLoRA Payee Extraction
=========================================================================
Implements four attack tiers in increasing strength:

  1. Loss Threshold         — your original baseline
  2. Quantile Calibration   — per-sample difficulty correction
  3. LiRA (online)          — likelihood-ratio with per-sample calibration
                              using N shadow models trained on random splits
  4. Reference-Model Attack — ratio of target loss vs reference model loss
                              (no shadow training needed, strong in practice)

Gold standard: LiRA (Carlini et al., 2022 — "Membership Inference Attacks
From First Principles").  Reference-model attack is nearly as strong with
far less compute (Ye et al., 2022 — "Enhanced MIA").

Usage:
    python mia_advanced.py --project_root .

    # Faster (skip shadow models, still gets reference-model attack):
    python mia_advanced.py --project_root . --n_shadows 0

    # Full LiRA with 8 shadow models:
    python mia_advanced.py --project_root . --n_shadows 8

Reads:
    <root>/data/train.jsonl                        → member set
    <root>/data/val.jsonl                          → non-member set
    <root>/outputs_1/payee-lora-dp/                → DP adapter
    <root>/outputs_8/outputs/payee-lora/           → non-DP adapter

Writes to:
    <root>/outputs_mia_advanced/
        metrics_*.json
        fig_roc_all_attacks.pdf
        fig_loss_distributions.pdf
        fig_attack_comparison.pdf
        fig_lira_scores.pdf
"""

import argparse, json, random, gc, copy, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import PeftModel, LoraConfig, get_peft_model
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as scipy_norm
from tqdm import tqdm

# ── aesthetics ────────────────────────────────────────────────────────────────
PALETTE = {
    "dp":      "#D79B00",   # amber  — DP model
    "nondp":   "#6C8EBF",   # steel  — non-DP model
    "random":  "#AAAAAA",   # grey   — random baseline
    "member":  "#6C8EBF",   # blue   — members
    "nonmem":  "#AE4132",   # red    — non-members
}
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
    "axes.facecolor":   "#F9F9F9",
    "axes.grid":        True,
    "grid.color":       "#E0E0E0",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 64


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", default=".")
    p.add_argument("--max_samples",  type=int, default=500)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--n_shadows",    type=int, default=4,
                   help="Number of shadow models for LiRA (0 = skip LiRA, "
                        "use reference-model attack only). More = better but slower.")
    p.add_argument("--shadow_steps", type=int, default=200,
                   help="Training steps per shadow model (keep low for speed)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: Path, max_n: int, seed: int) -> list:
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
    if "prompt" in rec and "response" in rec:
        return rec["prompt"].strip() + "\n" + rec["response"].strip()
    if "instruction" in rec and "response" in rec:
        return rec["instruction"].strip() + "\n" + rec["response"].strip()
    if "input" in rec and "output" in rec:
        return rec["input"].strip() + "\n" + rec["output"].strip()
    if "text" in rec:
        return rec["text"].strip()
    return " ".join(str(v) for v in rec.values() if isinstance(v, str))


class TextDataset(Dataset):
    def __init__(self, records: list, tokenizer, max_length: int):
        self.ids = []
        for rec in records:
            text = fmt_record(rec)
            enc  = tokenizer(
                text, max_length=max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            self.ids.append(enc["input_ids"].squeeze(0))

    def __len__(self): return len(self.ids)
    def __getitem__(self, i): return self.ids[i]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(adapter_path: Path):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tok


# ─────────────────────────────────────────────────────────────────────────────
# Loss computation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_losses(model, loader, pad_id: int, desc="computing losses") -> np.ndarray:
    """Token-averaged cross-entropy per sample."""
    losses = []
    for batch in tqdm(loader, desc=f"  {desc}", leave=False):
        ids    = batch.to(next(model.parameters()).device)
        labels = ids.clone()
        labels[labels == pad_id] = -100

        out    = model(input_ids=ids, labels=labels)
        logits = out.logits

        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), reduction="none",
        ).view(shift_labels.shape)

        mask     = (shift_labels != -100).float()
        seq_loss = (token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        losses.extend(seq_loss.cpu().tolist())

    return np.array(losses)


# ─────────────────────────────────────────────────────────────────────────────
# Attack 1 — Loss threshold (your original)
# ─────────────────────────────────────────────────────────────────────────────
def attack_loss_threshold(member_loss, nonmember_loss):
    """Score = -loss. Lower loss → more likely member."""
    y_true = np.concatenate([np.ones(len(member_loss)),
                              np.zeros(len(nonmember_loss))])
    scores = np.concatenate([-member_loss, -nonmember_loss])
    return y_true, scores


# ─────────────────────────────────────────────────────────────────────────────
# Attack 2 — Quantile / difficulty calibration
# ─────────────────────────────────────────────────────────────────────────────
def attack_quantile(member_loss, nonmember_loss, all_loss_pool):
    """
    Calibrate each sample's loss by its empirical quantile across the
    full population.  Corrects for 'easy' samples that have low loss
    regardless of membership.
    Ref: Carlini et al. 2022 §3.1
    """
    all_losses = np.concatenate([member_loss, nonmember_loss, all_loss_pool])
    def to_quantile(x):
        return np.array([np.mean(all_losses <= v) for v in x])

    # convert to rank-score; members should have lower quantile
    m_q  = to_quantile(member_loss)
    nm_q = to_quantile(nonmember_loss)

    y_true = np.concatenate([np.ones(len(m_q)), np.zeros(len(nm_q))])
    scores = np.concatenate([-m_q, -nm_q])   # lower quantile → more likely member
    return y_true, scores


# ─────────────────────────────────────────────────────────────────────────────
# Attack 3 — Reference model attack (no shadow training needed)
# ─────────────────────────────────────────────────────────────────────────────
def attack_reference_model(target_member_loss, target_nonmember_loss,
                            ref_member_loss,    ref_nonmember_loss):
    """
    Score = loss_ref(x) - loss_target(x)
    If target memorised x, its loss is lower than the reference → positive score.
    Ref: Ye et al., 2022 "Enhanced MIA"
    """
    ratio_m  = ref_member_loss    - target_member_loss
    ratio_nm = ref_nonmember_loss - target_nonmember_loss

    y_true = np.concatenate([np.ones(len(ratio_m)), np.zeros(len(ratio_nm))])
    scores = np.concatenate([ratio_m, ratio_nm])
    return y_true, scores


# ─────────────────────────────────────────────────────────────────────────────
# Attack 4 — LiRA (online variant, Carlini et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────
def train_shadow_model(base_model_path: Path, adapter_cfg: LoraConfig,
                       train_dataset: TextDataset,
                       tokenizer, steps: int, seed: int):
    """
    Fine-tune a shadow LoRA on a random 50% subset of train+val data.
    Returns the trained model WITHOUT offloading the base — caller must del.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model = get_peft_model(base, adapter_cfg)

    # Minimal training args — just enough for a shadow model
    targs = TrainingArguments(
        output_dir=str(base_model_path / f"shadow_{seed}"),
        max_steps=steps,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        logging_steps=9999,
        save_strategy="no",
        report_to="none",
        seed=seed,
        fp16=True,
    )

    class LMTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            ids    = inputs["input_ids"]
            labels = ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            out    = model(input_ids=ids, labels=labels)
            return (out.loss, out) if return_outputs else out.loss

    trainer = LMTrainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.eval()
    return model


def lira_attack(target_member_loss:    np.ndarray,
                target_nonmember_loss: np.ndarray,
                shadow_member_losses:    list,   # list of np.ndarray (one per shadow)
                shadow_nonmember_losses: list):
    """
    LiRA online attack:
      For each sample x:
        - Collect losses from shadows that INCLUDED x in training  → "in" dist
        - Collect losses from shadows that EXCLUDED x              → "out" dist
        - Fit Gaussians to each, compute log-likelihood ratio score

    Since we use a 50/50 random split per shadow, each sample appears in ~50%
    of the shadows (Bernoulli(0.5) inclusion), giving us both in/out samples.
    """
    n_shadows = len(shadow_member_losses)
    n_members = len(target_member_loss)
    n_nonmems = len(target_nonmember_loss)
    n_total   = n_members + n_nonmems

    # shadow_losses[shadow_idx, sample_idx] — stack all shadow losses
    # Each shadow was trained on a random 50% split of (members ∪ non-members)
    shadow_m_mat  = np.stack(shadow_member_losses,    axis=1)  # (n_m, n_shadows)
    shadow_nm_mat = np.stack(shadow_nonmember_losses, axis=1)  # (n_nm, n_shadows)

    def lira_score(target_loss, shadow_mat):
        """
        For each sample, use half the shadows as "in" and half as "out"
        (random assignment per sample simulates 50/50 inclusion).
        """
        scores = []
        rng = np.random.default_rng(42)
        for i in range(len(target_loss)):
            idx    = rng.choice(n_shadows, n_shadows, replace=False)
            half   = n_shadows // 2
            in_idx  = idx[:half]
            out_idx = idx[half:]

            in_losses  = shadow_mat[i, in_idx]
            out_losses = shadow_mat[i, out_idx]

            mu_in,  sig_in  = in_losses.mean(),  in_losses.std() + 1e-6
            mu_out, sig_out = out_losses.mean(), out_losses.std() + 1e-6

            tl = target_loss[i]
            log_p_in  = scipy_norm.logpdf(tl, mu_in,  sig_in)
            log_p_out = scipy_norm.logpdf(tl, mu_out, sig_out)
            scores.append(log_p_in - log_p_out)
        return np.array(scores)

    member_scores    = lira_score(target_member_loss,    shadow_m_mat)
    nonmember_scores = lira_score(target_nonmember_loss, shadow_nm_mat)

    y_true = np.concatenate([np.ones(n_members), np.zeros(n_nonmems)])
    scores = np.concatenate([member_scores, nonmember_scores])
    return y_true, scores


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, scores, label):
    auc      = roc_auc_score(y_true, scores)
    ap       = average_precision_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    j        = tpr - fpr
    best_i   = int(np.argmax(j))
    y_pred   = (scores >= thresholds[best_i]).astype(int)
    acc      = accuracy_score(y_true, y_pred)

    mask_10  = fpr <= 0.10
    tpr10    = float(tpr[mask_10][-1]) if mask_10.any() else 0.0

    m = {
        "label":         label,
        "mia_auc":       round(float(auc),  4),
        "avg_precision": round(float(ap),   4),
        "accuracy":      round(float(acc),  4),
        "tpr_at_fpr10":  round(tpr10,       4),
        "verdict":       verdict(auc),
    }
    return m, fpr, tpr


def verdict(auc):
    if auc < 0.55: return "Strong privacy — near random"
    if auc < 0.65: return "Good privacy"
    if auc < 0.75: return "Moderate — partial memorisation"
    if auc < 0.85: return "Weak — significant memorisation"
    return "Very weak — near-full memorisation"


def print_metrics(m):
    print(f"\n{'─'*55}")
    print(f"  {m['label']}")
    print(f"{'─'*55}")
    print(f"  MIA-AUC        : {m['mia_auc']:.4f}")
    print(f"  Avg Precision  : {m['avg_precision']:.4f}")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  TPR @ FPR≤0.10 : {m['tpr_at_fpr10']:.4f}")
    print(f"  Verdict        : {m['verdict']}")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_all(attack_results: dict, model_tag: str, out: Path):
    """One ROC curve per attack type for a single model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0,1],[0,1], color=PALETTE["random"], lw=1.2, ls="--",
            label="Random (AUC=0.500)")

    linestyles = ["-", "--", "-.", ":"]
    colors     = ["#D79B00","#6C8EBF","#AE4132","#82B366"]

    for i, (atk_name, (metrics, fpr, tpr)) in enumerate(attack_results.items()):
        ax.plot(fpr, tpr, lw=2.2,
                color=colors[i % len(colors)],
                ls=linestyles[i % len(linestyles)],
                label=f"{atk_name}  (AUC={metrics['mia_auc']:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"MIA — All Attacks — {model_tag}\n"
                 "Lower AUC = stronger privacy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.tight_layout()
    fname = out / f"fig_roc_{model_tag.replace(' ','_').lower()}.pdf"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


def plot_attack_comparison(dp_results: dict, nondp_results: dict, out: Path):
    """
    Bar chart: strongest attack AUC per model — the key takeaway figure.
    """
    attack_names = list(dp_results.keys())
    x     = np.arange(len(attack_names))
    width = 0.35

    dp_aucs    = [dp_results[k][0]["mia_auc"]    for k in attack_names]
    nondp_aucs = [nondp_results[k][0]["mia_auc"] for k in attack_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, nondp_aucs, width,
                   label="QLoRA (non-private)", color=PALETTE["nondp"],
                   edgecolor="white", zorder=3)
    bars2 = ax.bar(x + width/2, dp_aucs, width,
                   label="DP-QLoRA (ε≈1)", color=PALETTE["dp"],
                   edgecolor="white", zorder=3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.axhline(0.5, color=PALETTE["random"], lw=1.4, ls="--",
               label="Random baseline (0.500)", zorder=2)
    ax.axhline(0.6, color="#FF7043", lw=0.8, ls=":", alpha=0.7,
               label="Concern threshold (0.600)", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_names, fontsize=10)
    ax.set_ylabel("MIA-AUC", fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Attack Strength Comparison — DP vs Non-DP\n"
                 "Higher bar = stronger attack found / more memorisation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_attack_comparison.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_attack_comparison.pdf")


def plot_loss_distributions(dp_m, dp_nm, nondp_m, nondp_nm, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, m_loss, nm_loss, tag, color in [
        (axes[0], nondp_m, nondp_nm, "QLoRA (non-private)", PALETTE["nondp"]),
        (axes[1], dp_m,   dp_nm,    "DP-QLoRA (ε≈1)",      PALETTE["dp"]),
    ]:
        lo = min(m_loss.min(), nm_loss.min())
        hi = max(m_loss.max(), nm_loss.max())
        bins = np.linspace(lo, hi, 50)

        ax.hist(m_loss,  bins=bins, alpha=0.6, color=PALETTE["member"],
                density=True, label=f"Members  (n={len(m_loss)})")
        ax.hist(nm_loss, bins=bins, alpha=0.6, color=PALETTE["nonmem"],
                density=True, label=f"Non-members  (n={len(nm_loss)})")

        # KDE overlay
        for arr, c in [(m_loss, PALETTE["member"]), (nm_loss, PALETTE["nonmem"])]:
            kde_x = np.linspace(lo, hi, 300)
            bw    = 1.06 * arr.std() * len(arr)**(-0.2)
            kde_y = np.mean(
                scipy_norm.pdf(kde_x[:, None], arr[None, :], bw), axis=1
            )
            ax.plot(kde_x, kde_y, color=c, lw=2.0)

        overlap = np.minimum(
            np.histogram(m_loss,  bins=bins, density=True)[0],
            np.histogram(nm_loss, bins=bins, density=True)[0],
        ).sum() * (bins[1]-bins[0])
        ax.set_title(f"{tag}\nDistribution overlap = {overlap:.3f}  "
                     "(higher overlap = better privacy)",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Cross-Entropy Loss", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Loss Distributions: Members vs Non-Members\n"
                 "More overlap = model cannot distinguish training data",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "fig_loss_distributions.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_loss_distributions.pdf")


def plot_lira_scores(m_scores, nm_scores, model_tag, out: Path):
    """LiRA score distribution — in vs out samples."""
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(m_scores.min(), nm_scores.min())
    hi = max(m_scores.max(), nm_scores.max())
    bins = np.linspace(lo, hi, 60)

    ax.hist(m_scores,  bins=bins, alpha=0.6, color=PALETTE["member"],
            density=True, label=f"Members  (n={len(m_scores)})")
    ax.hist(nm_scores, bins=bins, alpha=0.6, color=PALETTE["nonmem"],
            density=True, label=f"Non-members  (n={len(nm_scores)})")
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Decision boundary (score=0)")

    ax.set_xlabel("LiRA Score  (log p_in − log p_out)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"LiRA Score Distribution — {model_tag}\n"
                 "Larger separation = more memorisation",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = out / f"fig_lira_{model_tag.replace(' ','_').lower()}.pdf"
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    out  = root / "outputs_mia_advanced"
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice       : {device}")
    print(f"Project root : {root}")
    print(f"Output dir   : {out}")
    print(f"Shadow models: {args.n_shadows}")

    # ── Load data ─────────────────────────────────────────────────
    members    = load_jsonl(root / "data/train.jsonl", args.max_samples, args.seed)
    nonmembers = load_jsonl(root / "data/val.jsonl",   args.max_samples, args.seed)
    n = min(len(members), len(nonmembers))
    members, nonmembers = members[:n], nonmembers[:n]
    print(f"\nBalanced at n={n} per class")

    # ── Load DP model and get losses ──────────────────────────────
    print("\n[1/4] Loading DP-QLoRA model...")
    model_dp, tokenizer = load_model(root / "outputs_1/payee-lora-dp")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    ds_m  = TextDataset(members,    tokenizer, MAX_LENGTH)
    ds_nm = TextDataset(nonmembers, tokenizer, MAX_LENGTH)
    dl_m  = DataLoader(ds_m,  batch_size=args.batch_size, shuffle=False)
    dl_nm = DataLoader(ds_nm, batch_size=args.batch_size, shuffle=False)

    dp_m_loss  = compute_losses(model_dp, dl_m,  pad_id, "DP — members")
    dp_nm_loss = compute_losses(model_dp, dl_nm, pad_id, "DP — non-members")
    del model_dp; gc.collect(); torch.cuda.empty_cache()

    # ── Load non-DP model and get losses ──────────────────────────
    print("\n[2/4] Loading non-DP LoRA model...")
    model_nd, _ = load_model(root / "outputs_8/outputs/payee-lora")

    nd_m_loss  = compute_losses(model_nd, dl_m,  pad_id, "nonDP — members")
    nd_nm_loss = compute_losses(model_nd, dl_nm, pad_id, "nonDP — non-members")
    del model_nd; gc.collect(); torch.cuda.empty_cache()

    # ── Attack 1: Loss Threshold ───────────────────────────────────
    print("\n[3/4] Running attacks...")
    print("  Attack 1: Loss Threshold")
    dp_attacks   = {}
    nondp_attacks = {}

    y, s = attack_loss_threshold(dp_m_loss, dp_nm_loss)
    dp_attacks["Loss Threshold"] = compute_metrics(y, s, "DP — Loss Threshold")
    print_metrics(dp_attacks["Loss Threshold"][0])

    y, s = attack_loss_threshold(nd_m_loss, nd_nm_loss)
    nondp_attacks["Loss Threshold"] = compute_metrics(y, s, "nonDP — Loss Threshold")
    print_metrics(nondp_attacks["Loss Threshold"][0])

    # ── Attack 2: Quantile Calibration ────────────────────────────
    print("\n  Attack 2: Quantile Calibration")
    pool = np.concatenate([dp_m_loss, dp_nm_loss, nd_m_loss, nd_nm_loss])

    y, s = attack_quantile(dp_m_loss, dp_nm_loss, pool)
    dp_attacks["Quantile Calib"] = compute_metrics(y, s, "DP — Quantile")
    print_metrics(dp_attacks["Quantile Calib"][0])

    y, s = attack_quantile(nd_m_loss, nd_nm_loss, pool)
    nondp_attacks["Quantile Calib"] = compute_metrics(y, s, "nonDP — Quantile")
    print_metrics(nondp_attacks["Quantile Calib"][0])

    # ── Attack 3: Reference Model Attack ──────────────────────────
    print("\n  Attack 3: Reference Model Attack")
    # Use DP model as reference for non-DP and vice versa
    y, s = attack_reference_model(dp_m_loss,  dp_nm_loss,
                                  nd_m_loss,  nd_nm_loss)
    dp_attacks["Reference Model"] = compute_metrics(y, s, "DP — Ref Model")
    print_metrics(dp_attacks["Reference Model"][0])

    y, s = attack_reference_model(nd_m_loss,  nd_nm_loss,
                                  dp_m_loss,  dp_nm_loss)
    nondp_attacks["Reference Model"] = compute_metrics(y, s, "nonDP — Ref Model")
    print_metrics(nondp_attacks["Reference Model"][0])

    # ── Attack 4: LiRA (optional — needs shadow models) ───────────
    if args.n_shadows > 0:
        print(f"\n  Attack 4: LiRA ({args.n_shadows} shadow models × "
              f"{args.shadow_steps} steps each)...")

        # Build combined dataset for shadow training
        all_records = members + nonmembers
        all_ds      = TextDataset(all_records, tokenizer, MAX_LENGTH)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none", task_type="CAUSAL_LM",
        )

        shadow_dp_m, shadow_dp_nm   = [], []
        shadow_nd_m, shadow_nd_nm   = [], []

        for i in range(args.n_shadows):
            seed_i = args.seed + i * 1000
            rng    = np.random.default_rng(seed_i)

            # Random 50% split of the full dataset
            idx    = rng.permutation(len(all_ds))
            half   = len(idx) // 2
            tr_idx = idx[:half]

            tr_subset = Subset(all_ds, tr_idx.tolist())
            tr_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True)

            print(f"    Shadow {i+1}/{args.n_shadows} — training on "
                  f"{len(tr_subset)} samples...")

            shadow = train_shadow_model(
                out / "shadow_models", lora_cfg, tr_subset,
                tokenizer, args.shadow_steps, seed=seed_i,
            )

            # Evaluate on member / non-member splits
            s_m  = compute_losses(shadow, dl_m,  pad_id, f"shadow {i+1} members")
            s_nm = compute_losses(shadow, dl_nm, pad_id, f"shadow {i+1} non-mem")
            shadow_dp_m.append(s_m)
            shadow_dp_nm.append(s_nm)
            shadow_nd_m.append(s_m)   # same shadow used for both models
            shadow_nd_nm.append(s_nm)

            del shadow; gc.collect(); torch.cuda.empty_cache()

        print("    Computing LiRA scores...")
        y, s = lira_attack(dp_m_loss, dp_nm_loss, shadow_dp_m, shadow_dp_nm)
        dp_attacks["LiRA"] = compute_metrics(y, s, "DP — LiRA")
        print_metrics(dp_attacks["LiRA"][0])

        # plot LiRA score distributions
        n_m  = len(dp_m_loss)
        all_scores = np.concatenate([s[:n_m], s[n_m:]])
        plot_lira_scores(s[:n_m], s[n_m:], "DP-QLoRA", out)

        y, s = lira_attack(nd_m_loss, nd_nm_loss, shadow_nd_m, shadow_nd_nm)
        nondp_attacks["LiRA"] = compute_metrics(y, s, "nonDP — LiRA")
        print_metrics(nondp_attacks["LiRA"][0])
        plot_lira_scores(s[:n_m], s[n_m:], "QLoRA-nonDP", out)

    # ── Save metrics JSON ─────────────────────────────────────────
    def save_json(attacks, fname):
        with open(out / fname, "w") as f:
            json.dump({k: v[0] for k,v in attacks.items()}, f, indent=2)

    save_json(dp_attacks,    "metrics_dp.json")
    save_json(nondp_attacks, "metrics_nondp.json")

    # ── Plots ─────────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    plot_roc_all(dp_attacks,    "DP-QLoRA",       out)
    plot_roc_all(nondp_attacks, "QLoRA-nonDP",    out)
    plot_attack_comparison(dp_attacks, nondp_attacks, out)
    plot_loss_distributions(dp_m_loss, dp_nm_loss, nd_m_loss, nd_nm_loss, out)

    # ── Final summary ─────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  FINAL SUMMARY — Strongest attack per model")
    print(f"{'═'*60}")
    for tag, attacks in [("DP-QLoRA (ε≈1)", dp_attacks),
                          ("QLoRA (non-private)", nondp_attacks)]:
        best_k   = max(attacks, key=lambda k: attacks[k][0]["mia_auc"])
        best_auc = attacks[best_k][0]["mia_auc"]
        print(f"  {tag:<22}  best AUC = {best_auc:.4f} "
              f"({best_k})  →  {attacks[best_k][0]['verdict']}")
    print(f"{'═'*60}")
    print(f"\nOutputs saved to: {out}")


if __name__ == "__main__":
    main()