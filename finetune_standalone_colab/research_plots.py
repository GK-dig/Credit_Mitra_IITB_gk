"""
Research-oriented plots for payee LoRA fine-tuning (training + evaluation).
Saves each figure as PNG and EPS; displays inline via plt.show().

NOTE: The Colab notebook embeds this module inline. After editing here, run:
  python build_notebook.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
except ImportError:
    roc_curve = precision_recall_curve = auc = None

# Default style for publication-friendly figures
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def save_and_show(fig, plots_dir: Path, stem: str, show: bool = True) -> dict[str, str]:
    """Save figure as PNG + EPS; optionally display."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "eps"):
        path = plots_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", format=ext)
        paths[ext] = str(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return paths


def _extract_log_series(log_history: list[dict]) -> dict[str, list]:
    """Parse Hugging Face Trainer log_history into plottable series."""
    out = {
        "train_steps": [],
        "train_loss": [],
        "eval_steps": [],
        "eval_loss": [],
        "lr_steps": [],
        "lr": [],
        "grad_steps": [],
        "grad_norm": [],
        "epoch_steps": [],
        "epoch": [],
    }
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry and "eval_loss" not in entry:
            out["train_steps"].append(step)
            out["train_loss"].append(entry["loss"])
        if "eval_loss" in entry:
            out["eval_steps"].append(step)
            out["eval_loss"].append(entry["eval_loss"])
        if "learning_rate" in entry:
            out["lr_steps"].append(step)
            out["lr"].append(entry["learning_rate"])
        if "grad_norm" in entry:
            out["grad_steps"].append(step)
            out["grad_norm"].append(entry["grad_norm"])
        if "epoch" in entry:
            out["epoch_steps"].append(step)
            out["epoch"].append(entry["epoch"])
    return out


def plot_training_phase(
    log_history: list[dict],
    plots_dir: Path,
    model_name: str = "",
    eval_accuracy_history: list[dict] | None = None,
    show: bool = True,
) -> list[str]:
    """
    Plots from Trainer log_history (and optional mid-training accuracy snapshots).
    Returns list of saved file stems.
    """
    plots_dir = Path(plots_dir)
    series = _extract_log_series(log_history)
    saved: list[str] = []

    def line_plot(x, y, title, ylabel, stem, color, log_y=False):
        if not x:
            print(f"Skip (no data): {title}")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color=color, linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if log_y and min(y) > 0:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, stem, show=show)
        saved.append(stem)

    line_plot(
        series["train_steps"],
        series["train_loss"],
        f"Training loss — {model_name}".strip(" —"),
        "Cross-entropy loss",
        "01_training_loss",
        "#E74C3C",
    )
    line_plot(
        series["eval_steps"],
        series["eval_loss"],
        f"Validation loss — {model_name}".strip(" —"),
        "Eval cross-entropy loss",
        "02_eval_loss",
        "#2ECC71",
    )
    line_plot(
        series["lr_steps"],
        series["lr"],
        "Learning rate schedule",
        "Learning rate",
        "03_learning_rate",
        "#9B59B6",
    )

    if series["grad_steps"]:
        line_plot(
            series["grad_steps"],
            series["grad_norm"],
            "Gradient norm",
            "grad_norm",
            "04_gradient_norm",
            "#3498DB",
        )

    # Train vs eval loss on shared axis
    if series["train_steps"] and series["eval_steps"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            series["train_steps"],
            series["train_loss"],
            label="Train loss",
            color="#E74C3C",
            linewidth=1.5,
        )
        ax.plot(
            series["eval_steps"],
            series["eval_loss"],
            label="Eval loss",
            color="#2ECC71",
            linewidth=1.5,
            marker="s",
            markersize=4,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs validation loss", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, "05_train_vs_eval_loss", show=show)
        saved.append("05_train_vs_eval_loss")

    # Smoothed train loss (moving average)
    if len(series["train_loss"]) >= 5:
        w = min(9, len(series["train_loss"]) // 2 * 2 + 1)
        kernel = np.ones(w) / w
        smooth = np.convolve(series["train_loss"], kernel, mode="valid")
        xs = series["train_steps"][w // 2 : w // 2 + len(smooth)]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(series["train_steps"], series["train_loss"], alpha=0.35, label="Raw", color="#E74C3C")
        ax.plot(xs, smooth, label=f"MA({w})", color="#C0392B", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Smoothed training loss", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, "06_smoothed_train_loss", show=show)
        saved.append("06_smoothed_train_loss")

    # Mid-training accuracy from callback
    if eval_accuracy_history:
        steps = [h["step"] for h in eval_accuracy_history]
        acc = [h["accuracy"] for h in eval_accuracy_history]
        nem = [h.get("normalized_accuracy", h["accuracy"]) for h in eval_accuracy_history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, acc, marker="o", label="Exact match", color="#2980B9")
        ax.plot(steps, nem, marker="s", label="Normalized exact match", color="#16A085")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title("Validation accuracy during fine-tuning", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, "07_midtraining_accuracy", show=show)
        saved.append("07_midtraining_accuracy")

    # Combined dashboard
    n_panels = 3 + (1 if eval_accuracy_history else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    idx = 0
    if series["train_steps"]:
        axes[idx].plot(series["train_steps"], series["train_loss"], color="#E74C3C")
        axes[idx].set_title("Train loss", fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
    idx += 1
    if series["eval_steps"]:
        axes[idx].plot(series["eval_steps"], series["eval_loss"], color="#2ECC71")
        axes[idx].set_title("Eval loss", fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
    idx += 1
    if series["lr_steps"]:
        axes[idx].plot(series["lr_steps"], series["lr"], color="#9B59B6")
        axes[idx].set_title("Learning rate", fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
    idx += 1
    if eval_accuracy_history and idx < len(axes):
        axes[idx].plot(steps, nem, color="#16A085", marker="o")
        axes[idx].set_ylim(0, 1.05)
        axes[idx].set_title("Val accuracy", fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
    fig.suptitle(f"Training dashboard — {model_name}".strip(" —"), fontweight="bold", y=1.02)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "08_training_dashboard", show=show)
    saved.append("08_training_dashboard")

    return saved


def _eval_arrays(eval_rows: list) -> dict[str, np.ndarray]:
    """Convert EvalRow-like dicts/objects to numpy arrays."""
    def g(row, key):
        if isinstance(row, dict):
            return row[key]
        return getattr(row, key)

    return {
        "exact": np.array([g(r, "exact_match") for r in eval_rows], dtype=float),
        "nem": np.array([g(r, "normalized_exact_match") for r in eval_rows], dtype=float),
        "char_sim": np.array([g(r, "char_similarity") for r in eval_rows], dtype=float),
        "jaccard": np.array([g(r, "token_jaccard") for r in eval_rows], dtype=float),
        "mse_char": np.array([g(r, "mse_char") for r in eval_rows], dtype=float),
        "mse_jaccard": np.array([g(r, "mse_jaccard") for r in eval_rows], dtype=float),
        "gold_len": np.array([len(g(r, "gold") or "") for r in eval_rows]),
        "pred_len": np.array([len(g(r, "pred") or "") for r in eval_rows]),
        "types": [g(r, "txn_type") if g(r, "txn_type") else "unknown" for r in eval_rows],
    }


def plot_evaluation_phase(
    eval_rows: list,
    metrics: dict,
    plots_dir: Path,
    model_name: str = "",
    show: bool = True,
) -> list[str]:
    """Research plots from full validation predictions."""
    if roc_curve is None:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    plots_dir = Path(plots_dir)
    arr = _eval_arrays(eval_rows)
    saved: list[str] = []
    n = len(eval_rows)

    # --- Accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Exact match", "Normalized\nexact match", "Mean char\nsimilarity", "Mean token\nJaccard"]
    vals = [
        metrics.get("exact_match", arr["exact"].mean()),
        metrics.get("normalized_exact_match", arr["nem"].mean()),
        metrics.get("avg_char_similarity", arr["char_sim"].mean()),
        metrics.get("avg_token_jaccard", arr["jaccard"].mean()),
    ]
    colors = ["#3498DB", "#2ECC71", "#9B59B6", "#E67E22"]
    bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Validation metrics (n={n})", fontweight="bold")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "10_accuracy_metrics_bar", show=show)
    saved.append("10_accuracy_metrics_bar")

    # --- MSE summary ---
    fig, ax = plt.subplots(figsize=(7, 4))
    mse_labels = ["MSE (1 − char sim)", "MSE (1 − Jaccard)", "RMSE char", "RMSE Jaccard"]
    mse_vals = [
        metrics.get("mse_char", arr["mse_char"].mean()),
        metrics.get("mse_jaccard", arr["mse_jaccard"].mean()),
        metrics.get("rmse_char", np.sqrt(arr["mse_char"].mean())),
        metrics.get("rmse_jaccard", np.sqrt(arr["mse_jaccard"].mean())),
    ]
    ax.bar(mse_labels, mse_vals, color=["#E74C3C", "#C0392B", "#F39C12", "#D35400"])
    ax.set_ylabel("Error")
    ax.set_title("MSE / RMSE on similarity (research proxy)", fontweight="bold")
    fig.tight_layout()
    save_and_show(fig, plots_dir, "11_mse_metrics_bar", show=show)
    saved.append("11_mse_metrics_bar")

    # --- ROC (binary correct vs similarity score) ---
    y_true = arr["nem"]
    for score, label, stem in [
        (arr["char_sim"], "Char similarity", "12_roc_char_similarity"),
        (arr["jaccard"], "Token Jaccard", "13_roc_token_jaccard"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, color="#2980B9", linewidth=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {label} as score", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, stem, show=show)
        saved.append(stem)

    # --- Precision-Recall ---
    for score, label, stem in [
        (arr["char_sim"], "Char similarity", "14_pr_char_similarity"),
        (arr["jaccard"], "Token Jaccard", "15_pr_token_jaccard"),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, score)
        pr_auc = auc(recall, precision)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, color="#8E44AD", linewidth=2, label=f"AUC = {pr_auc:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision–Recall — {label}", fontweight="bold")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_and_show(fig, plots_dir, stem, show=show)
        saved.append(stem)

    # --- Similarity histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(arr["char_sim"], bins=30, color="#3498DB", edgecolor="white", alpha=0.85)
    axes[0].set_title("Char similarity distribution")
    axes[0].set_xlabel("Similarity")
    axes[1].hist(arr["jaccard"], bins=30, color="#2ECC71", edgecolor="white", alpha=0.85)
    axes[1].set_title("Token Jaccard distribution")
    axes[1].set_xlabel("Jaccard")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Prediction similarity to gold", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "16_similarity_histograms", show=show)
    saved.append("16_similarity_histograms")

    # --- MSE per-sample histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(arr["mse_char"], bins=30, color="#E74C3C", edgecolor="white", alpha=0.85)
    axes[0].set_title("Per-sample MSE (char)")
    axes[1].hist(arr["mse_jaccard"], bins=30, color="#C0392B", edgecolor="white", alpha=0.85)
    axes[1].set_title("Per-sample MSE (Jaccard)")
    fig.tight_layout()
    save_and_show(fig, plots_dir, "17_mse_histograms", show=show)
    saved.append("17_mse_histograms")

    # --- Cumulative accuracy vs threshold ---
    thresholds = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(8, 4))
    for score, label, c in [
        (arr["char_sim"], "Char sim ≥ t", "#2980B9"),
        (arr["jaccard"], "Jaccard ≥ t", "#27AE60"),
    ]:
        acc_at_t = [(score >= t).mean() for t in thresholds]
        ax.plot(thresholds, acc_at_t, label=label, color=c, linewidth=1.5)
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("Fraction of samples ≥ threshold")
    ax.set_title("Threshold vs coverage (soft accuracy)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "18_threshold_coverage", show=show)
    saved.append("18_threshold_coverage")

    # Strict accuracy vs threshold (pred counted correct if sim >= t AND nem)
    fig, ax = plt.subplots(figsize=(8, 4))
    for score, label, c in [
        (arr["char_sim"], "Char sim", "#8E44AD"),
        (arr["jaccard"], "Jaccard", "#16A085"),
    ]:
        strict = [((score >= t) & (arr["nem"] == 1)).mean() for t in thresholds]
        ax.plot(thresholds, strict, label=f"Correct if score≥t ({label})", linewidth=1.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Strict match rate")
    ax.set_title("Threshold-based decision accuracy", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "19_threshold_strict_accuracy", show=show)
    saved.append("19_threshold_strict_accuracy")

    # --- Scatter char vs jaccard ---
    fig, ax = plt.subplots(figsize=(7, 6))
    correct = arr["nem"] == 1
    ax.scatter(
        arr["char_sim"][~correct],
        arr["jaccard"][~correct],
        alpha=0.5,
        s=18,
        c="#E74C3C",
        label="Incorrect",
    )
    ax.scatter(
        arr["char_sim"][correct],
        arr["jaccard"][correct],
        alpha=0.5,
        s=18,
        c="#2ECC71",
        label="Correct",
    )
    ax.set_xlabel("Char similarity")
    ax.set_ylabel("Token Jaccard")
    ax.set_title("Similarity scatter (correct vs incorrect)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "20_similarity_scatter", show=show)
    saved.append("20_similarity_scatter")

    # --- Length scatter ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(arr["gold_len"], arr["pred_len"], c=arr["char_sim"], cmap="viridis", alpha=0.6, s=20)
    lim = max(arr["gold_len"].max(), arr["pred_len"].max(), 1)
    ax.plot([0, lim], [0, lim], "--", color="gray")
    ax.set_xlabel("Gold payee length (chars)")
    ax.set_ylabel("Predicted length (chars)")
    ax.set_title("Length comparison (color = char similarity)", fontweight="bold")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Char sim")
    fig.tight_layout()
    save_and_show(fig, plots_dir, "21_length_scatter", show=show)
    saved.append("21_length_scatter")

    # --- Calibration (binned) ---
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    bin_centers, bin_acc = [], []
    for i in range(n_bins):
        mask = (arr["char_sim"] >= bins[i]) & (arr["char_sim"] < bins[i + 1])
        if i == n_bins - 1:
            mask = (arr["char_sim"] >= bins[i]) & (arr["char_sim"] <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[i] + bins[i + 1]) / 2)
        bin_acc.append(arr["nem"][mask].mean())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(bin_centers, bin_acc, "o-", color="#2980B9", label="Observed")
    ax.set_xlabel("Mean predicted score (char sim bin)")
    ax.set_ylabel("Fraction correct (NEM)")
    ax.set_title("Calibration curve (char similarity)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "22_calibration_char", show=show)
    saved.append("22_calibration_char")

    # --- By transaction type ---
    types = arr["types"]
    unique_types = sorted(set(types))
    if len(unique_types) > 1 and len(unique_types) <= 25:
        type_acc = []
        type_n = []
        for t in unique_types:
            m = np.array([ti == t for ti in types])
            type_acc.append(arr["nem"][m].mean())
            type_n.append(m.sum())
        fig, ax = plt.subplots(figsize=(max(8, len(unique_types) * 0.5), 4))
        xpos = np.arange(len(unique_types))
        ax.bar(xpos, type_acc, color="#3498DB", edgecolor="black", linewidth=0.4)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(unique_types, type_n)], rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Normalized exact match")
        ax.set_title("Accuracy by transaction type", fontweight="bold")
        fig.tight_layout()
        save_and_show(fig, plots_dir, "23_accuracy_by_type", show=show)
        saved.append("23_accuracy_by_type")

        fig, ax = plt.subplots(figsize=(max(8, len(unique_types) * 0.5), 4))
        data = [arr["char_sim"][np.array([ti == t for ti in types])] for t in unique_types]
        ax.boxplot(data, labels=unique_types)
        ax.set_ylabel("Char similarity")
        ax.set_title("Similarity by transaction type", fontweight="bold")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        save_and_show(fig, plots_dir, "24_boxplot_similarity_by_type", show=show)
        saved.append("24_boxplot_similarity_by_type")

    # --- Top payees confusion (subset) ---
    from collections import Counter

    def _field(row, key):
        return row[key] if isinstance(row, dict) else getattr(row, key)

    golds = [_field(r, "gold") for r in eval_rows]
    preds = [_field(r, "pred") for r in eval_rows]
    gold_counts = Counter(golds)
    top_golds = [g for g, _ in gold_counts.most_common(12)]
    if top_golds:
        idx_map = {g: i for i, g in enumerate(top_golds)}
        cm = np.zeros((len(top_golds), len(top_golds)))
        for g, p in zip(golds, preds):
            if g in idx_map and p in idx_map:
                cm[idx_map[g], idx_map[p]] += 1
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(top_golds)))
        ax.set_yticks(range(len(top_golds)))
        ax.set_xticklabels(top_golds, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(top_golds, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title("Confusion matrix (top 12 gold payees)", fontweight="bold")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        save_and_show(fig, plots_dir, "25_confusion_top_payees", show=show)
        saved.append("25_confusion_top_payees")

    # Save extended metrics JSON
    research = dict(metrics)
    fpr_c, tpr_c, _ = roc_curve(y_true, arr["char_sim"])
    fpr_j, tpr_j, _ = roc_curve(y_true, arr["jaccard"])
    research["roc_auc_char"] = float(auc(fpr_c, tpr_c))
    research["roc_auc_jaccard"] = float(auc(fpr_j, tpr_j))
    p, r, _ = precision_recall_curve(y_true, arr["char_sim"])
    research["pr_auc_char"] = float(auc(r, p))
    research["mse_char_std"] = float(pstdev(arr["mse_char"].tolist())) if n > 1 else 0.0
    out_path = plots_dir.parent.parent / "eval" / "research_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(research, f, indent=2, ensure_ascii=False)
    print("Saved research metrics:", out_path)

    return saved


# ═══════════════════════════════════════════════════════════════════════════
# Differential Privacy (DP-SGD) Plots
# ═══════════════════════════════════════════════════════════════════════════


def plot_dp_epsilon_convergence(
    dp_history: list[dict],
    plots_dir: Path,
    target_epsilon: float | None = None,
    show: bool = True,
) -> str:
    """Plot cumulative epsilon consumed over training steps.

    dp_history: list of dicts with keys 'step', 'epsilon', 'loss' (logged each step).
    """
    plots_dir = Path(plots_dir)
    steps = [h["step"] for h in dp_history]
    epsilons = [h["epsilon"] for h in dp_history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, epsilons, color="#E74C3C", linewidth=2, label="ε spent")
    if target_epsilon is not None and target_epsilon < float("inf"):
        ax.axhline(y=target_epsilon, color="#95A5A6", linestyle="--", linewidth=1.5,
                   label=f"Target ε = {target_epsilon}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cumulative ε (privacy budget spent)")
    ax.set_title("Privacy budget consumption during training", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "30_dp_epsilon_convergence", show=show)
    return "30_dp_epsilon_convergence"


def plot_dp_privacy_utility_tradeoff(
    experiment_results: list[dict],
    plots_dir: Path,
    show: bool = True,
) -> str:
    """Bar chart comparing accuracy metrics across different epsilon values.

    experiment_results: list of dicts, each with keys:
        'label', 'epsilon', 'exact_match', 'normalized_exact_match',
        'avg_char_similarity', 'avg_token_jaccard'
    """
    plots_dir = Path(plots_dir)
    n_exp = len(experiment_results)
    labels = [r["label"] for r in experiment_results]
    metric_names = ["Exact Match", "Normalized EM", "Char Similarity", "Token Jaccard"]
    metric_keys = ["exact_match", "normalized_exact_match", "avg_char_similarity", "avg_token_jaccard"]
    colors = ["#3498DB", "#2ECC71", "#9B59B6", "#E67E22"]

    x = np.arange(n_exp)
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(10, n_exp * 3), 5))

    for i, (mk, mn, c) in enumerate(zip(metric_keys, metric_names, colors)):
        vals = [r.get(mk, 0) for r in experiment_results]
        offset = (i - len(metric_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=mn, color=c, edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Privacy–Utility Tradeoff: Accuracy vs Privacy Budget (ε)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_and_show(fig, plots_dir, "31_dp_privacy_utility_tradeoff", show=show)
    return "31_dp_privacy_utility_tradeoff"


def plot_dp_vs_nondp_comparison(
    nondp_metrics: dict,
    dp_metrics: dict,
    plots_dir: Path,
    dp_epsilon: float = 8.0,
    show: bool = True,
) -> str:
    """Side-by-side comparison of DP vs non-DP metrics."""
    plots_dir = Path(plots_dir)
    metric_names = ["Exact Match", "Normalized EM", "Char Similarity", "Token Jaccard"]
    metric_keys = ["exact_match", "normalized_exact_match", "avg_char_similarity", "avg_token_jaccard"]

    nondp_vals = [nondp_metrics.get(k, 0) for k in metric_keys]
    dp_vals = [dp_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, nondp_vals, width, label="Non-DP (ε = ∞)",
                   color="#2ECC71", edgecolor="black", linewidth=0.4)
    bars2 = ax.bar(x + width / 2, dp_vals, width, label=f"DP-SGD (ε = {dp_epsilon})",
                   color="#E74C3C", edgecolor="black", linewidth=0.4)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("DP vs Non-DP: Payee Extraction Accuracy", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_and_show(fig, plots_dir, "32_dp_vs_nondp_comparison", show=show)
    return "32_dp_vs_nondp_comparison"


def plot_dp_gradient_norm_distribution(
    grad_norms: list[float],
    max_grad_norm: float,
    plots_dir: Path,
    show: bool = True,
) -> str:
    """Histogram of per-sample gradient norms, with clipping threshold line."""
    plots_dir = Path(plots_dir)
    fig, ax = plt.subplots(figsize=(8, 4))

    norms = np.array(grad_norms)
    ax.hist(norms, bins=50, color="#3498DB", edgecolor="white", alpha=0.85, density=True)
    ax.axvline(x=max_grad_norm, color="#E74C3C", linestyle="--", linewidth=2,
               label=f"Clipping threshold C = {max_grad_norm}")

    clipped_frac = (norms > max_grad_norm).mean() * 100
    ax.set_xlabel("Per-sample gradient norm")
    ax.set_ylabel("Density")
    ax.set_title(f"Gradient norm distribution ({clipped_frac:.1f}% clipped)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "33_dp_gradient_norm_distribution", show=show)
    return "33_dp_gradient_norm_distribution"


def plot_dp_training_dashboard(
    dp_history: list[dict],
    plots_dir: Path,
    target_epsilon: float | None = None,
    model_name: str = "",
    show: bool = True,
) -> str:
    """Combined DP training dashboard: loss, epsilon, and noise over steps."""
    plots_dir = Path(plots_dir)
    steps = [h["step"] for h in dp_history]
    losses = [h.get("loss", 0) for h in dp_history]
    epsilons = [h.get("epsilon", 0) for h in dp_history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Loss
    axes[0].plot(steps, losses, color="#E74C3C", linewidth=1.5)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("DP Training Loss", fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Epsilon
    axes[1].plot(steps, epsilons, color="#2980B9", linewidth=1.5)
    if target_epsilon is not None and target_epsilon < float("inf"):
        axes[1].axhline(y=target_epsilon, color="#95A5A6", linestyle="--",
                       label=f"Target ε = {target_epsilon}")
        axes[1].legend()
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("ε (cumulative)")
    axes[1].set_title("Privacy Budget Consumption", fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    title = f"DP-SGD Training Dashboard — {model_name}".strip(" —")
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "34_dp_training_dashboard", show=show)
    return "34_dp_training_dashboard"


def plot_dp_loss_comparison(
    nondp_log_history: list[dict],
    dp_history: list[dict],
    plots_dir: Path,
    dp_epsilon: float = 8.0,
    show: bool = True,
) -> str:
    """Overlay training loss curves for DP vs non-DP runs."""
    plots_dir = Path(plots_dir)
    fig, ax = plt.subplots(figsize=(9, 4))

    # Non-DP losses
    nondp_series = _extract_log_series(nondp_log_history)
    if nondp_series["train_steps"]:
        ax.plot(nondp_series["train_steps"], nondp_series["train_loss"],
                color="#2ECC71", linewidth=1.5, alpha=0.8, label="Non-DP (ε = ∞)")

    # DP losses
    dp_steps = [h["step"] for h in dp_history]
    dp_losses = [h.get("loss", 0) for h in dp_history]
    if dp_steps:
        ax.plot(dp_steps, dp_losses, color="#E74C3C", linewidth=1.5, alpha=0.8,
                label=f"DP-SGD (ε = {dp_epsilon})")

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: DP vs Non-DP", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_and_show(fig, plots_dir, "35_dp_loss_comparison", show=show)
    return "35_dp_loss_comparison"


def plot_all(
    log_history: list[dict] | None,
    eval_rows: list | None,
    metrics: dict | None,
    plots_training_dir: Path,
    plots_eval_dir: Path,
    model_name: str = "",
    eval_accuracy_history: list[dict] | None = None,
    show: bool = True,
) -> dict[str, list[str]]:
    """Run training and/or evaluation plot suites."""
    result = {"training": [], "evaluation": []}
    if log_history:
        print("\n=== Training-phase plots ===")
        result["training"] = plot_training_phase(
            log_history,
            plots_training_dir,
            model_name=model_name,
            eval_accuracy_history=eval_accuracy_history,
            show=show,
        )
    if eval_rows and metrics:
        print("\n=== Evaluation-phase plots ===")
        result["evaluation"] = plot_evaluation_phase(
            eval_rows,
            metrics,
            plots_eval_dir,
            model_name=model_name,
            show=show,
        )
    return result
