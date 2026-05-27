"""
================================================================================
  PAYEE EXTRACTION — FINE-TUNING + EVALUATION SCRIPT  (Single-File, Standalone)
================================================================================

PURPOSE
-------
Fine-tune a causal language model (default: TinyLlama-1.1B-Chat) with LoRA
adapters to extract payee names from Indian bank transaction narrations, then
evaluate the fine-tuned model on a held-out validation set and save training
curves as image files.

HOW TO RUN
----------
    1. Create a virtual environment and activate it:
        python -m venv .venv
        .venv\Scripts\activate        (Windows)
        source .venv/bin/activate     (Linux / macOS)

    2. Install dependencies:
        pip install -r requirements.txt

    3. (Optional) For GPU-accelerated training with CUDA:
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        pip install bitsandbytes
       If you skip this, the script will fall back to CPU (much slower).

    4. (Windows only) Fix encoding before running:
        set PYTHONUTF8=1              (Command Prompt)
        $env:PYTHONUTF8 = "1"         (PowerShell)

    5. Run the script:
        python script.py

OUTPUTS
-------
After the script finishes you will find:
    outputs/
    ├── payee-lora/            # Saved LoRA adapter weights + tokenizer
    ├── eval/
    │   ├── metrics.json       # Aggregate evaluation metrics
    │   ├── predictions.jsonl  # Per-sample predictions vs. gold labels
    │   └── errors_top20.jsonl # 20 worst prediction errors for debugging
    └── plots/
        ├── training_loss.png       # Training loss curve
        ├── eval_loss.png           # Validation loss curve
        ├── learning_rate.png       # Learning rate schedule
        └── combined_curves.png     # All curves in a single figure
================================================================================
"""

import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


# =============================================================================
# CONFIGURATION — Change these variables to customise the run
# =============================================================================

# ── Model ────────────────────────────────────────────────────────────────────
# The Hugging Face model ID (or local path) for the base model.
# Change this to try a different model, e.g.:
#   "Qwen/Qwen2.5-1.5B-Instruct"
#   "HuggingFaceTB/SmolLM2-360M-Instruct"
#   "microsoft/phi-2"
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

# ── Data paths ───────────────────────────────────────────────────────────────
# Paths to the JSONL files produced by prepare_dataset.py.
# Each line should have "prompt" and "response" fields.
TRAIN_FILE = "data/train.jsonl"
VAL_FILE   = "data/val.jsonl"

# ── Output paths ─────────────────────────────────────────────────────────────
# Where the LoRA adapter, evaluation results, and plots will be saved.
OUTPUT_DIR   = "outputs/payee-lora"   # LoRA weights + tokenizer
EVAL_DIR     = "outputs/eval"         # Evaluation metrics & predictions
PLOTS_DIR    = "outputs/plots"        # Training curve images

# ── Training hyper-parameters ────────────────────────────────────────────────
NUM_EPOCHS           = 1        # Number of full passes through the training data
LEARNING_RATE        = 2e-4     # Peak learning rate for the AdamW optimiser
BATCH_SIZE           = 2        # Per-device training (and eval) batch size
GRADIENT_ACCUM_STEPS = 8        # Accumulate gradients over this many steps
MAX_SEQ_LENGTH       = 512      # Maximum token length for input sequences
LOGGING_STEPS        = 20       # Log training metrics every N steps
EVAL_STEPS           = 100      # Run validation every N steps
SAVE_STEPS           = 100      # Checkpoint the model every N steps
SAVE_TOTAL_LIMIT     = 2        # Keep only the N most recent checkpoints

# ── LoRA hyper-parameters ────────────────────────────────────────────────────
LORA_R       = 16       # Rank of the low-rank matrices
LORA_ALPHA   = 32       # Scaling factor (alpha / r)
LORA_DROPOUT = 0.05     # Dropout applied to LoRA layers
# Which linear layers inside the transformer to adapt:
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Evaluation settings ──────────────────────────────────────────────────────
MAX_NEW_TOKENS = 32     # Max tokens the model may generate per prediction
MAX_EVAL_SAMPLES = 0    # 0 = evaluate ALL validation samples; >0 = subset

# =============================================================================
# END OF CONFIGURATION — You should not need to edit below this line
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def supports_bf16() -> bool:
    """Check if the current GPU supports bfloat16 (Ampere or newer)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8


def build_prompt(narration: str) -> str:
    """
    Build the instruction prompt that the model sees at both training and
    inference time.  Keeping this consistent is critical for good results.
    """
    return (
        "You are an information extraction model. Extract only the payee name "
        "from the transaction narration. Return only the payee text, with no "
        "extra words.\n\n"
        f"Transaction narration:\n{narration}\n\n"
        "Payee:"
    )


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse whitespace, remove punctuation."""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`.,;:!?()\[\]{}]", "", text)
    return text


def jaccard_token_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    a_set = set(normalize_text(a).split())
    b_set = set(normalize_text(b).split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def char_similarity(a: str, b: str) -> float:
    """Character-level similarity using SequenceMatcher (similar to Levenshtein ratio)."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training():
    """
    Fine-tune the base model with LoRA adapters.

    Returns
    -------
    trainer : SFTTrainer
        The trainer object (contains training logs for plotting curves).
    output_dir : Path
        Where the adapter was saved.
    """
    print("=" * 70)
    print("  STEP 1 / 3 — FINE-TUNING")
    print("=" * 70)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load the dataset ─────────────────────────────────────────────────
    # The JSONL files should have "prompt" and "response" columns.
    # The SFTTrainer from trl will handle formatting internally.
    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

    # Rename "response" → "completion" if needed (trl expects "completion").
    if (
        "response" in dataset["train"].column_names
        and "completion" not in dataset["train"].column_names
    ):
        dataset = dataset.map(lambda x: {"completion": x["response"]})

    print(f"  Model     : {MODEL_NAME}")
    print(f"  Train size: {len(dataset['train']):,}")
    print(f"  Val size  : {len(dataset['validation']):,}")
    print()

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model loading (with optional 4-bit quantisation on GPU) ──────────
    use_cuda = torch.cuda.is_available()
    model_kwargs = {"device_map": "auto"}

    if use_cuda:
        # 4-bit quantisation via bitsandbytes for memory efficiency.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["quantization_config"] = quant_config
    else:
        # CPU fallback — no quantisation, use float32.
        model_kwargs["torch_dtype"] = torch.float32

    # Try loading with quantisation first; if it fails (version mismatch
    # between torch / transformers / bitsandbytes), fall back gracefully.
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    except (AttributeError, RuntimeError, Exception) as e:
        print(f"  ⚠  Quantised loading failed: {e}")
        print("  ⚠  Falling back to non-quantised loading (FP16/FP32) ...")
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

    model.config.use_cache = False  # Required for gradient checkpointing

    # ── LoRA configuration ───────────────────────────────────────────────
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    # ── Training arguments ───────────────────────────────────────────────
    # We build the kwargs dict manually so we can handle the
    # evaluation_strategy → eval_strategy rename across transformers versions.
    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=supports_bf16(),
        fp16=use_cuda and not supports_bf16(),
        report_to="none",
    )

    # Handle the evaluation_strategy / eval_strategy rename across
    # different versions of the transformers library.
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        training_kwargs["evaluation_strategy"] = "steps"
    else:
        training_kwargs["eval_strategy"] = "steps"

    training_args = TrainingArguments(**training_kwargs)

    # ── Build the trainer and start fine-tuning ──────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # ── Save the adapter + tokenizer ─────────────────────────────────────
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  ✓ LoRA adapter saved to: {output_dir}\n")

    return trainer, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalRow:
    """Holds one evaluation prediction alongside its metrics."""
    id: str
    narration: str
    gold: str            # Ground-truth payee
    pred: str            # Model prediction
    exact_match: int     # 1 if pred == gold (case-sensitive)
    normalized_exact_match: int   # 1 if normalised pred == normalised gold
    char_similarity: float        # Character-level similarity (0–1)
    token_jaccard: float          # Word-level Jaccard similarity (0–1)


def predict_payee(model, tokenizer, narration: str) -> str:
    """
    Generate a payee prediction for a single transaction narration.
    Uses greedy decoding (temperature=0) for deterministic results.
    """
    prompt = build_prompt(narration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # The prediction is everything after the prompt.
    return full_text[len(prompt):].strip()


def compute_summary(rows: list[EvalRow]) -> dict:
    """Aggregate per-sample metrics into a summary dict."""
    total = len(rows)
    if total == 0:
        return {
            "samples": 0,
            "exact_match": 0.0,
            "normalized_exact_match": 0.0,
            "avg_char_similarity": 0.0,
            "avg_token_jaccard": 0.0,
        }
    return {
        "samples": total,
        "exact_match": sum(r.exact_match for r in rows) / total,
        "normalized_exact_match": sum(r.normalized_exact_match for r in rows) / total,
        "avg_char_similarity": mean(r.char_similarity for r in rows),
        "avg_token_jaccard": mean(r.token_jaccard for r in rows),
    }


def run_evaluation(lora_path: Path):
    """
    Load the fine-tuned model and evaluate it on the validation set.

    Saves:
        - metrics.json          — aggregate scores
        - predictions.jsonl     — every sample with pred vs. gold
        - errors_top20.jsonl    — the 20 worst mistakes for review
    """
    print("=" * 70)
    print("  STEP 2 / 3 — EVALUATION")
    print("=" * 70)

    # ── Load validation data ─────────────────────────────────────────────
    val_rows = load_jsonl(Path(VAL_FILE))
    if MAX_EVAL_SAMPLES > 0:
        val_rows = val_rows[:MAX_EVAL_SAMPLES]

    print(f"  Evaluating {len(val_rows)} samples ...\n")

    # ── Load base model + LoRA adapter ───────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()

    # ── Run inference on each validation sample ──────────────────────────
    eval_rows: list[EvalRow] = []

    for row in tqdm(val_rows, desc="  Evaluating"):
        narration = row.get("prompt", "")

        # Strip the instruction wrapper if present (the prompt field from
        # prepare_dataset.py includes the full instruction + narration).
        if "Transaction narration:\n" in narration and "\n\nPayee:" in narration:
            narration = (
                narration
                .split("Transaction narration:\n", 1)[1]
                .split("\n\nPayee:", 1)[0]
            )

        gold = str(row.get("response", "")).strip()
        pred = predict_payee(model, tokenizer, narration)

        em  = int(pred == gold)
        nem = int(normalize_text(pred) == normalize_text(gold))

        eval_rows.append(
            EvalRow(
                id=str(row.get("id", "")),
                narration=narration,
                gold=gold,
                pred=pred,
                exact_match=em,
                normalized_exact_match=nem,
                char_similarity=char_similarity(pred, gold),
                token_jaccard=jaccard_token_similarity(pred, gold),
            )
        )

    # ── Compute aggregate metrics ────────────────────────────────────────
    metrics = compute_summary(eval_rows)

    # ── Save results ─────────────────────────────────────────────────────
    eval_dir = Path(EVAL_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / "metrics.json"
    preds_path   = eval_dir / "predictions.jsonl"
    errors_path  = eval_dir / "errors_top20.jsonl"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with preds_path.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # Save the 20 worst errors (lowest char_similarity among mismatches).
    errors = [r for r in eval_rows if r.normalized_exact_match == 0]
    errors.sort(key=lambda x: x.char_similarity)
    with errors_path.open("w", encoding="utf-8") as f:
        for r in errors[:20]:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print("\n  ── Evaluation Results ──")
    print(json.dumps(metrics, indent=2))
    print(f"\n  Metrics     : {metrics_path}")
    print(f"  Predictions : {preds_path}")
    print(f"  Worst errors: {errors_path}")
    print()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Training Curves / Plots
# ─────────────────────────────────────────────────────────────────────────────

def save_training_curves(trainer):
    """
    Extract the training logs from the trainer and save several plots:
        1. Training loss vs. step
        2. Validation (eval) loss vs. step
        3. Learning rate vs. step
        4. A combined figure with all three sub-plots
    """
    print("=" * 70)
    print("  STEP 3 / 3 — SAVING TRAINING CURVES")
    print("=" * 70)

    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract logged values from the trainer ───────────────────────────
    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses   = [], []
    lr_steps, lr_values       = [], []

    for entry in log_history:
        step = entry.get("step", None)
        if step is None:
            continue

        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])

        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_losses.append(entry["eval_loss"])

        if "learning_rate" in entry:
            lr_steps.append(step)
            lr_values.append(entry["learning_rate"])

    # ── Helper to save individual plots ──────────────────────────────────
    def _save_plot(x, y, xlabel, ylabel, title, filename, color="#4A90D9"):
        if not x:
            print(f"  ⚠  No data for '{title}' — skipping.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, color=color, linewidth=1.5)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved: {path}")

    _save_plot(
        train_steps, train_losses,
        "Step", "Loss",
        "Training Loss", "training_loss.png",
        color="#E74C3C",
    )
    _save_plot(
        eval_steps, eval_losses,
        "Step", "Loss",
        "Validation Loss", "eval_loss.png",
        color="#2ECC71",
    )
    _save_plot(
        lr_steps, lr_values,
        "Step", "Learning Rate",
        "Learning Rate Schedule", "learning_rate.png",
        color="#9B59B6",
    )

    # ── Combined figure (3 sub-plots in one image) ───────────────────────
    has_data = any([train_steps, eval_steps, lr_steps])
    if has_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Sub-plot 1: Training loss
        if train_steps:
            axes[0].plot(train_steps, train_losses, color="#E74C3C", linewidth=1.5)
            axes[0].set_title("Training Loss", fontweight="bold")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[0].set_title("Training Loss")

        # Sub-plot 2: Validation loss
        if eval_steps:
            axes[1].plot(eval_steps, eval_losses, color="#2ECC71", linewidth=1.5)
            axes[1].set_title("Validation Loss", fontweight="bold")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[1].set_title("Validation Loss")

        # Sub-plot 3: Learning rate
        if lr_steps:
            axes[2].plot(lr_steps, lr_values, color="#9B59B6", linewidth=1.5)
            axes[2].set_title("Learning Rate Schedule", fontweight="bold")
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel("Learning Rate")
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[2].set_title("Learning Rate")

        fig.suptitle(
            f"Training Curves — {MODEL_NAME}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        combined_path = plots_dir / "combined_curves.png"
        fig.savefig(str(combined_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {combined_path}")
    else:
        print("  ⚠  No training log data found — no combined plot generated.")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Orchestrates the full pipeline:
        1. Fine-tune the model
        2. Evaluate on the validation set
        3. Save training curves
    """
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  PAYEE EXTRACTION — Fine-Tune + Evaluate + Plot                    ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # ── GPU check ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU detected : {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  GPU detected : None — training will run on CPU (this will be slow)")
    print()

    # Step 1 — Fine-tune
    trainer, lora_path = run_training()

    # Step 2 — Evaluate the fine-tuned model
    metrics = run_evaluation(lora_path)

    # Step 3 — Save training curves to disk
    save_training_curves(trainer)

    # ── Final summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("  ALL DONE!")
    print("=" * 70)
    print(f"  Model adapter : {OUTPUT_DIR}")
    print(f"  Eval metrics  : {EVAL_DIR}/metrics.json")
    print(f"  Training plots: {PLOTS_DIR}/")
    print()
    print("  Key metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:30s} : {v:.4f}")
        else:
            print(f"    {k:30s} : {v}")
    print()


if __name__ == "__main__":
    main()
