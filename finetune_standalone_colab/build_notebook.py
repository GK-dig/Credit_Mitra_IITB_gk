"""Generate Payee_Extraction_LoRA_Finetune.ipynb with inline code and theory cells."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "Payee_Extraction_LoRA_Finetune.ipynb"


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in s.strip().split("\n")]}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [ln + "\n" for ln in s.strip().split("\n")],
        "execution_count": None,
        "outputs": [],
    }


def _strip_docstring(lines: list[str]) -> list[str]:
    if not lines or not lines[0].strip().startswith('"""'):
        return lines
    if lines[0].count('"""') >= 2:
        return lines[1:]
    for i in range(1, len(lines)):
        if '"""' in lines[i]:
            return lines[i + 1 :]
    return lines


def _from_first_def(lines: list[str]) -> str:
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            return "\n".join(lines[i:])
    return "\n".join(lines)


def plotting_cell_source() -> str:
    plots_lines = _strip_docstring((ROOT / "research_plots.py").read_text(encoding="utf-8").splitlines())
    cb_lines = _strip_docstring((ROOT / "payee_eval_callback.py").read_text(encoding="utf-8").splitlines())
    plots_body = _from_first_def(plots_lines)
    cb_body = _from_first_def(cb_lines)

    return f"""
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

from sklearn.metrics import auc, precision_recall_curve, roc_curve
from transformers import TrainerCallback
import torch
import random

plt.rcParams.update({{
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
}})

{cb_body}

{plots_body}

print("Inline plotting library ready.")
"""


TRAIN_CODE = '''
import os
import inspect
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ["ACCELERATE_MIXED_PRECISION"] = "no"

assert TRAIN_FILE.exists(), f"Missing: {TRAIN_FILE}"
assert VAL_FILE.exists(), f"Missing: {VAL_FILE}"

dataset = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
)
if "response" in dataset["train"].column_names and "completion" not in dataset["train"].column_names:
    dataset = dataset.map(lambda x: {"completion": x["response"]})

print("Train:", len(dataset["train"]), "| Val:", len(dataset["validation"]))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_cuda = torch.cuda.is_available()
print("CUDA:", use_cuda, torch.cuda.get_device_name(0) if use_cuda else "CPU")

use_quant = False
model_kwargs = {"device_map": "auto"}
if use_cuda:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    use_quant = True
else:
    model_kwargs["torch_dtype"] = torch.float32

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
except Exception as e:
    print("Quantized load failed, fallback:", e)
    use_quant = False
    model_kwargs.pop("quantization_config", None)
    model_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

model.config.use_cache = False
if use_quant:
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
)

training_kwargs = dict(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    bf16=False,
    fp16=False,
    optim="adamw_torch",
    report_to="none",
)
sig = inspect.signature(TrainingArguments.__init__)
training_kwargs["evaluation_strategy" if "evaluation_strategy" in sig.parameters else "eval_strategy"] = "steps"
training_args = TrainingArguments(**training_kwargs)

val_rows_for_callback = load_jsonl(VAL_FILE)
callbacks = []
payee_callback = None
if CALLBACK_EVAL_SAMPLES > 0:
    payee_callback = PayeeAccuracyCallback(tokenizer, val_rows_for_callback, max_samples=CALLBACK_EVAL_SAMPLES)
    callbacks.append(payee_callback)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
    callbacks=callbacks,
)

if use_cuda:
    torch.cuda.empty_cache()
print("Starting training...")
trainer.train()
trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("Saved LoRA to", OUTPUT_DIR)
'''


EVAL_CODE = '''
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

val_rows = load_jsonl(VAL_FILE)
if MAX_EVAL_SAMPLES > 0:
    val_rows = val_rows[:MAX_EVAL_SAMPLES]
print("Evaluating", len(val_rows), "samples...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, str(OUTPUT_DIR))
model.eval()


def predict_payee(narration: str) -> str:
    prompt = build_prompt(narration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()


eval_rows = []
for row in tqdm(val_rows, desc="Evaluating"):
    narration = extract_narration_from_prompt(str(row.get("prompt", "")))
    gold = str(row.get("response", "")).strip()
    pred = predict_payee(narration)
    cs = char_similarity(pred, gold)
    tj = jaccard_token_similarity(pred, gold)
    eval_rows.append(EvalRow(
        id=str(row.get("id", "")),
        narration=narration,
        gold=gold,
        pred=pred,
        exact_match=int(pred == gold),
        normalized_exact_match=int(normalize_text(pred) == normalize_text(gold)),
        char_similarity=cs,
        token_jaccard=tj,
        mse_char=(1.0 - cs) ** 2,
        mse_jaccard=(1.0 - tj) ** 2,
        txn_type=str(row.get("type", "unknown")),
    ))

n = len(eval_rows)
metrics = {
    "samples": n,
    "exact_match": sum(r.exact_match for r in eval_rows) / max(1, n),
    "normalized_exact_match": sum(r.normalized_exact_match for r in eval_rows) / max(1, n),
    "avg_char_similarity": mean(r.char_similarity for r in eval_rows) if n else 0.0,
    "avg_token_jaccard": mean(r.token_jaccard for r in eval_rows) if n else 0.0,
    "mse_char": mean(r.mse_char for r in eval_rows) if n else 0.0,
    "mse_jaccard": mean(r.mse_jaccard for r in eval_rows) if n else 0.0,
    "rmse_char": (mean(r.mse_char for r in eval_rows) ** 0.5) if n else 0.0,
    "rmse_jaccard": (mean(r.mse_jaccard for r in eval_rows) ** 0.5) if n else 0.0,
}

EVAL_DIR.mkdir(parents=True, exist_ok=True)
with open(EVAL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
with open(EVAL_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\\n")
errors = sorted([r for r in eval_rows if r.normalized_exact_match == 0], key=lambda x: x.char_similarity)
with open(EVAL_DIR / "errors_top20.jsonl", "w", encoding="utf-8") as f:
    for r in errors[:20]:
        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\\n")

print(json.dumps(metrics, indent=2))
'''


DP_TRAIN_CODE = '''
import os
import torch
import warnings
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from tqdm.auto import tqdm

# We must tokenize manually for the custom loop
def tokenize_for_dp(examples, tokenizer, max_length=128):
    prompts = examples["prompt"]
    completions = examples["response"] if "response" in examples else examples["completion"]
    full_texts = [p + c for p, c in zip(prompts, completions)]
    
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    # Causal LM: labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    # Ignore padding tokens in loss computation
    tokenized["labels"][tokenized["input_ids"] == tokenizer.pad_token_id] = -100
    
    return tokenized

print("Loading dataset for DP training...")
dataset = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare training data
train_dataset = dataset["train"].map(
    lambda x: tokenize_for_dp(x, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names
)
train_dataset.set_format(type="torch")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Loading base model {MODEL_NAME} in 4-bit...")
use_cuda = torch.cuda.is_available()

# 4-bit Quantization (QLoRA) config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config if use_cuda else None,
    torch_dtype=torch.float16 if use_cuda else torch.float32,
    device_map="auto",
)
model.config.use_cache = False
if use_cuda:
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

print("Applying LoRA adapters...")
peft_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, peft_config)

# IMPORTANT: Ensure ONLY LoRA parameters require gradients
for name, param in model.named_parameters():
    if "lora" not in name.lower():
        param.requires_grad = False
    else:
        # Opacus requires parameters requiring gradients to be float32 or float16, not int8/4
        # Since LoRA params are added by PEFT, they are already standard floats.
        pass

# Fix any module incompatibilities (e.g. LayerNorm -> GroupNorm for DP)
# We only do this strictly for parameters that require gradients.
errors = ModuleValidator.validate(model, strict=False)
if errors:
    print(f"Opacus ModuleValidator found {len(errors)} issues. Applying fixes...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ModuleValidator.fix(model)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

print(f"Attaching Opacus PrivacyEngine (Target ε = {TARGET_EPSILON})...")
privacy_engine = PrivacyEngine()
target_delta = TARGET_DELTA or (1.0 / len(train_dataset))

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=NUM_EPOCHS,
    target_epsilon=TARGET_EPSILON,
    target_delta=target_delta,
    max_grad_norm=MAX_GRAD_NORM_DP,
)

print(f"Starting DP-SGD Training... (Memory-safe physical batch size: {MAX_PHYSICAL_BATCH_SIZE})")
model.train()
dp_log_history = []
global_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    with BatchMemoryManager(
        data_loader=train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=optimizer
    ) as memory_safe_loader:
        
        pbar = tqdm(memory_safe_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, batch in enumerate(pbar):
            # Batch memory manager handles the virtual batching loop
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            # Opacus updates the gradients and takes optimization steps according to logical batches.
            # We track loss continuously.
            epoch_loss += loss.item()
            
            if optimizer.step_was_taken:
                global_step += 1
                current_epsilon = privacy_engine.get_epsilon(target_delta)
                
                # Log metrics
                if global_step % LOGGING_STEPS == 0:
                    dp_log_history.append({
                        "step": global_step,
                        "epoch": epoch + (i / len(memory_safe_loader)),
                        "loss": loss.item(),
                        "epsilon": current_epsilon,
                        "grad_norm": model.grad_sample_norm() if hasattr(model, 'grad_sample_norm') else 0,
                    })
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}", "ε": f"{current_epsilon:.2f}"})

    current_epsilon = privacy_engine.get_epsilon(target_delta)
    print(f"End of Epoch {epoch+1} — Loss: {epoch_loss/len(memory_safe_loader):.4f} — ε: {current_epsilon:.2f} (δ: {target_delta})")

# Save the DP-trained adapter
print(f"Saving DP-trained LoRA adapter to {DP_OUTPUT_DIR}...")
# Remove the opacus wrapper from the model to save correctly using PEFT
unwrapped_model = model._module
unwrapped_model.save_pretrained(str(DP_OUTPUT_DIR))
tokenizer.save_pretrained(str(DP_OUTPUT_DIR))
print("DP Training Complete!")
'''


DP_EVAL_CODE = '''
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

val_rows = load_jsonl(VAL_FILE)
if MAX_EVAL_SAMPLES > 0:
    val_rows = val_rows[:MAX_EVAL_SAMPLES]
print("Evaluating DP Model on", len(val_rows), "samples...")

tokenizer = AutoTokenizer.from_pretrained(str(DP_OUTPUT_DIR))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, str(DP_OUTPUT_DIR))
model.eval()

def predict_payee_dp(narration: str) -> str:
    prompt = build_prompt(narration)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()

dp_eval_rows = []
for row in tqdm(val_rows, desc="Evaluating DP"):
    narration = extract_narration_from_prompt(str(row.get("prompt", "")))
    gold = str(row.get("response", "")).strip()
    pred = predict_payee_dp(narration)
    cs = char_similarity(pred, gold)
    tj = jaccard_token_similarity(pred, gold)
    dp_eval_rows.append(EvalRow(
        id=str(row.get("id", "")),
        narration=narration,
        gold=gold,
        pred=pred,
        exact_match=int(pred == gold),
        normalized_exact_match=int(normalize_text(pred) == normalize_text(gold)),
        char_similarity=cs,
        token_jaccard=tj,
        mse_char=(1.0 - cs) ** 2,
        mse_jaccard=(1.0 - tj) ** 2,
        txn_type=str(row.get("type", "unknown")),
    ))

n = len(dp_eval_rows)
dp_metrics = {
    "samples": n,
    "exact_match": sum(r.exact_match for r in dp_eval_rows) / max(1, n),
    "normalized_exact_match": sum(r.normalized_exact_match for r in dp_eval_rows) / max(1, n),
    "avg_char_similarity": mean(r.char_similarity for r in dp_eval_rows) if n else 0.0,
    "avg_token_jaccard": mean(r.token_jaccard for r in dp_eval_rows) if n else 0.0,
    "mse_char": mean(r.mse_char for r in dp_eval_rows) if n else 0.0,
    "mse_jaccard": mean(r.mse_jaccard for r in dp_eval_rows) if n else 0.0,
    "rmse_char": (mean(r.mse_char for r in dp_eval_rows) ** 0.5) if n else 0.0,
    "rmse_jaccard": (mean(r.mse_jaccard for r in dp_eval_rows) ** 0.5) if n else 0.0,
}

DP_EVAL_DIR.mkdir(parents=True, exist_ok=True)
with open(DP_EVAL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(dp_metrics, f, indent=2, ensure_ascii=False)
with open(DP_EVAL_DIR / "predictions.jsonl", "w", encoding="utf-8") as f:
    for r in dp_eval_rows:
        f.write(json.dumps(asdict(r), ensure_ascii=False) + "\\n")

print("DP Metrics:")
print(json.dumps(dp_metrics, indent=2))
'''


cells = [
    md("""
# Payee extraction — LoRA fine-tuning & research plots

End-to-end notebook: **train** LoRA on payee extraction, **evaluate**, and produce **research figures** (PNG + EPS). All code is **inside this notebook** — upload only this `.ipynb` to Colab.

| Section | What you get |
|---------|----------------|
| 6–7 | Metric definitions + inline plotting library |
| 8–9 | QLoRA training + **training curves** |
| 10–11 | Full validation run + **ROC, PR, MSE, calibration**, etc. |
| 12 | Re-plot without retraining |
| 15–17 | **DP-SLM research**: theory, DP-SGD training, DP evaluation |
| 18 | Multi-ε experiment suite |
"""),
    md("## 1. Install dependencies"),
    code("%pip install -q transformers datasets accelerate peft trl bitsandbytes scikit-learn tqdm matplotlib huggingface_hub opacus"),
    md("## 2. GPU check\n\n**Colab:** Runtime → Change runtime type → **GPU**."),
    code("""
import sys
import torch
print("Python:", sys.version)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"""),
    md("""
## 3. Configuration

| Variable | Role |
|----------|------|
| `BASE_DIR` | Root with `data/train.jsonl`, `data/val.jsonl` |
| `SHOW_PLOTS` | `plt.show()` after each saved figure |
| `CALLBACK_EVAL_SAMPLES` | Val samples for **accuracy during training** (0 = off) |
"""),
    code("""
from pathlib import Path

USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/Credit_Mitra_IITB/finetune_standalone")
else:
    BASE_DIR = Path("/content/finetune_standalone")

TRAIN_FILE = BASE_DIR / "data" / "train.jsonl"
VAL_FILE = BASE_DIR / "data" / "val.jsonl"
OUTPUT_DIR = BASE_DIR / "outputs" / "payee-lora"
EVAL_DIR = BASE_DIR / "outputs" / "eval"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
PLOTS_TRAINING_DIR = PLOTS_DIR / "training"
PLOTS_EVAL_DIR = PLOTS_DIR / "evaluation"

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUM_STEPS = 8
LOGGING_STEPS = 20
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_NEW_TOKENS = 32
MAX_EVAL_SAMPLES = 0
SHOW_PLOTS = True
CALLBACK_EVAL_SAMPLES = 48

# ── Differential Privacy (DP-SGD) ──
USE_DP = True                    # Toggle DP training on/off
TARGET_EPSILON = 8.0             # Privacy budget (lower = more private)
TARGET_DELTA = None              # Auto-set to 1/len(train) if None
MAX_GRAD_NORM_DP = 1.0           # Per-sample gradient clipping norm
MAX_PHYSICAL_BATCH_SIZE = 2      # For Opacus BatchMemoryManager
DP_OUTPUT_DIR = BASE_DIR / "outputs" / "payee-lora-dp"
DP_EVAL_DIR = BASE_DIR / "outputs" / "eval-dp"
PLOTS_DP_DIR = PLOTS_DIR / "dp"

for p in (OUTPUT_DIR, EVAL_DIR, PLOTS_DIR, PLOTS_TRAINING_DIR, PLOTS_EVAL_DIR,
          DP_OUTPUT_DIR, DP_EVAL_DIR, PLOTS_DP_DIR):
    p.mkdir(parents=True, exist_ok=True)
print("TRAIN exists:", TRAIN_FILE.exists(), "| VAL exists:", VAL_FILE.exists())
"""),
    md("## 4. Data format\n\nJSONL with `prompt` (instruction + narration + `Payee:`) and `response` (gold payee). Optional `type` (e.g. P2P) for breakdown plots."),
    md("## 5. Hugging Face login (optional)"),
    code("# from huggingface_hub import login\n# login()\npass"),
    md("""
## 6. Shared helpers — prompts & metrics

**Theory:** Generative extraction — the model completes text after `Payee:`.

- **Exact match (EM):** `pred == gold`
- **Normalized EM (NEM):** compare after lowercase + punctuation strip + whitespace normalize
- **Char similarity:** `SequenceMatcher` ratio ∈ [0,1] — soft score for ROC/PR
- **Token Jaccard:** word overlap / union
- **MSE proxy:** `(1 − similarity)²` averaged over samples (not embedding MSE; useful for error magnitude on [0,1])
"""),
    code("""
import json
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from statistics import mean


def build_prompt(narration: str) -> str:
    return (
        "You are an information extraction model. Extract only the payee name "
        "from the transaction narration. Return only the payee text, with no "
        "extra words.\\n\\n"
        f"Transaction narration:\\n{narration}\\n\\n"
        "Payee:"
    )


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\\s+", " ", text)
    text = re.sub(r"[\\"'`.,;:!?()\\[\\]{}]", "", text)
    return text


def jaccard_token_similarity(a: str, b: str) -> float:
    a_set, b_set = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def extract_narration_from_prompt(prompt_field: str) -> str:
    if "Transaction narration:\\n" in prompt_field and "\\n\\nPayee:" in prompt_field:
        return prompt_field.split("Transaction narration:\\n", 1)[1].split("\\n\\nPayee:", 1)[0]
    return prompt_field


@dataclass
class EvalRow:
    id: str
    narration: str
    gold: str
    pred: str
    exact_match: int
    normalized_exact_match: int
    char_similarity: float
    token_jaccard: float
    mse_char: float
    mse_jaccard: float
    txn_type: str


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

print("Helpers OK.")
"""),
    md("""
## 7. Plotting library (all code inline)

**Theory — training curves**

| File stem | Meaning |
|-----------|---------|
| `01_training_loss` | Cross-entropy on train batch (should decrease) |
| `02_eval_loss` | CE on validation — **detect overfitting** if train↓ but eval↑ |
| `03_learning_rate` | Scheduler / constant LR |
| `05_train_vs_eval_loss` | Overlay: gap suggests overfitting |
| `06_smoothed_train_loss` | Moving average reduces step noise |
| `07_midtraining_accuracy` | EM / NEM on a **small val subset** each eval step |
| `08_training_dashboard` | Summary panel |

**Theory — evaluation curves** (after full val inference)

| File stem | Meaning |
|-----------|---------|
| `10_accuracy_metrics_bar` | EM, NEM, mean similarities |
| `11_mse_metrics_bar` | MSE/RMSE on (1−sim)² |
| `12–13` ROC | NEM as label, similarity as **ranking score**; AUC ≈ separability |
| `14–15` PR | Precision–recall (imbalanced correct/incorrect) |
| `16–17` histograms | Distribution of similarity / per-sample MSE |
| `18–19` threshold curves | Operating point if you threshold similarity |
| `20` scatter | Char sim vs Jaccard, colored by correct/wrong |
| `21` length scatter | Over/under-generation vs gold length |
| `22` calibration | Does higher similarity bin → higher accuracy? |
| `23–24` by `type` | P2P vs other txn types |
| `25` confusion | Top-12 gold payees |
"""),
    code(plotting_cell_source()),
    md("""
## 8. Fine-tuning (QLoRA + SFTTrainer)

**Theory: LoRA & 4-bit Quantization (QLoRA)**
- **LoRA (Low-Rank Adaptation):** Instead of updating all parameters of the 135M model, we freeze the original weights and inject trainable rank decomposition matrices (adapters) into the attention layers (`q_proj`, `k_proj`, etc.). This reduces trainable parameters from ~135M to ~1M, slashing memory usage.
- **4-bit Quantization (NF4):** The frozen base model is loaded in 4-bit precision instead of 16-bit or 32-bit. This drastically reduces VRAM requirements, allowing the model to fit on a standard Colab T4 GPU.
- **Effective Batch Size:** Training uses gradient accumulation. True batch size = `BATCH_SIZE × GRADIENT_ACCUM_STEPS`.

**Run this cell once** — it saves the adapter to `outputs/payee-lora/`. Then run **Section 9** for training plots.
"""),
    code(TRAIN_CODE),
    md("""
## 9. Training curves — run after Section 8

**Theory: Cross-Entropy Loss vs Target Accuracy**
- `trainer.state.log_history` records scalars each `logging_steps` / `eval_steps`. 
- **Loss:** The metric tracked here is token-level **Cross-Entropy (CE) Loss**, measuring the model's confidence in predicting the exact next token. While CE loss correlates with our end goal (payee extraction), it is not perfectly equivalent to Exact Match (EM). A model might have high loss on the exact token (e.g. predicting "Reliance" instead of "Reliance Fresh") but still be highly useful.
- **Callback Accuracy:** If `CALLBACK_EVAL_SAMPLES > 0`, the training loop pauses periodically to run *generative inference* (greedy decoding) on a validation subset. This produces **accuracy vs step** curves that are independent of CE loss, giving a truer sense of real-world performance during training.
"""),
    code("""
eval_accuracy_history = payee_callback.history if payee_callback else None
training_stems = plot_training_phase(
    trainer.state.log_history,
    PLOTS_TRAINING_DIR,
    model_name=MODEL_NAME,
    eval_accuracy_history=eval_accuracy_history,
    show=SHOW_PLOTS,
)
print(f"Done: {len(training_stems)} training figures in {PLOTS_TRAINING_DIR}")
"""),
    md("""
## 10. Full validation evaluation

**Theory: Inference Strategy for Extraction**
- **Greedy Decoding:** We use `do_sample=False` and `max_new_tokens=32`. For information extraction tasks (unlike creative writing or chatting), we want the single most probable sequence of tokens. This ensures deterministic, reproducible results without hallucinations.
- **Evaluation Loop:** Each validation row generates one prediction. This prediction is compared to the gold standard payee via the strict (Exact Match) and soft (Jaccard, Char Sim) metrics defined in Section 6.

Outputs: `metrics.json`, `predictions.jsonl`, `errors_top20.jsonl`. Then run **Section 11** for evaluation plots.
"""),
    code(EVAL_CODE),
    md("""
## 11. Evaluation curves — run after Section 10

**Theory: Advanced Evaluation Metrics**

- **ROC (Receiver Operating Characteristic):** We treat Normalized Exact Match (NEM) as the binary ground truth, and our soft similarities (Char Sim / Jaccard) as the prediction "score". The ROC curve shows the tradeoff between True Positive Rate and False Positive Rate as we vary the similarity threshold. The AUC (Area Under Curve) indicates how well the soft score separates perfect matches from incorrect ones.
- **PR (Precision-Recall):** Similar to ROC, but more informative when class balance is highly skewed (e.g., if the model easily gets 90% of payees correct, PR highlights the difficulty of the remaining 10%).
- **MSE (Mean Squared Error) Bars:** We report mean `(1−sim)²`. Think of this as the squared error on a 0–1 similarity scale (0 = perfect match, 1 = completely different). This helps quantify the magnitude of the errors when the model is wrong.

All figures saved to `outputs/plots/evaluation/` as `.png` and `.eps`.
"""),
    code("""
evaluation_stems = plot_evaluation_phase(
    eval_rows,
    metrics,
    PLOTS_EVAL_DIR,
    model_name=MODEL_NAME,
    show=SHOW_PLOTS,
)
print(f"Done: {len(evaluation_stems)} evaluation figures in {PLOTS_EVAL_DIR}")
"""),
    md("""
## 12. Re-plot everything (no retrain)

Reload `predictions.jsonl` if needed; reuse `trainer.state.log_history` from Section 8.
"""),
    code("""
if "eval_rows" not in dir():
    eval_rows = []
    with open(EVAL_DIR / "predictions.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_rows.append(json.loads(line))
    with open(EVAL_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

hist = payee_callback.history if payee_callback else None
summary = plot_all(
    log_history=trainer.state.log_history,
    eval_rows=eval_rows,
    metrics=metrics,
    eval_accuracy_history=hist,
    show=SHOW_PLOTS,
)
summary
"""),
    md("""
## 15. Research: DP-SLM (Differentially Private Small Language Models)

**Why Differential Privacy for Financial NLP?**
Bank statements contain highly sensitive Personally Identifiable Information (PII) such as payee names, account numbers, and UPI IDs. Standard fine-tuning of Large Language Models (LLMs) poses a risk: models can memorize and later leak exact training data. **Differential Privacy (DP)** provides a mathematical guarantee against this memorization.

**What is DP-SGD?**
Differentially Private Stochastic Gradient Descent modifies the standard training loop by:
1. **Clipping:** Bounding the maximum gradient norm ($C$) for each individual training sample.
2. **Noise Injection:** Adding calibrated Gaussian noise (proportional to $C$) to the aggregated batch gradient.

**The DP-SLM Paradigm**
Applying DP-SGD directly to massive LLMs (like Llama-3 70B) is computationally prohibitive and destroys model utility due to the sheer volume of noise added (noise scales with the number of parameters).
The **DP-SLM paradigm** solves this by:
- Using a **Small Language Model** (e.g., Qwen2.5-1.5B or SmolLM2-135M).
- Using **Parameter-Efficient Fine-Tuning (PEFT/LoRA)** to train only ~1% of the weights.
- Because we train far fewer parameters, the required DP noise is drastically reduced, allowing the model to learn the task while remaining strictly private.
"""),
    md("""
## 16. DP-SGD Fine-tuning (Custom Loop)

**Theory:**
We cannot use HuggingFace `SFTTrainer` for DP-SGD because Opacus requires deep integration with the PyTorch `Optimizer` and `DataLoader` to compute per-sample gradients.
Furthermore, Opacus is incompatible with 4-bit quantized layers. The solution is to keep the base model frozen in 4-bit, and apply DP-SGD **only to the LoRA adapters** (which are standard FP16/32).

*This cell will train a private adapter and save it to `outputs/payee-lora-dp/`.*
"""),
    code("if USE_DP:\n" + "".join("    " + line + "\n" for line in DP_TRAIN_CODE.splitlines())),
    md("""
## 17. DP Evaluation & Comparison

Let's evaluate the DP model and compare it against the Non-DP baseline.
"""),
    code("if USE_DP:\n" + "".join("    " + line + "\n" for line in DP_EVAL_CODE.splitlines())),
    code("""
if USE_DP:
    print("--- DP Training Dashboard ---")
    plot_dp_training_dashboard(dp_log_history, PLOTS_DP_DIR, target_epsilon=TARGET_EPSILON, model_name=MODEL_NAME, show=SHOW_PLOTS)
    
    print("--- DP vs Non-DP Comparison ---")
    plot_dp_vs_nondp_comparison(metrics, dp_metrics, PLOTS_DP_DIR, dp_epsilon=TARGET_EPSILON, show=SHOW_PLOTS)
    
    if len(dp_log_history) > 0 and 'grad_norm' in dp_log_history[0]:
        plot_dp_gradient_norm_distribution([h['grad_norm'] for h in dp_log_history], MAX_GRAD_NORM_DP, PLOTS_DP_DIR, show=SHOW_PLOTS)
"""),
    md("""
## 18. Multi-Epsilon Experiment Suite (Optional)

Run the DP training loop multiple times across different privacy budgets to plot the **Privacy-Utility Tradeoff**.
(Uncomment and run if you have time/compute).
"""),
    code("""
import copy

EXPERIMENTS = [
    {"epsilon": float("inf"), "label": "Non-DP (baseline)"},
    {"epsilon": 8.0, "label": "DP (ε=8, relaxed)"},
    {"epsilon": 3.0, "label": "DP (ε=3, moderate)"},
    {"epsilon": 1.0, "label": "DP (ε=1, strict)"},
]
results = []

# For the baseline, we already have the metrics from Section 10
if "metrics" in globals():
    baseline_result = dict(metrics)
    baseline_result["label"] = "Non-DP (baseline)"
    baseline_result["epsilon"] = float("inf")
    results.append(baseline_result)

# We will define a helper to run DP for a specific epsilon
def run_dp_experiment(epsilon, label):
    print(f"\\n{'='*50}\\nRunning DP Experiment: {label}\\n{'='*50}")
    
    # 1. Re-initialize model and optimizer
    exp_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map="auto",
    )
    exp_model.config.use_cache = False
    if use_cuda:
        exp_model = prepare_model_for_kbit_training(exp_model)
        exp_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
    exp_model = get_peft_model(exp_model, peft_config)
    
    for name, param in exp_model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
            
    _ = ModuleValidator.fix(exp_model)
    
    exp_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, exp_model.parameters()),
        lr=LEARNING_RATE
    )
    
    # 2. Attach PrivacyEngine
    exp_privacy_engine = PrivacyEngine()
    exp_target_delta = TARGET_DELTA or (1.0 / len(train_dataset))
    
    exp_model, exp_optimizer, exp_train_loader = exp_privacy_engine.make_private_with_epsilon(
        module=exp_model,
        optimizer=exp_optimizer,
        data_loader=train_loader,
        epochs=NUM_EPOCHS,
        target_epsilon=epsilon,
        target_delta=exp_target_delta,
        max_grad_norm=MAX_GRAD_NORM_DP,
    )
    
    # 3. Train
    exp_model.train()
    for epoch in range(NUM_EPOCHS):
        with BatchMemoryManager(
            data_loader=exp_train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=exp_optimizer
        ) as memory_safe_loader:
            for batch in tqdm(memory_safe_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (ε={epsilon})"):
                exp_optimizer.zero_grad()
                outputs = exp_model(
                    input_ids=batch["input_ids"].to(exp_model.device),
                    attention_mask=batch["attention_mask"].to(exp_model.device),
                    labels=batch["labels"].to(exp_model.device)
                )
                loss = outputs.loss
                loss.backward()
                exp_optimizer.step()
                
    # 4. Evaluate
    exp_model.eval()
    exp_eval_rows = []
    
    def exp_predict(narration: str) -> str:
        prompt = build_prompt(narration)
        inputs = tokenizer(prompt, return_tensors="pt").to(exp_model.device)
        with torch.no_grad():
            out = exp_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
        
    for row in tqdm(val_rows, desc=f"Evaluating {label}"):
        narration = extract_narration_from_prompt(str(row.get("prompt", "")))
        gold = str(row.get("response", "")).strip()
        pred = exp_predict(narration)
        cs = char_similarity(pred, gold)
        tj = jaccard_token_similarity(pred, gold)
        exp_eval_rows.append(EvalRow(
            id=str(row.get("id", "")),
            narration=narration,
            gold=gold,
            pred=pred,
            exact_match=int(pred == gold),
            normalized_exact_match=int(normalize_text(pred) == normalize_text(gold)),
            char_similarity=cs,
            token_jaccard=tj,
            mse_char=(1.0 - cs) ** 2,
            mse_jaccard=(1.0 - tj) ** 2,
            txn_type=str(row.get("type", "unknown")),
        ))
        
    n = len(exp_eval_rows)
    exp_metrics = {
        "epsilon": epsilon,
        "label": label,
        "exact_match": sum(r.exact_match for r in exp_eval_rows) / max(1, n),
        "normalized_exact_match": sum(r.normalized_exact_match for r in exp_eval_rows) / max(1, n),
        "avg_char_similarity": mean(r.char_similarity for r in exp_eval_rows) if n else 0.0,
        "avg_token_jaccard": mean(r.token_jaccard for r in exp_eval_rows) if n else 0.0,
    }
    
    return exp_metrics

# Run experiments for the DP configurations
if USE_DP:
    for exp in EXPERIMENTS:
        if exp["epsilon"] != float("inf"):
            # Avoid repeating the target epsilon if it was already run in Section 16
            if exp["epsilon"] == TARGET_EPSILON and "dp_metrics" in globals():
                res = dict(dp_metrics)
                res["label"] = exp["label"]
                res["epsilon"] = exp["epsilon"]
                results.append(res)
            else:
                res = run_dp_experiment(exp["epsilon"], exp["label"])
                results.append(res)
                
    print("\\nExperiment Results:")
    for r in results:
        print(f"{r['label']}: NEM={r['normalized_exact_match']:.3f}, CharSim={r['avg_char_similarity']:.3f}")
        
    plot_dp_privacy_utility_tradeoff(results, PLOTS_DP_DIR, show=SHOW_PLOTS)
"""),
    md("## 19. Single narration demo"),
    code("""
sample_narration = "UPI/DR/867530921456/Reliance Fresh/Groceries/REF-2345/ICICI Bank"
print("Predicted payee:", predict_payee(sample_narration))
"""),
    md("""
## 20. Next steps

- Increase `NUM_EPOCHS` if eval loss still drops.
- Try `Qwen/Qwen2.5-1.5B-Instruct` (more VRAM).
- Set `CALLBACK_EVAL_SAMPLES = 0` for faster training (fewer mid-run generations).
- Download `outputs/plots/training/`, `outputs/plots/evaluation/`, and `outputs/plots/dp/` for your thesis/report.
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print("Wrote", OUT, "with", len(cells), "cells")
