"""Payee accuracy callback — used by build_notebook.py to embed in the notebook."""

import random

import torch
from transformers import TrainerCallback


class PayeeAccuracyCallback(TrainerCallback):
    """On each evaluation, greedy-decode payees on a small val subset and log accuracy."""

    def __init__(self, tokenizer, val_rows, max_samples=48, seed=42):
        self.tokenizer = tokenizer
        self.history = []
        rng = random.Random(seed)
        pool = list(val_rows)
        rng.shuffle(pool)
        self._subset = pool[: min(max_samples, len(pool))]

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None or not self._subset:
            return control
        model.eval()
        exact = nem = n = 0
        for row in self._subset:
            prompt_field = str(row.get("prompt", ""))
            narration = extract_narration_from_prompt(prompt_field)
            gold = str(row.get("response", "")).strip()
            prompt = build_prompt(narration)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            pred = self.tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt) :].strip()
            exact += int(pred == gold)
            nem += int(normalize_text(pred) == normalize_text(gold))
            n += 1
        acc, nacc = exact / max(1, n), nem / max(1, n)
        self.history.append({
            "step": state.global_step,
            "accuracy": acc,
            "normalized_accuracy": nacc,
            "samples": n,
        })
        print(f"  [Callback] step={state.global_step} exact={acc:.4f} norm={nacc:.4f} n={n}")
        return control
