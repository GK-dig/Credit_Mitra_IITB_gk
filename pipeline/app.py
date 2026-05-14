"""
Credit Mitra – PDF → Structured Transactions Pipeline
=====================================================
Upload a bank-statement PDF.  Docling extracts tables,
the fine-tuned SmolLM2 LoRA adapter predicts payee names,
and you download the enriched JSON.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from dp import extract_with_dp, privatise_payee, budget

import torch
import uvicorn
from docling.document_converter import DocumentConverter
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── paths ────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
LORA_PATH = str(
    Path(__file__).resolve().parent.parent / "Fine-tuning" / "outputs" / "payee-lora-smollm2"
)

# ── model loading (once at startup) ─────────────────────────────────────
print("Loading base model …")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
print("Loading LoRA adapter …")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("Model ready ✓")

# ── inference helper ─────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are an information extraction model. Extract only the payee name "
    "from the transaction narration. Return only the payee text, with no extra words."
)


def build_prompt(narration: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Transaction narration:\n{narration}\n\n"
        f"Payee:"
    )


NOISE_TOKENS = ["UPI", "HDFC"]


def clean_narration(narration: str) -> str:
    """Remove noisy bank-specific tokens before feeding to the SLM."""
    cleaned = narration
    for token in NOISE_TOKENS:
        cleaned = cleaned.replace(token, "")
    # collapse multiple slashes / spaces left behind
    import re
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip("/ ")


def predict_payee(narration: str, max_new_tokens: int = 32) -> str:
    prompt = build_prompt(clean_narration(narration))
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


# ── Docling extraction ───────────────────────────────────────────────────
def extract_transactions(pdf_path: str) -> list[dict]:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    rows: list[dict] = []
    for table in result.document.tables:
        df = table.export_to_dataframe()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.fillna("")

        for _, row in df.iterrows():
            rec = {
                "date": str(row.get("date", "")).strip(),
                "particulars": str(row.get("particulars", "")).strip(),
                "deposits": str(row.get("deposits", "")).strip(),
                "withdrawals": str(row.get("withdrawals", "")).strip(),
                "balance": str(row.get("balance", "")).strip(),
            }
            # skip entirely empty rows
            if not any(rec.values()):
                continue
            rows.append(rec)
    return rows


# ── FastAPI ──────────────────────────────────────────────────────────────
app = FastAPI(title="Credit Mitra Pipeline")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/process-pdf")
async def process_pdf(pdf: UploadFile = File(...)):
    # save uploaded file
    suffix = Path(pdf.filename or "upload.pdf").suffix or ".pdf"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await pdf.read()
    tmp.write(content)
    tmp.close()

    try:
        # Step 1 — Docling extraction
        transactions = extract_transactions(tmp.name)
        transactions, dp_report1 = extract_with_dp(transactions, epsilon=2.0)


        # Step 2 — SLM payee prediction for each row with a narration
        for txn in transactions:
            narration = txn.get("particulars", "")
            if narration and narration not in ("Opening Balance", "Closing Balance"):
                txn["payee"] = predict_payee(narration)
                dp_result = privatise_payee(txn["payee"], epsilon=1.0)
                txn["payee_tokens_dp"] = dp_result["payee_tokens_dp"]
                txn["payee"] = " ".join(dp_result["payee_tokens_dp"]) 
            else:
                txn["payee"] = ""

        return JSONResponse(content={"status": "success", "transactions": transactions})
    finally:
        os.unlink(tmp.name)


# ── run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
