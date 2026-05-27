# Payee extraction LoRA fine-tuning — Colab notebook

This folder contains a **step-by-step Jupyter notebook** equivalent to the `finetune_standalone` pipeline (LoRA fine-tuning, evaluation, plots).

## Contents

| File | Purpose |
|------|---------|
| `Payee_Extraction_LoRA_Finetune.ipynb` | **Self-contained** notebook (all code + theory cells; upload only this to Colab) |
| `research_plots.py` | Optional source mirror; regenerate notebook via `python build_notebook.py` |
| `payee_eval_callback.py` | Optional source mirror for callback class |
| `build_notebook.py` | Rebuilds the `.ipynb` from sources |
| `requirements-colab.txt` | Pip packages for local or Colab `pip install -r` |

## How to open in Google Colab with a GPU

### 1. Upload the notebook to Colab

- Go to [Google Colab](https://colab.research.google.com/).
- **File → Upload notebook** and choose `Payee_Extraction_LoRA_Finetune.ipynb` from this folder.

Alternatively: push this repo to GitHub and use **File → Open notebook → GitHub** and select the `.ipynb` file.

### 2. Enable a GPU runtime

- **Runtime → Change runtime type**.
- Set **Hardware accelerator** to **T4 GPU** (free tier) or **A100 / L4** (Colab Pro if available).
- Click **Save**.

### 3. (Recommended) Mount Google Drive for data and outputs

If your `train.jsonl` / `val.jsonl` live on Drive (or you copy `finetune_standalone/data` there):

- Run the notebook cell that mounts Drive (`google.colab.drive.mount`).
- Set `BASE_DIR` in the notebook to your Drive path, e.g.  
  `/content/drive/MyDrive/Credit_Mitra/finetune_standalone`.

### 4. Put data where the notebook expects it

The notebook expects JSONL with **`prompt`** and **`response`** (same as `finetune_standalone/data/`).

- Either upload `finetune_standalone/data/train.jsonl` and `val.jsonl` into Colab’s `/content/...`,  
- Or copy that `data` folder into Drive and point `BASE_DIR` at it.

### 5. Hugging Face (if the model is gated)

If you use a gated model, create a token at [Hugging Face Settings](https://huggingface.co/settings/tokens) and in a Colab cell run:

```python
from huggingface_hub import login
login()  # paste token when prompted
```

Or set the `HF_TOKEN` secret in Colab (**Secrets** in the left sidebar) and read it in the notebook.

### 6. Run all cells

- **Runtime → Run all** (or run section by section top to bottom).

### 7. Research plots (automatic)

Plots are generated at two points:

1. **After training (Section 6)** — loss, eval loss, learning rate, train vs eval overlay, smoothed loss, mid-training accuracy, dashboard → `outputs/plots/training/` (`.png` + `.eps`).
2. **After evaluation (Section 7)** — accuracy bars, MSE/RMSE, ROC & PR curves, similarity/MSE histograms, threshold curves, calibration, type breakdown, confusion matrix → `outputs/plots/evaluation/`.

Set `SHOW_PLOTS = True` in the config cell to display each figure inline. Section 8 can re-plot without retraining.

**Colab:** upload **only** `Payee_Extraction_LoRA_Finetune.ipynb` — no extra Python files required.

### 8. Download artifacts (optional)

After training, download from Colab’s file browser or from Drive:

- `outputs/payee-lora/` — LoRA adapter
- `outputs/eval/metrics.json`, `research_metrics.json`, `predictions.jsonl`
- `outputs/plots/training/` and `outputs/plots/evaluation/` — PNG and EPS figures

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| **Out of memory** | Lower `BATCH_SIZE`, increase `GRADIENT_ACCUM_STEPS`, use a smaller `MODEL_NAME`, or enable 4-bit (already in notebook for CUDA). |
| **bitsandbytes fails on Colab** | Colab often works with the default `pip install bitsandbytes`; if not, the notebook falls back to non-quantized loading like `script.py`. |
| **Slow on CPU** | You must use a **GPU** runtime for reasonable training time. |

## Local use (optional)

```bash
cd finetune_standalone_colab
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements-colab.txt
jupyter notebook Payee_Extraction_LoRA_Finetune.ipynb
```

Point `BASE_DIR` in the notebook to your local `finetune_standalone` directory if you keep data there.
