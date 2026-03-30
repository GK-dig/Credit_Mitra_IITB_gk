# Credit Mitra – Full System Documentation

## Overview

**Credit Mitra** is an end-to-end AI-powered pipeline for processing banking transaction narration strings. It takes raw PDF bank statements or CSV transaction files, extracts structured data, identifies merchants, extracts payee names, categorizes transactions, and stores the results in a database — all orchestrated via LangGraph and exposed through a FastAPI backend with a React frontend.

**Core Use Case**: Financial institutions and fintech apps need to transform messy, unstructured narration strings (like `"UPI/DR/123456789/ZOMATO/YESB/somecode"`) into clean, structured, categorized records.

**Project Plan**:
You can find the project plan [Here][].
<!-- Link Definitions (can be placed at the end of the file) -->
[Here]: https://docs.google.com/spreadsheets/d/1R4jnfN6i1RINwZK6Y1QYTtlt_zAcFxYfLw7zYKLjdpI/edit?usp=sharin

---

## System Architecture

### Technology Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- LangGraph (pipeline orchestration)
- PyMongo + MongoDB (persistence)
- Pydantic (data models)
- PDF processing libraries
- Modular extraction pipelines (each as its own module)

**Frontend**
- React 19 + Vite
- TailwindCSS
- Axios (HTTP client)
- React Router DOM
- Recharts (data visualization)
- Heroicons

---

### Frontend ↔ Backend Communication

Each row is a matched pair: the React screen on the left calls the FastAPI endpoint on the right.

```mermaid
flowchart LR
    subgraph FE["React Frontend — localhost:5173"]
        A1[PDF upload screen]
        A2[CSV preview table]
        A3[Process selected]
        A4[Dashboard charts]
        A5[Payee lookup tools]
    end

    subgraph BE["FastAPI Backend — 127.0.0.1:8000"]
        B1[POST /extract-pdf\nreturns CSV file]
        B2[POST /upload-csv\nreturns transaction list]
        B3[POST /process-selected\nruns pipeline + saves to DB]
        B4[GET /statistics\ncategory breakdown]
        B5["POST /payee-llm *\n4 LLM lookup variants"]
    end

    A1 -->|multipart PDF| B1
    B1 -->|FileResponse CSV| A1
    A2 -->|multipart CSV| B2
    B2 -->|transaction list JSON| A2
    A3 -->|JSON array of rows| B3
    B3 -->|pipeline output JSON| A3
    A4 -->|GET| B4
    B4 -->|stats JSON| A4
    A5 -->|plain string| B5
    B5 -->|payee + summary JSON| A5
```

---

### LangGraph Pipeline Internals

What happens inside `run_pipeline()` for every transaction string. The conditional branch at `merchant?` is the key design decision — merchant transactions run two extra nodes, non-merchants skip straight to finalization.

```mermaid
flowchart TD
    START([transaction string])
    START --> N1

    N1["payee name extraction
    LLM parses raw narration string
    → state.payee_name"]

    N1 --> N2

    N2["merchant identification
    binary classifier
    → state.is_merchant: true / false"]

    N2 --> DECISION{merchant?}

    DECISION -->|yes| N3
    DECISION -->|no| N5

    N3["merchant info extraction
    name, summary, websites
    → state.merchant_summary
    → state.merchant_websites"]

    N3 --> N4

    N4["categorization
    food, travel, utilities...
    → state.merchant_category"]

    N4 --> N5

    N5["finalization
    save_record_to_db()
    CSV fields + pipeline state → MongoDB"]

    N5 --> END([done])
```

**Pipeline state keys written by each node:**

| Key | Written by | Value |
|---|---|---|
| `payee_name` | payee extraction | string — e.g. `"Zomato"` |
| `is_merchant` | merchant identification | boolean |
| `merchant_summary` | merchant info extraction | free-text description |
| `merchant_websites` | merchant info extraction | list of URLs |
| `merchant_category` | categorization | label — e.g. `"Food & Dining"` |

---

### Data Flow (PDF → MongoDB)

How data physically moves through the system, including the intentional re-upload loop. The CSV is returned to the user as a download so they can inspect or edit rows before choosing which ones to process.

```mermaid
flowchart TD
    U([user browser])
    U -->|multipart POST /extract-pdf| S1

    S1["PDF saved to disk
    uploads/{filename}.pdf"]

    S1 --> S2

    S2["process_pdf()
    extraction_from_pdfs module
    raw rows extracted"]

    S2 --> S3

    S3["CSV written to disk
    output/transactions_{name}.csv"]

    S3 -->|FileResponse download| U

    U -->|user re-uploads CSV
    POST /upload-csv| S4

    S4["read_transactions_from_csv()
    CSV parsed row by row
    → List[Transaction]"]

    S4 --> GLOBAL1[(csv_transactions_global\nin-memory list)]

    S4 -->|user selects rows
    POST /process-selected| S5

    S5["run_pipeline per transaction
    LangGraph DAG"]

    S5 --> S6

    S6["save_record_to_db()
    merges CSV fields + pipeline state
    → MongoDB document"]

    S6 --> DB[(transactions_db.transactions\nMongoDB)]

    S5 --> GLOBAL2[(processed_output_global\nin-memory dict)]

    GLOBAL1 -->|cross-referenced| STATS
    GLOBAL2 -->|cross-referenced| STATS

    STATS["get_statistics()
    category totals, spent/earned
    merchant vs non-merchant counts"]

    STATS -->|GET /statistics response| U
```

**MongoDB document shape** (written by `save_record_to_db`):

```json
{
  "date": "2024-01-15",
  "amount": 450.0,
  "credit/debit": "DR",
  "balance": 12000.0,
  "reference_number": "...",
  "type": "Food & Dining",
  "transaction": "UPI/DR/.../ZOMATO/...",
  "payee_name": "Zomato",
  "is_merchant": true,
  "merchant_summary": "Indian food delivery platform...",
  "merchant_websites": ["zomato.com"]
}
```

---

### Diagram 4 — Deployment / Infrastructure

Local dev uses `localhost:27017` and local disk. Production swaps those for MongoDB Atlas and AWS S3 via environment variable updates only — no code changes needed.

```mermaid
flowchart TD
    USER([browser / user])

    USER -->|HTTPS static assets| CDN
    USER -->|REST API calls| LB

    subgraph FRONTEND["Frontend — Vercel / Netlify"]
        CDN[CloudFront CDN\nReact + Vite SPA]
    end

    subgraph BACKEND["Backend — AWS EC2 / ECS"]
        LB[Uvicorn + Gunicorn\nFastAPI — 4 workers]
    end

    subgraph AWS["AWS Infrastructure"]
        S3[S3 Bucket\nPDF + CSV file storage]
        MONGO[(MongoDB Atlas\ntransactions_db)]
        LANG[LangGraph runtime\nin-process orchestrator]
    end

    subgraph EXTERNAL["External APIs"]
        LLM[LLM API\nOpenAI / Gemini]
        SEARCH[Web Search API\nSerpAPI / Tavily]
    end

    LB -->|file upload and download| S3
    LB -->|insert and query| MONGO
    LB -->|orchestrates| LANG
    LANG -->|inference calls| LLM
    LANG -->|langsearch variants only| SEARCH
```

**Local dev vs production equivalents:**

| Component | Local dev | Production |
|---|---|---|
| Database | `mongodb://localhost:27017/` | MongoDB Atlas (`MONGO_URI`) |
| File storage | `uploads/` and `output/` on disk | AWS S3 bucket |
| Backend server | `uvicorn main:app --reload` | Gunicorn + Uvicorn workers |
| Frontend | `npm run dev` on port 5173 | Vercel / Netlify / CloudFront |

---

## Project Structure

```
Smart-Narration-Parser/
│
├── main.py                                # FastAPI entry point — all API routes
├── finalize.py                            # Core pipeline runner + DB storage + stats
├── requirements.txt
├── .gitignore
│
├── extraction_from_pdfs/                  # PDF → CSV transaction extraction
├── payee_name_extraction/                 # Extract payee names from narration strings
├── merchant_non_merchant_identification/  # Classify: is this a merchant transaction?
├── merchant_information_extraction/       # Extract structured merchant info
├── categorization_of_merchants/           # Category tagging (food, travel, utilities…)
├── langgraph_orchaestration/              # LangGraph DAG wiring all pipeline stages
├── finalization_and_storage_in_db/        # Final DB write logic
│
├── uploads/                               # Temp storage for uploaded PDFs
├── output/                                # Generated CSV files
│
└── client/                                # React + Vite frontend
```

---

## API Endpoints

Base URL: `http://127.0.0.1:8000`  
Swagger docs: `http://127.0.0.1:8000/docs`

---

### POST `/extract-pdf`

**Purpose**: Upload a PDF bank statement and extract transactions into a CSV file.

**Request**: `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `pdf` | File | PDF bank statement |

**Response**: Returns a downloadable `transactions.csv` file.

**Notes**:
- PDF is saved to `uploads/` directory
- Output CSV is saved to `output/transactions_<filename>.csv`
- Response is a `FileResponse` with `text/csv` content type

---

### POST `/upload-csv`

**Purpose**: Upload a CSV of transactions to parse and preview before processing.

**Request**: `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | CSV file of transactions |

**Response**:
```json
{
  "status": "success",
  "transactions": [
    {
      "transaction": "UPI/DR/123456789/ZOMATO/YESB/...",
      "amount": 450.0,
      "date": "2024-01-15"
    }
  ]
}
```

**Notes**: Transactions are also stored globally in `csv_transactions_global` for later statistics computation.

---

### POST `/process-selected`

**Purpose**: Run the full AI pipeline on a user-selected subset of transactions.

**Request Body**: JSON array of transaction objects.

```json
[
  { "transaction": "UPI/DR/123456789/ZOMATO/YESB/...", "amount": 450.0, "date": "2024-01-15" },
  { "transaction": "NEFT/CR/9876543210/HDFC/...", "amount": 10000.0, "date": "2024-01-16" }
]
```

**Response**:
```json
{
  "status": "success",
  "processed": 2,
  "details": [
    {
      "transaction": "UPI/DR/123456789/ZOMATO/YESB/...",
      "pipeline_output": {
        "payee_name": "Zomato",
        "is_merchant": true,
        "merchant_category": "Food & Dining",
        "merchant_summary": "...",
        "merchant_websites": ["zomato.com"]
      }
    }
  ]
}
```

---

### GET `/statistics`

**Purpose**: Aggregated statistics cross-referencing pipeline output with original CSV data.

**Response**:
```json
{
  "status": "success",
  "statistics": {
    "Food & Dining": {
      "total_transactions": 12,
      "merchant_transactions": 12,
      "non_merchant_transactions": 0,
      "total_spent": 5400.0,
      "total_earned": 0.0
    }
  }
}
```

**Error** (if no transactions processed yet):
```json
{ "status": "error", "message": "No processed transactions yet." }
```

---

### POST `/payee-llm`

LLM-only payee extraction. Fastest variant, no web search.

**Request**: plain string — `UPI/DR/123456789/ZOMATO/YESB/somecode`

**Response**: `{ "payee_name": "Zomato", "merchant_summary": "..." }`

---

### POST `/payee-llm-langsearch`

LLM extraction with web search fallback if summary is empty or unhelpful.

**Response**: `{ "payee_name": "...", "merchant_summary": "...", "merchant_websites": [...] }`

---

### POST `/given-payee-llm`

Given a known payee name, generate merchant info via LLM only.

**Request**: plain string — `Zomato`

---

### POST `/given-payee-llm-langsearch`

Given a known payee name, generate enriched merchant info via LLM + web search fallback.

> All four `/payee-*` variants share the same fallback logic — web search only fires when `generate_merchant_summary()` returns `None`, `""`, or `"No information found"`.

---

## Core Logic

### `finalize.py` — The Heart of the System

| Function | Purpose |
|---|---|
| `read_transactions_from_csv()` | Parse CSV into `Transaction` objects |
| `run_pipeline()` | Execute the full LangGraph pipeline on one transaction |
| `save_record_to_db()` | Persist a processed transaction to MongoDB |
| `get_statistics()` | Compute aggregated stats from processed + CSV data |
| `api_1_payee_llm()` | LLM-only payee extraction |
| `api_2_payee_llm_langsearch()` | LLM + web search payee extraction |
| `api_3_given_payee_llm()` | LLM merchant info from known payee |
| `api_4_given_payee_llm_langsearch()` | LLM + web search merchant info |

### Transaction Data Model (Pydantic)

```python
class Transaction(BaseModel):
    date: Optional[str]
    amount: Optional[float]
    type: Optional[str]           # "DR" or "CR"
    balance: Optional[float]
    reference_number: Optional[str]
    category: Optional[str]
    transaction: str              # Raw narration string — the key input field
```

### Global State

```python
processed_output_global: List[Dict]        # Results from /process-selected
csv_transactions_global: List[Transaction] # Transactions from /upload-csv
```

Both are cross-referenced by `get_statistics()`. **Both reset on server restart** — a known production limitation (see Known Issues).

---

## Local Development Setup

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

### Frontend

```bash
cd client
npm install
npm run dev
# http://localhost:5173
```

---
## End-to-End Usage Example

```
1. Upload PDF
   POST /extract-pdf  →  returns transactions.csv download

2. Review and re-upload CSV
   POST /upload-csv   →  returns parsed transaction list

3. Select rows in UI, submit
   POST /process-selected  →  runs LangGraph pipeline, saves to MongoDB

4. View results
   GET /statistics  →  category breakdown, merchant counts, spend totals
```

Or for single-transaction testing via Swagger at `/docs`:

```
POST /payee-llm                  →  fast LLM-only extraction
POST /payee-llm-langsearch       →  LLM + web search fallback
POST /given-payee-llm            →  merchant info from a known payee name
POST /given-payee-llm-langsearch →  enriched merchant info with search fallback
```
