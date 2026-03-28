# Credit Mitra – Full System Documentation

## Overview

**Credit Mitra** is an end-to-end AI-powered pipeline for processing banking transaction narration strings. It takes raw PDF bank statements or CSV transaction files, extracts structured data, identifies merchants, extracts payee names, categorizes transactions, and stores the results in a database — all orchestrated via LangGraph and exposed through a FastAPI backend with a React frontend.

**Core Use Case**: Financial institutions and fintech apps need to transform messy, unstructured narration strings (like `"UPI/DR/123456789/ZOMATO/YESB/somecode"`) into clean, structured, categorized records.

---

## System Architecture

### Technology Stack

**Backend**
- Python 3.10+
- FastAPI + Uvicorn
- LangGraph (pipeline orchestration)
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

## Project Structure

```
Smart-Narration-Parser/
│
├── main.py                            # FastAPI entry point — all API routes
├── finalize.py                        # Core pipeline runner + DB storage + stats
├── requirements.txt
├── .gitignore
│
├── extraction_from_pdfs/              # PDF → CSV transaction extraction
├── payee_name_extraction/             # Extract payee names from narration strings
├── merchant_non_merchant_identification/  # Classify: is this a merchant transaction?
├── merchant_information_extraction/   # Extract structured merchant info
├── categorization_of_merchants/       # Category tagging (food, travel, utilities…)
├── langgraph_orchaestration/          # LangGraph DAG wiring all pipeline stages
├── finalization_and_storage_in_db/    # Final DB write logic
│
├── uploads/                           # Temp storage for uploaded PDFs
├── output/                            # Generated CSV files
│
└── client/                            # React + Vite frontend
```

---

## Processing Pipeline

The system processes each transaction narration through a sequential multi-stage pipeline, orchestrated by LangGraph:

```
PDF Upload
    │
    ▼
[1] PDF Extraction          →  Raw transaction rows extracted to CSV
    │
    ▼
[2] Payee Name Extraction    →  "ZOMATO", "SWIGGY", "HDFC BANK" identified
    │
    ▼
[3] Merchant Identification  →  Is this a merchant or a peer/bank transfer?
    │
    ▼
[4] Merchant Info Extraction →  Name, location, business type extracted
    │
    ▼
[5] Categorization           →  Food, Travel, Utilities, Shopping, etc.
    │
    ▼
[6] Finalization & DB Store  →  Structured record saved to database
    │
    ▼
[7] Statistics Output        →  Aggregated summary returned to frontend
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

**Request Body**: JSON array of transaction objects (selected by the user in the frontend).

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
        "category": "Food & Dining",
        "merchant_info": { ... }
      }
    }
  ]
}
```

**Notes**: Each transaction is run through `run_pipeline()`, saved to the database via `save_record_to_db()`, and the results are stored globally in `processed_output_global`.

---

### GET `/statistics`

**Purpose**: Get aggregated statistics on the processed transactions.

**Response**:
```json
{
  "status": "success",
  "statistics": {
    "total_transactions": 50,
    "merchant_count": 32,
    "non_merchant_count": 18,
    "categories": {
      "Food & Dining": 12,
      "Travel": 8,
      "Utilities": 5
    }
  }
}
```

**Error** (if no transactions processed yet):
```json
{
  "status": "error",
  "message": "No processed transactions yet."
}
```

---

### POST `/payee-llm`

**Purpose**: Extract payee name from a raw narration string using LLM only.

**Request Body**: Plain string (not JSON)

```
UPI/DR/123456789/ZOMATO/YESB/somecode
```

**Response**: Extracted payee name and metadata.

---

### POST `/payee-llm-langsearch`

**Purpose**: Extract payee name using LLM + web search for better accuracy on ambiguous narrations.

**Request Body**: Plain string

**Notes**: Uses LangGraph's web search integration to validate or enrich the extracted payee name.

---

### POST `/given-payee-llm`

**Purpose**: Given a known payee name, extract structured merchant information using LLM.

**Request Body**: Plain string (payee name)

```
Zomato
```

**Response**: Structured merchant info (category, business type, etc.)

---

### POST `/given-payee-llm-langsearch`

**Purpose**: Given a payee name, fetch enriched merchant data via LLM + web search.

**Request Body**: Plain string (payee name)

**Notes**: More accurate than `/given-payee-llm` but slower due to web search.

---

## Core Business Logic

### `finalize.py` — The Heart of the System

This file contains all core logic:

| Function | Purpose |
|---|---|
| `read_transactions_from_csv()` | Parse CSV into `Transaction` objects |
| `run_pipeline()` | Execute the full LangGraph pipeline on one transaction |
| `save_record_to_db()` | Persist a processed transaction to the database |
| `get_statistics()` | Compute aggregated stats from processed + CSV data |
| `api_1_payee_llm()` | LLM-only payee extraction |
| `api_2_payee_llm_langsearch()` | LLM + web search payee extraction |
| `api_3_given_payee_llm()` | LLM merchant info from known payee |
| `api_4_given_payee_llm_langsearch()` | LLM + web search merchant info |

### Transaction Data Model

```python
class Transaction:
    transaction: str   # Raw narration string
    amount: float      # Transaction amount
    date: str          # Transaction date
```

### Global State

The FastAPI app maintains two in-memory global stores:

```python
processed_output_global: List[Dict]      # Results from /process-selected
csv_transactions_global: List[Transaction]  # Transactions from /upload-csv
```

These allow the `/statistics` endpoint to cross-reference CSV data with pipeline output without re-reading files.

---

## LangGraph Orchestration

The pipeline is structured as a LangGraph DAG (Directed Acyclic Graph). Each node in the graph is a processing stage:

```
START
  │
  ├─► payee_extraction_node
  │         │
  ├─► merchant_classifier_node
  │         │
  │    ┌────┴────┐
  │  (merchant) (non-merchant)
  │    │              │
  ├─► info_extraction │
  │    │              │
  ├─► categorization  │
  │         │         │
  └─► finalization ◄──┘
          │
         END
```

The `langgraph_orchaestration/` module wires these nodes together and handles conditional routing (e.g., skipping merchant info extraction for non-merchant transactions).

---

## Frontend (React + Vite)

Located in the `client/` directory.

### Setup

```bash
cd client
npm install
npm run dev
# Runs at http://localhost:5173
```

### API Connection

```javascript
axios.create({ baseURL: "http://127.0.0.1:8000" })
```

### Screens / Features

Based on the dependencies, the frontend provides:

- **PDF Upload** — drag and drop or file picker → calls `/extract-pdf`
- **CSV Preview** — table showing extracted transactions → calls `/upload-csv`
- **Transaction Selection** — checkboxes to select which transactions to process
- **Processing** — sends selected transactions to `/process-selected`
- **Dashboard / Charts** — Recharts visualizations of categories and merchant breakdown → calls `/statistics`

---

## Configuration

### Environment Variables

```env
# Database connection
DB_URI=...

# LLM API Key (for payee extraction and merchant info)
OPENAI_API_KEY=... or GOOGLE_API_KEY=...

# Web search (if using langsearch variants)
SERPAPI_KEY=... or TAVILY_API_KEY=...
```

### CORS

The backend is configured with wide-open CORS for local development:

```python
allow_origins=["*"]
allow_methods=["*"]
allow_headers=["*"]
```

**Note**: Restrict `allow_origins` to specific domains before deploying to production.

---

## Local Development Setup

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start server with hot reload
uvicorn main:app --reload

# Swagger UI available at:
# http://127.0.0.1:8000/docs
```

### Frontend

```bash
cd client
npm install
npm run dev
# http://localhost:5173
```

---

## Production Deployment

### Backend
- **Server**: Uvicorn + Gunicorn (`gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app`)
- **Hosting**: AWS EC2, ECS/Fargate, or any Python-compatible PaaS
- **File Storage**: Move `uploads/` and `output/` to AWS S3
- **Database**: AWS RDS or MongoDB Atlas for structured transaction storage

### Frontend
- **Build**: `npm run build` → generates `dist/`
- **Hosting**: Vercel, Netlify, or AWS CloudFront + S3

---

## Known Issues & Recommended Improvements

### Current Limitations

- **In-memory global state** — `processed_output_global` and `csv_transactions_global` are reset on every server restart. For production, these should be persisted in a database or Redis cache.
- **No authentication** — any client can call any endpoint. Add API key or JWT auth for production.
- **Wide-open CORS** — `allow_origins=["*"]` is fine for development but must be restricted for production.
- **Temp file handling** — uploaded CSVs use `tempfile` but PDFs are saved to `uploads/` permanently. Add a cleanup job.
- **No input validation** — the `/process-selected` endpoint accepts arbitrary dicts; add Pydantic model validation.

### Recommended Improvements

1. Add Pydantic request models for all endpoints
2. Replace in-memory globals with a proper database session
3. Add authentication middleware (API keys or JWT)
4. Add background task support (`BackgroundTasks`) for long-running pipeline runs
5. Add a `/status/{job_id}` polling endpoint for async processing
6. Add structured logging (e.g., `loguru`)
7. Write unit tests for each pipeline stage independently
8. Add rate limiting to the LLM-based endpoints

---

## End-to-End Usage Example

```
1. Upload PDF
   POST /extract-pdf  →  returns transactions.csv

2. Load CSV into UI
   POST /upload-csv   →  returns transaction list for preview

3. User selects transactions in UI

4. Process selected
   POST /process-selected  →  runs pipeline, saves to DB

5. View results
   GET /statistics  →  returns category breakdown, merchant counts, etc.
```

Or for single transaction testing:

```
POST /payee-llm                  →  quick LLM-only payee extraction
POST /payee-llm-langsearch       →  LLM + web search for accuracy
POST /given-payee-llm            →  merchant info from known name
POST /given-payee-llm-langsearch →  enriched merchant info via search
```
