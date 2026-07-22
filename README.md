# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![Celery](https://img.shields.io/badge/Celery-5.3.6%2B-37814A?logo=celery)](https://docs.celeryq.dev/)
[![Polars](https://img.shields.io/badge/Engine-Polars%20Rust-CD412B?logo=rust)](https://pola.rs/)
[![WeasyPrint](https://img.shields.io/badge/PDF_Engine-WeasyPrint%2061.2%2B-FF6600)](https://weasyprint.org/)

---

## Overview
GetReport is an intelligent data analysis platform that transforms raw CSV/Excel and multi-format datasets into comprehensive PDF reports with minimal effort. It combines **high-performance statistical analysis**, **hybrid-search RAG**, **cross-format deduplication**, and **security-first design** to deliver actionable business intelligence.

## Problem Solved
Data analysis traditionally requires:
- Technical expertise in statistics and programming
- Hours of manual exploration and cleaning
- Separate tools for visualization and reporting

**GetReport eliminates these barriers** by providing a single platform that:
1. Automatically detects data quality issues across single or multi-file uploads
2. Applies intelligent cleaning based on user approval via an interactive Issue Ledger
3. Performs deep statistical analysis **(optimized for speed with Polars)**
4. Generates publication-ready PDF reports

---

## Key Features

### High-Performance Analysis & Extended Ingestion
- **Modular Architecture**: Clean, maintainable `app/services/analysis/` package structure.
- **Extended Ingestion Support**: Native parsing for **11+ file formats** (`.csv`, `.tsv`, `.xls`, `.xlsx`, `.parquet`, `.json`, `.jsonl`, `.ndjson`, `.feather`, `.arrow`, `.gz`).
- **Multi-File Batch Uploads (`POST /upload/batch`)**: Ingest up to 10 datasets simultaneously grouped under a single session `batch_id`.
- **Polars Lazy Execution**: Single-pass computation for summary statistics (~10x faster).
- **Winsorization (Outlier Capping)**: Replaces outlier values beyond IQR thresholds with boundary limits to preserve dataset size without artificial variance/mean skew.
- **Time-Series Detection**: Automatic trend and seasonality analysis when date columns present.
- **Real-Time Job Updates**: WebSocket connection supporting Redis PubSub (or polling fallback) for real-time progress updates.

### Trust Foundation & Deduplication Strategy
- **Two-Tier Cross-Format Deduplication**:
  - **Tier 1 (SHA-256 Byte Hash)**: Computes streaming byte digests (`file_hash`) during upload to immediately catch binary duplicates.
  - **Tier 2 (Polars Semantic Fingerprinting)**: Computes normalized row-level fingerprints across column names, row counts, and data checksums to catch duplicate records even across different file formats (e.g. matching `.parquet` against `.csv`).
- **Column Confidence Scores**: Grades every column on Completeness, Consistency, Validity, and Stability.
- **Machine Learning Readiness Score**: Computes a comprehensive dataset readiness score (0-100%) evaluating column-level completeness, constant or near-zero variance features, format consistency, outliers, and class imbalance with explicit expert recommendations.
- **Interactive Issue Ledger**: Review data quality alerts, approve or reject automated cleaning actions, modify values, and track changes.
- **Decision Transparency**: Logs why specific tests (Correlation, Time-Series, Anova) were run or skipped.
- **Transformation DAG**: Tracks data transformations as a directed acyclic graph for complete auditability.

### Advanced Intelligence (RAG Engine)
- **Hybrid RAG Engine**: Combines **Dense Vector Search** with **Sparse Keyword Scoring** for precise context retrieval.
- **Table-Aware Text Splitting (`TableAwareTextSplitter`)**: Splits text by paragraphs and tabular section boundaries while prefixing schema headers to prevent row-boundary fragmentation.
- **Resilient Embedding Fallbacks (`TFIDFVectorStore`)**: If OpenAI embedding API keys are unconfigured or unavailable, the system automatically falls back to an in-memory TF-IDF + Cosine search engine, ensuring RAG chat is 100% operational.
- **Interactive RAG Chat**: Ask questions directly about the dataset and its generated analysis using the context-aware chat interface, equipped with early blank-query guards and input length limits.

### Audit History Dashboard & PDF Generation
- **Audit History Dashboard (`/dashboard`)**: Dedicated dashboard page to view past dataset audits, search records by filename/grade, and download PDF reports directly.
- **Dual-Engine PDF Generation**: Uses `WeasyPrint` (production HTML/CSS caching) or `ReportLab` (fast local dev without system dependencies).

### Security First Design
- **Magic Number Validation**: Strictly verifies file signatures (`.xlsx`, `.xls`, `.parquet`, `.feather`, `.gz`) to prevent extension spoofing.
- **Content Inspection**: Rejects binary files masquerading as text CSVs.
- **Input Sanitization**: Guards against prompt injection in RAG workflows (max query length limit).
- **Zip Bomb Mitigation**: Pre-validates compressed file parameters (ratio, file count, and decompressed XML size) to prevent OOM/DoS attacks during Excel file ingestion.

---

## Tech Stack

### Frontend
| Component | Technology |
|-----------|------------|
| Framework | React 18 + Vite |
| Language | TypeScript |
| Styling | Tailwind CSS, Vanilla CSS, Shadcn/UI |
| Navigation | React Router DOM v6 |
| Icons | Lucide React |
| State | TanStack Query |

### Backend
| Component | Technology |
|-----------|------------|
| Framework | FastAPI (Python 3.12+) |
| Task Queue | Celery + Redis |
| Data Processing | Polars (Rust Engine), NumPy, SciPy |
| PDF Engine | WeasyPrint (with CSS Cache) / ReportLab |
| Database | SQLite (Local) / PostgreSQL (Prod) |
| Storage | Local Disk / AWS S3 (Configurable) |
| AI | OpenRouter (Free Models Fallback Chain) + OpenAI (GPT-4o) |

---

## Project Structure
```
GetReport/
├── Frontend/               # React frontend (Vite + TypeScript)
│   ├── src/
│   │   ├── components/     # FileUpload, DataPreview, ChatInterface, IssueLedger, ProcessPipeline, ReportGeneration
│   │   ├── pages/          # Index, Workspace, Dashboard, Features, HowItWorks, ApiDocs, Documentation, etc.
│   │   └── services/       # API integration layer (api.ts)
├── Backend/
│   ├── app/
│   │   ├── api/            # FastAPI router configuration
│   │   │   ├── endpoints.py # Core router aggregator and orchestrator
│   │   │   └── routes/      # Focused, single-responsibility route handlers
│   │   │       ├── upload.py # Single & batch file ingestion with SHA-256 calculation
│   │   │       ├── status.py # Polling & WebSockets status updates
│   │   │       ├── report.py # PDF download endpoints
│   │   │       ├── chat.py   # RAG-powered Q&A endpoint
│   │   │       └── issues.py # Issue Ledger CRUD and state management
│   │   ├── core/           # Security, authentication, and system configs
│   │   │   ├── config.py    # Pydantic Settings management
│   │   │   ├── auth.py      # Header & query API key verification
│   │   │   ├── limiter.py   # API rate limiting
│   │   │   ├── file_validation.py # Magic number and signature verification (XLSX, Parquet, Feather, GZ)
│   │   │   ├── rag_utils.py # TableAwareTextSplitter & TFIDFVectorStore fallback engine
│   │   │   ├── security_headers.py # HTTP security hardening headers
│   │   │   └── celery_app.py # Celery broker configuration
│   │   ├── services/       # Core business logic
│   │   │   ├── analysis/     # Modular Statistical Analysis Engine
│   │   │   ├── data_processing.py # Polars loader & automated cleaning rule application
│   │   │   ├── issue_ledger.py # Ledger generation & compute_dataset_fingerprint
│   │   │   ├── report_generator.py # PDF report orchestration
│   │   │   ├── report_weasyprint.py # Production HTML-to-PDF engine
│   │   │   ├── storage.py    # S3 / Database / Local storage adapter
│   │   │   ├── task_manager.py # DB Job management with batch_id & file_hash
│   │   │   └── rag_service.py # Hybrid Vector + Sparse search RAG service with local fallback
│   │   ├── db.py           # Sync/Async wrappers for SQLite and PostgreSQL (batch_id & file_hash fields)
│   │   └── tasks.py        # Asynchronous Celery task definitions
│   └── Dockerfile          # Multi-stage production container build
├── render.yaml             # Render deployment blueprint (Infrastructure as Code)
└── README.md               # Main project documentation
```

---

## Setup and Running

### Backend (Local)
```bash
cd Backend
# Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run server (Defaults to PDF_ENGINE=reportlab)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Local)
```bash
cd Frontend
npm install
npm run dev
```

Access the application at `http://localhost:8080` (or `http://localhost:5173`)

---

## Deployment (Production)

The project is optimized for deployment on **Render.com** (or any Docker-based cloud environment).

### Deployment Artifacts
- **render.yaml**: Blueprint for fully automated deployment (API, Worker, Redis, Frontend).
- **Dockerfile**: Multi-stage build installing WeasyPrint system dependencies (Pango, Cairo) and SSL certs.

### Environment Variables
| Variable | Default (Local) | Production | Description |
|----------|-----------------|------------|-------------|
| `PDF_ENGINE` | `reportlab` | `weasyprint` | PDF Rendering engine choice (`reportlab` avoids system dependencies in local dev). |
| `DATABASE_URL` | (empty) -> uses SQLite | `postgres://...` | DB connection string. Local dev defaults to SQLite (`Backend/data/tasks.db`). |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://...` | Connection URL for Celery message broker/backend. |
| `STORAGE_TYPE` | `local` | `db` | Storage provider: `local` (disk), `db` (database blob storage), or `s3`. |
| `AWS_ACCESS_KEY_ID` | (optional) | (required for S3) | AWS access key for S3 file storage option. |
| `AWS_SECRET_ACCESS_KEY` | (optional) | (required for S3) | AWS secret key for S3 file storage option. |
| `AWS_BUCKET_NAME` | (optional) | (required for S3) | S3 Bucket name for file uploads and outputs. |
| `OPENAI_API_KEY`| (required for AI) | (required for AI) | OpenAI API Key for GPT-4o analytics features. |
| `OPENROUTER_API_KEY` | (optional) | (optional) | Alternative LLM provider key to use OpenRouter free model fallback chain. |
| `API_KEY` | (empty) | (optional) | Restricts client API access. If set, requires `X-API-Key` request header. |
| `CORS_ORIGINS` | `http://localhost:5173,...` | (restrict in prod) | Comma-separated list of allowed origins. |
| `RATE_LIMIT_ENABLED` | `True` | `True` | Set to `False` to disable API rate-limiting. |
| `DB_POOL_MIN_SIZE` | `1` | `1` | Minimum database connections in pool. |
| `DB_POOL_MAX_SIZE` | `10` | `10` | Maximum database connections in pool to avoid PostgreSQL connection exhaustion. |
| `MAX_EXCEL_DECOMPRESSED_SIZE_MB` | `200` | `200` | Max decompressed XML size for uploaded Excel zip archives to mitigate Zip Bomb attacks. |

---

## Architecture Highlights

### Asynchronous Pipeline
`Client` -> `API` -> `Redis` -> `Celery Worker` -> `PDF Engine`

### Two-Stage Pipeline
1. **Inspection Phase**: Single or batch file upload triggers immediate data profiling and SHA-256 / dataset fingerprint deduplication. User reviews quality issues and selects cleaning actions.
2. **Analysis Phase**: Approved rules are applied, full statistical analysis runs, and PDF is generated.

---

## Performance & Scalability Limits

GetReport is designed to handle business-scale datasets efficiently. The following limits ensure system stability, 100% computational accuracy, and memory safety without choking active processes:

### Data Volume & Row Capacity
- **CSV / TSV Datasets**: Supports up to **50 MB** uploaded files (~500,000 to 1,000,000 rows). Polars compiles and analyzes these datasets in **under 200ms**.
- **Parquet / Feather / JSON / Arrow**: Native compressed binary parsing with Polars lazy streaming.
- **Excel Datasets (`.xlsx`/`.xls`)**: Supports up to **50 MB** uploads, with a strict decompressed XML memory limit of **200 MB** to guard against Zip Bomb attacks.

### Statistical & Insight Accuracy
- **100% Mathematical Accuracy**: All cleaning steps (Winsorization, missing value imputation) and statistical calculations are performed eagerly by Polars on the **entire dataset**.
- **Dynamic Downsampling for LLM Insights**: Dataset summaries are token-checked using `tiktoken`. Prompts are capped at **3,000 tokens** (~12,000 characters), dynamically truncating wide metrics while maintaining representative statistical context.

---

## License
MIT License - See LICENSE file for details.
