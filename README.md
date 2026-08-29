# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![Google Antigravity SDK](https://img.shields.io/badge/Agent-Google%20Antigravity%20SDK-4285F4?logo=google)](https://ai.google.dev/)
[![Celery](https://img.shields.io/badge/Celery-5.3.6%2B-37814A?logo=celery)](https://docs.celeryq.dev/)
[![Polars](https://img.shields.io/badge/Engine-Polars%20Rust-CD412B?logo=rust)](https://pola.rs/)
[![WeasyPrint](https://img.shields.io/badge/PDF_Engine-WeasyPrint%2061.2%2B-FF6600)](https://weasyprint.org/)
[![Tests](https://img.shields.io/badge/Tests-234%20Passed-brightgreen)](https://github.com/)

---

## Overview
GetReport is an automated data intelligence and exploratory data analysis (EDA) platform that transforms raw, multi-format datasets into publication-ready PDF reports, machine learning features, and conversational intelligence.

It combines a Polars data engine, non-parametric statistical estimators, forensic Benford's Law anomaly detection, symbolic equation discovery, multivariate conditional imputation (MICE), and hybrid-search retrieval-augmented generation (RAG).

---

## Problem Solved
Traditional data analysis workflows require:
- Deep statistical and programming expertise
- Time-consuming manual data profiling, cleaning, and sanity checking
- Fragmented tooling for analysis, feature engineering, and document generation

GetReport unifies this workflow into a single platform:
1. Ingests single or multi-file datasets with automatic encoding and format resilience.
2. Identifies data quality anomalies through an interactive Issue Ledger with user-approved remediations.
3. Computes statistical distributions, correlations, outliers, missingness structures, and time-series trends.
4. Generates executive PDF reports and machine-learning-ready engineered features.

---

## Key Capabilities & Algorithms

### 1. Statistical Core and Exploratory Data Analysis
- **Non-Parametric Dispersion and Shape**: Calculates Median Absolute Deviation (MAD), Interquartile Range (IQR), 5% Trimmed Mean, Coefficient of Variation (CV), and Bowley's resistant quartile skewness alongside standard moments.
- **Correlation and Multicollinearity**: Evaluates Pearson (r) and Spearman rank-order (rho) correlation matrices with zero-variance protection on constant features and multicollinearity alerts (|r| >= 0.90).
- **Skewness-Adjusted Outlier Bounds**: Dynamically adjusts Tukey fences using sample skewness to avoid false-positive alarms on naturally skewed distributions.
- **Categorical Information Theory**: Computes Shannon Entropy, Normalized Evenness, Simpson's Diversity index, and identifies rare categories (<1%).
- **Missingness Pattern Diagnostics**: Computes pairwise Phi-coefficient matrices to classify MCAR, MAR, and MNAR co-occurrences, along with listwise deletion row-survival estimators.
- **Time-Series Analysis**: Non-parametric Mann-Kendall monotonic trend test with lag-1, lag-7, and lag-30 autocorrelation analysis.

### 2. Forensic Confidence Scoring and Integrity Auditing
- **Benford's Law Forensic Audit**: Analyzes leading digit distributions against the logarithmic first-digit law using Pearson Chi-Square goodness-of-fit to detect fabricated or manipulated numbers.
- **Role-Adaptive Confidence Weighting**: Dynamically calibrates Completeness, Consistency, Validity, and Stability weights according to inferred column roles (identifiers, metrics, categories, dates).
- **Population Stability Index (PSI)**: Monitors distribution shifts and concept drift across dataset versions.

### 3. Smart Schema and Relational Discovery
- **Symbolic Linear Equation Discovery**: Discovers arithmetic relationships across numeric features (sums, differences, products, ratios) with high confidence.
- **Functional Dependency Mining (X -> Y)**: Identifies determinant relationships where attribute values uniquely specify dependent attributes.
- **Entity Disambiguation**: Uses Jaro-Winkler string similarity to cluster typographical variations and near-duplicate entities into canonical values.

### 4. Feature Engineering and Imputation
- **Empirical Bayes Target Encoding**: Smoothed category encodings with Out-of-Fold (OOF) K-Fold regularization to prevent target leakage.
- **Fourier Cyclical Embeddings**: Maps cyclical calendar and temporal attributes into continuous sine and cosine coordinates.
- **Multivariate MICE Imputation**: Implements Multivariate Imputation by Chained Equations using iterative Ridge regression to preserve joint feature covariances.
- **Interaction Synthesis**: Generates non-linear feature products and safe numerical ratios.

### 5. Ingestion Engine
- **Multi-Encoding Fallback**: Automatically decodes UTF-8, UTF-8-BOM (utf-8-sig), ISO-8859-1, Latin-1, Windows-1252, and UTF-16.
- **Delimiter Detection**: Detects comma, semicolon, tab, pipe, and colon separators.
- **Format Support**: Ingests CSV, TSV, XLS, XLSX, Parquet, JSON, JSONL, NDJSON, Feather, Arrow, and GZ archives.
- **Batch Processing**: Supports multi-dataset uploads grouped under a unified session batch identifier.
- **Security Sandboxing**: Enforces magic byte verification, strict path resolution, and zip bomb decompression limits.

### 6. Document Generation
- **Executive Styling**: Consistent document layout with custom typography, headers, dynamic pagination, and confidentiality notices.
- **Dual PDF Engines**: Production HTML/CSS rendering via WeasyPrint and standalone rendering via ReportLab.

### 7. Conversational RAG
- **Google Antigravity Agent**: Grounded analytical Q&A powered by the Google Antigravity SDK with direct dataset execution tools.
- **Hybrid Search**: Combines dense semantic vector search with reciprocal rank fusion (RRF) and deterministic fallback stores.

---

## Tech Stack

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 + Vite |
| Language | TypeScript |
| Styling | Tailwind CSS, Vanilla CSS, Shadcn/UI |
| Navigation | React Router DOM v6 |
| Icons | Lucide React |
| State and Fetching | TanStack Query |

### Backend
| Component | Technology |
|---|---|
| Framework | FastAPI (Python 3.12+) with Pydantic v2 |
| AI Agent | Google Antigravity SDK (google-antigravity) |
| LLM Providers | Google Gemini (2.5/3.7 Flash), OpenRouter, OpenAI |
| Task Queue | Celery + Redis |
| Data Processing | Polars (Rust Engine), NumPy, SciPy |
| PDF Engine | WeasyPrint / ReportLab |
| Database | SQLite (WAL Mode) / PostgreSQL (pgvector) |
| Storage | Local Sandboxed Disk / Database BYTEA / AWS S3 |

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

# Run server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (Local)
```bash
cd Frontend
npm install
npm run dev
```

Access the application at `http://localhost:8080` (or `http://localhost:5173`).

---

## Testing and Verification

GetReport includes a test suite covering unit tests, property-based hypothesis tests, edge cases, and integration workflows:

```bash
cd Backend
pytest -v
```

```
============================ 234 passed in 38.51s =============================
```

- **234 Automated Tests**: Covering statistical estimators, forensic tests, single-pass streaming ingestion, database composite indexes, vector SVG charting, Polars lazy streaming execution, MICE covariance preservation, WAL concurrency, entity clustering, symbolic equation discovery, memory allocation bounds, and database concurrency.
- **Zero Failures and Zero Warnings**.

---

## Deployment (Production)

The application is configured for deployment on Render or any Docker-compatible infrastructure.

### Deployment Files
- **render.yaml**: Infrastructure blueprint for web API, background worker, Redis instance, and frontend static site.
- **Dockerfile**: Production container image with required Pango and Cairo rendering libraries.

### Environment Configuration
| Variable | Default (Local) | Production | Description |
|---|---|---|---|
| `PDF_ENGINE` | `reportlab` | `weasyprint` | PDF Rendering engine (`reportlab` for lightweight local dev, `weasyprint` for production). |
| `DATABASE_URL` | (empty) -> SQLite | `postgres://...` | Database connection string. |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://...` | Message broker URL for Celery. |
| `STORAGE_TYPE` | `local` | `db` | Storage provider (`local`, `db`, or `s3`). |
| `AWS_ACCESS_KEY_ID` | (optional) | (required for S3) | AWS access key. |
| `AWS_SECRET_ACCESS_KEY` | (optional) | (required for S3) | AWS secret key. |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | (optional) | (recommended) | API Key for Google Antigravity Agent and vector embeddings. |
| `OPENAI_API_KEY` | (optional) | (optional) | OpenAI API Key. |
| `OPENROUTER_API_KEY` | (optional) | (optional) | OpenRouter API Key for fallback provider chain. |
| `API_KEY` | (empty) | (optional) | Optional API key requirement for client requests (X-API-Key header). |
| `CORS_ORIGINS` | `http://localhost:5173,...` | (configured domain) | Allowed origin domains for CORS. |
| `RATE_LIMIT_ENABLED` | `True` | `True` | Rate limiting toggle. |
| `DB_POOL_MIN_SIZE` | `1` | `1` | Minimum pool connections. |
| `DB_POOL_MAX_SIZE` | `10` | `10` | Maximum pool connections. |
| `MAX_EXCEL_DECOMPRESSED_SIZE_MB` | `200` | `200` | Decompression limit for Excel zip bomb prevention. |

---

## License
MIT License - See LICENSE file for details.
