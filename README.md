# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![Google Antigravity SDK](https://img.shields.io/badge/Agent-Google%20Antigravity%20SDK-4285F4?logo=google)](https://ai.google.dev/)
[![Polars](https://img.shields.io/badge/Engine-Polars%20Rust-CD412B?logo=rust)](https://pola.rs/)
[![DuckDB](https://img.shields.io/badge/OLAP-DuckDB%201.0%2B-FFF000?logo=duckdb)](https://duckdb.org/)
[![Great Expectations](https://img.shields.io/badge/Contracts-Great%20Expectations-FF5A00)](https://greatexpectations.io/)
[![WeasyPrint](https://img.shields.io/badge/PDF_Engine-WeasyPrint%2061.2%2B-FF6600)](https://weasyprint.org/)
[![Security](https://img.shields.io/badge/Security-0%20Vulnerabilities%20%7C%20Audit%20Hardened-brightgreen?logo=shield)](https://github.com/)
[![Tests](https://img.shields.io/badge/Tests-332%20Passed-brightgreen)](https://github.com/)

---

## Overview
**GetReport** is an enterprise-grade automated data intelligence and exploratory data analysis (EDA) platform. It transforms raw, multi-format datasets into publication-ready PDF reports, machine-learning-ready engineered feature pipelines, interactive visual analytics, and conversational intelligence.

GetReport unifies high-performance data processing, mathematical profiling, and agentic AI:
- **Zero-Copy Hybrid Engine**: Combines **Polars (Rust)** for streaming aggregations with **DuckDB** for in-process SQL OLAP queries over Apache Arrow tables.
- **Advanced Statistical Profiling & Time-Series**: Non-parametric dispersion estimators (MAD, IQR, Trimmed Mean), Bowley quartile skewness, Phik ($\Phi_K$) non-linear correlation matrices, Kats/Mann-Kendall trend detection, and Benford's Law forensic audits.
- **High-Dimensional Visual Analytics**: Multi-parameter Parallel Coordinates plots powered by HiPlot and responsive SVG charts.
- **Enterprise Data Governance**: Interactive Issue Ledger ("Jira for Dirty Data"), Great Expectations data contracts exporter, and immutable cryptographic DAG provenance.
- **Dual-Level RAG & Conversational Sandboxing**: Dual-level local/global Knowledge Graph traversal (LightRAG pattern), grounded Q&A with Google Antigravity Agent, and AST-sandboxed Python analyst runtime.
- **Zero-Vulnerability Security Architecture**: Complete defensive remediation across all attack surfaces: strict Polars AST whitelisting, WeasyPrint SSRF/HTML autoescaping, CSV formula injection neutralization (CWE-1236), DuckDB memory bounds, watchdog daemon thread termination, and fail-closed authentication.

---

## Core Capabilities & Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Raw Dataset Ingestion       │
                                  │ (CSV, Excel, Parquet, JSON)   │
                                  └───────────────┬───────────────┘
                                                  │ (Single-Pass Stream & Magic Bytes)
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ Polars Rust Analytical Engine │
                                  └───────┬───────────────┬───────┘
                     ┌────────────────────┘               └────────────────────┐
                     ▼                                                         ▼
     ┌───────────────────────────────┐                         ┌───────────────────────────────┐
     │  Issue Ledger & Remediation   │                         │ DuckDB In-Process OLAP Engine │
     │  (Detect, Review, Lock, Exec) │                         │ (Zero-Copy Arrow, Sandboxed)  │
     └───────────────┬───────────────┘                         └───────────────┬───────────────┘
                     │                                                         │
                     ▼                                                         ▼
     ┌───────────────────────────────┐                         ┌───────────────────────────────┐
     │ Transformation DAG Lineage    │                         │ Interactive SQL & Golden Store│
     │ (Audit-Grade Provenance)      │                         │ (Few-Shot Natural Language Q&A│
     └───────────────┬───────────────┘                         └───────────────┬───────────────┘
                     │                                                         │
                     ▼                                                         ▼
     ┌───────────────────────────────┐                         ┌───────────────────────────────┐
     │ Virtual Concept Synthesizer   │                         │ Conversational Agent & Charts │
     │ (Dynamic Formula Compilation) │                         │ (AST Sandbox + Matplotlib Agg)│
     └───────────────┬───────────────┘                         └───────────────┬───────────────┘
                     │                                                         │
                     └────────────────────┬────────────────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Publication-Ready Deliverable  │
                         │ (Executive PDF + GX Contracts)  │
                         └─────────────────────────────────┘
```

### 1. In-Process DuckDB OLAP & Golden Query Store
- **Hardware Zero-Copy Execution**: Maps cleaned Polars DataFrames directly to DuckDB views via in-memory Apache Arrow record batches, querying millions of rows at memory-bus speeds without serialization overhead.
- **Engine-Level Sandboxing**: DuckDB instances enforce `SET enable_external_access = false;` at the native C++ level, preventing any filesystem access or socket connections from user queries.
- **Golden Query Catalog**: Curates verified natural-language-to-SQL query pairs with execution metrics, schema caching, and continuous regression testing.

### 2. Statistical Core & Advanced Data Profiling
- **Non-Parametric Dispersion and Shape**: Calculates Median Absolute Deviation (MAD), Interquartile Range (IQR), 5% Trimmed Mean, Coefficient of Variation (CV), and Bowley quartile skewness alongside Pearson moments.
- **Non-Linear Bivariate Correlation ($\Phi_K$)**: Evaluates non-linear and mixed-type associations across categorical, ordinal, and continuous intervals simultaneously.
- **Extended Profiling Alert Taxonomy**: Identifies constant columns, high cardinality, negative monetary values, bimodal distributions, uniform densities, and zero-inflation patterns.
- **Skewness-Adjusted Outlier Fences**: Dynamically adjusts Tukey multiplier bounds according to sample skewness, preventing false alarms on heavy-tailed or monetary distributions.
- **Missingness Structure Diagnostics**: Evaluates pairwise Phi-coefficient missingness matrices to classify MCAR, MAR, and MNAR structures, providing listwise deletion row-survival predictions.

### 3. Forensic Confidence Scoring & Integrity Auditing
- **Benford's Law Forensic Audit**: Analyzes leading digit distributions against the logarithmic first-digit law using Pearson Chi-Square goodness-of-fit to uncover fabricated, synthetic, or manipulated numbers.
- **Role-Adaptive Confidence Weighting**: Calibrates Completeness, Consistency, Validity, and Stability scores according to inferred column roles (identifiers, metrics, categories, dates).
- **Population Stability Index (PSI)**: Monitors distribution drift and concept shifts across dataset versions.

### 4. Conversational Sandboxing & Visual Concept Synthesis
- **AST-Sandboxed Python Analyst Agent**: Parses generated code into an AST before execution. Blocks all forbidden imports (`os`, `sys`, `subprocess`, `socket`, `requests`, `pathlib`), reflection dunder attributes (`__class__`, `__subclasses__`, `__dict__`), and dangerous calls (`eval`, `exec`, `open`, `compile`).
- **Headless Matplotlib Generation**: Captures high-resolution plots via Matplotlib's `Agg` headless backend, outputting Base64-encoded PNG cards directly into the conversational UI.
- **Virtual Concept Synthesizer**: Translates natural-language intent or raw formulas (e.g. `margin = (revenue - cost) / revenue`) into valid, optimized Polars expressions, compiling them to augmented columns and recording provenance in the `TransformationDAG`.

### 5. Data Governance, Contracts & Lineage
- **Issue Ledger ("Jira for Dirty Data")**: Identifies quality defects, produces automated remediation code, and enforces an approve/reject/modify lifecycle before applying fixes in a restricted Python scope.
- **Great Expectations (GX) Contract Exporter**: Translates profile constraints into production-ready `ExpectationSuite` specifications (`.json` and standalone `.py` scripts) with column type, null percentage, and value range checks.
- **Transformation DAG**: Tracks all dataset mutations with cryptographic data hashes, parent/child node linkages, execution durations, and automated reversibility drop hints.

### 6. Time-Series Intelligence & High-Dimensional Analytics (Kats & HiPlot)
- **Kats Time-Series Forecasting & Anomaly Engine**: Performs monotonic trend analysis via Mann-Kendall tests, computes seasonal decomposition (additive/multiplicative STL) across temporal columns, and pinpoints structural changepoints.
- **HiPlot Parallel Coordinates**: High-dimensional multi-parameter visual analytics rendering continuous and categorical axes with interactive brush selection and sub-population slicing.
- **LightRAG Dual-Level Knowledge Graph**: Combines low-level entity-relationship extractions with high-level global dataset themes for grounded exploratory Q&A.

### 7. Zero-Vulnerability Security Architecture (Audit-Hardened)
GetReport enforces defense-in-depth security across every ingestion, execution, and rendering boundary:
- **Strict Polars AST Whitelisting (`VULN-01`)**: The concept synthesis AST validator strictly allows only safe root tokens (`ALLOWED_POLARS_ROOTS` = `{"pl", "col", "lit", "when"}`), blocking arbitrary attribute access, file I/O operations (`read_*`, `scan_*`, `pipe`, `map_elements`), functions, and lambda expressions.
- **Zero-Bypass HTML/SVG Escaping (`VULN-02`)**: Complete neutralization of stored/reflected XSS in executive report PDF rendering. AI insights, executive summaries, and recommendations are strictly autoescaped. Vector SVG charts are guarded against `<script>` and `<foreignObject>` tags.
- **CSV Formula Injection (CWE-1236) Neutralization (`VULN-03`)**: Neutralizes Dynamic Data Exchange (DDE) and formula code execution in spreadsheet applications (Excel, Calc) by auto-sanitizing text fields starting with `=, +, -, @, \t, \r` with prepended single quotes (`'`) across all CSV export endpoints.
- **DuckDB OOM DoS & Memory Hardening (`VULN-04`)**: Prevents out-of-memory denial of service by strictly capping analytical query materialization via Polars `.head(max_rows)` before `.to_dicts()` conversion, and constraining the API `limit` parameter with Pydantic validation (`1 <= limit <= 1000`).
- **Watchdog Thread Termination (`VULN-05`)**: Python analyst execution threads are monitored by an active watchdog timer; runaway or spinning threads are asynchronously terminated via `PyThreadState_SetAsyncExc(tid, SystemExit)` to prevent daemon thread CPU exhaustion, with AST loop depth capped to 3.
- **Fail-Closed API & WebSocket Authentication (`VULN-06`)**: WebSocket and REST authentication enforce fail-closed security when `DATABASE_URL` is configured, preventing unauthenticated access in production. The frontend client automatically propagates the `X-API-Key` header across all REST and WS connections.
- **Real Client IP Spoofing Prevention (`VULN-07`)**: The global rate limiter extracts trusted client IPs by prioritizing authenticated edge proxy headers (Cloudflare `CF-Connecting-IP`) over easily spoofed `X-Forwarded-For` header chains.
- **CRLF & Header Injection Defense (CWE-113) (`VULN-08`)**: Request ID middleware validates all incoming `X-Request-ID` headers against a strict whitelist regex (`^[a-zA-Z0-9_-]{1,64}$`), discarding any newline (`\r`, `\n`) or delimiter injection payloads.
- **DuckDB Native Sandbox**: Native C++ external access disabled (`SET enable_external_access = false;`), preventing local file inclusion (LFI), network calls, or unauthorized file reads.

---

## Tech Stack

### Frontend
| Component | Technology |
|---|---|
| **Framework** | React 18 + Vite |
| **Language** | TypeScript 5.8 |
| **Styling** | Tailwind CSS 3.4, Vanilla CSS, Shadcn/UI primitives |
| **Sanitization** | DOMPurify |
| **Charts & Motion** | Recharts, Framer Motion, Lucide React |
| **State & Data** | TanStack Query v5 |

### Backend
| Component | Technology |
|---|---|
| **Framework** | FastAPI (Python 3.12+) with Pydantic v2 |
| **OLAP Engine** | DuckDB 1.0+ (In-Process, Apache Arrow Zero-Copy) |
| **Data Engine** | Polars (Rust Core), NumPy, SciPy, Scikit-Learn |
| **AI Agent** | Google Antigravity SDK (`google-antigravity`) |
| **LLM Providers** | Google Gemini (2.5/3.7 Flash), OpenRouter, OpenAI |
| **Contracts** | Great Expectations (GX) |
| **Task Queue** | Celery + Redis |
| **PDF Engines** | WeasyPrint (Production HTML/CSS) / ReportLab (Local) |
| **Storage** | Sandboxed Local Disk / PostgreSQL BYTEA / AWS S3 |

---

## API Reference (Key Endpoints)

### Ingestion & Jobs
- `POST /api/upload`: Single-pass streaming ingestion with magic byte signature verification.
- `POST /api/upload/batch`: Multi-file ingestion under a unified batch identifier.
- `POST /api/upload/join`: Multi-dataset relational joins (`inner`, `left`, `full`, `anti`, `semi`).
- `GET /api/status/{task_id}`: Polling endpoint for job progress and structured results.
- `WS /api/ws/status/{task_id}`: Real-time WebSocket stream with 15s heartbeats and Redis PubSub.

### OLAP & Data Contracts
- `POST /api/jobs/{task_id}/query-sql`: Executes read-only, hardware-accelerated SQL via in-process DuckDB over zero-copy Arrow batches (memory-bounded).
- `GET /api/jobs/{task_id}/golden-queries`: Lists verified few-shot query examples for the dataset.
- `POST /api/jobs/{task_id}/golden-queries`: Registers a verified natural-language-to-SQL pair.
- `DELETE /api/jobs/{task_id}/golden-queries/{query_id}`: Deletes an existing golden query pair.
- `GET /api/jobs/{task_id}/export-gx`: Generates a production Great Expectations (GX) data contract (`.json` or `.py`).
- `GET /api/jobs/{task_id}/export/{export_format}`: Securely exports sanitized dataset (`csv` with formula injection neutralization, `excel`, `parquet`, `json`, `pdf`).

### Virtual Concepts & Sandboxing
- `POST /api/jobs/{task_id}/sandbox-exec`: Executes arbitrary user analytical code in the AST sandbox with Matplotlib plot capture and watchdog thread termination.
- `POST /api/jobs/{task_id}/concepts/derive`: Derives a new column via whitelisted Polars expressions, updates the stored dataset, and records DAG audit provenance.
- `GET /api/jobs/{task_id}/concepts`: Lists all derived virtual concepts for the dataset.

### Issue Ledger & Governance
- `GET /api/jobs/{task_id}/issues`: Retrieves detected data quality issues and proposed fixes.
- `POST /api/jobs/{task_id}/issues/{issue_id}/approve`: Approves a single remediation for execution.
- `POST /api/jobs/{task_id}/issues/{issue_id}/reject`: Rejects a proposed remediation.
- `POST /api/jobs/{task_id}/issues/approve-all`: Batch-approves all pending remediations.
- `POST /api/jobs/{task_id}/issues/reject-all`: Batch-rejects all pending remediations.
- `POST /api/jobs/{task_id}/issues/lock`: Locks the ledger, applying approved fixes and recording audit node.
- `GET /api/jobs/{task_id}/dag`: Retrieves the complete audit lineage graph.
- `GET /api/jobs/{task_id}/dag/summary`: High-level summary of transformations and reversibility hints.

### Reports & Conversational RAG
- `POST /api/jobs/{task_id}/report`: Asynchronously schedules PDF compilation via Celery / WeasyPrint.
- `GET /api/jobs/{task_id}/report`: Retrieves generated report JSON payload.
- `GET /api/jobs/{task_id}/report/full`: Retrieves comprehensive executive report with all modules.
- `GET /api/jobs/{task_id}/comparison`: Compares two datasets or versions across schema and statistics.
- `POST /api/jobs/{task_id}/chat`: Context-aware Q&A with smart dataset fallback and dual-level Knowledge Graph.
- `POST /api/jobs/{task_id}/chat/stream`: Real-time Server-Sent Events (SSE) token streaming.

---

## Getting Started

### Prerequisites
- **Python 3.12+**
- **Node.js 18+ & npm**
- **Redis** (optional for local dev, required for Celery task queuing)

### Backend Setup
```bash
# 1. Navigate to Backend directory
cd Backend

# 2. Activate virtual environment
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
# 1. Navigate to Frontend directory
cd Frontend

# 2. Install dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

Open your browser at `http://localhost:8080` (or `http://localhost:5173`).

---

## Verification & Testing

GetReport maintains an exhaustive automated test suite verifying mathematical accuracy, security constraints, and UI compilation:

```bash
cd Backend
pytest -q
```

```
................................................................................................... [ 30%]
................................................................................................... [ 60%]
................................................................................................... [ 90%]
................................                                                                    [100%]
=================================== 332 passed in 66.98s ====================================
```

### Test Suite Highlights
- **17 Security Verification Tests** (`test_security_audit.py`): Validates complete 0-vulnerability posture across the full application surface:
  - DuckDB LFI rejection and C++ engine filesystem block (`SET enable_external_access = false;`)
  - AST sandbox dunder blocking, subscript reflection defense, and format string introspection protection
  - Concept synthesizer AST whitelisting: blocks unauthorized method invocations, file I/O (`read_*`, `scan_*`), lambdas, and functions
  - Issue Ledger execution namespace isolation and AST validation
  - RAG HTML formatting XSS neutralization preserving safe semantic tags
  - CSV formula injection (CWE-1236) neutralization (`=`, `+`, `-`, `@`, `\t`, `\r` prefix apostrophe-escaping)
  - DuckDB query result memory bounds (`max_rows` cap via Polars `.head()`)
  - WebSocket authentication fail-closed enforcement when `DATABASE_URL` is configured
  - HTTP Header / CRLF injection defense (CWE-113) with regex-enforced Request IDs
  - Cloudflare `CF-Connecting-IP` real client IP spoofing defense
- **5 DuckDB Engine Tests** (`test_duckdb_engine.py`): Validates zero-copy Arrow queries, profile calculations, and statement-chaining defenses.
- **9 Advanced RAG & Graph Pipeline Tests** (`test_rag_advanced.py`, `test_rag_graph_pipeline.py`): Validates dual-level knowledge graph extraction, token streaming, prompt injection filters, and footnote citations.
- **14 Sandboxing & Synthesis Tests** (`test_sandboxed_analyst.py`, `test_concept_synthesizer.py`, `test_phase3_api_endpoints.py`): Validates sandboxed execution, plot rendering, thread timeout termination, and DAG persistence.
- **Kats & Time-Series Tests** (`test_timeseries_kats.py`): Validates Mann-Kendall monotonic trend tests, seasonal decomposition, and changepoint detection.
- **HiPlot Parallel Coordinates Tests** (`test_hiplot_service.py`): Validates multi-dimensional parallel coordinate rendering and parameter bounds.
- **Frontend Production Build**: Verified with `npm run build` (0 TypeScript errors, clean bundle compilation).

---

## Production Deployment

### Docker Container
The included `Dockerfile` packages the complete backend along with Pango, Cairo, and font libraries for high-fidelity WeasyPrint PDF compilation:

```bash
docker build -t getreport-backend -f Backend/Dockerfile .
docker run -p 8000:8000 --env-file Backend/.env.example getreport-backend
```

### Environment Variables
| Variable | Default (Local) | Production | Description |
|---|---|---|---|
| `PDF_ENGINE` | `reportlab` | `weasyprint` | PDF engine (`reportlab` for lightweight local dev, `weasyprint` for production). |
| `DATABASE_URL` | (empty) -> SQLite | `postgres://...` | PostgreSQL connection string (supports pgvector). |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://...` | Redis broker for Celery and WebSocket PubSub. |
| `STORAGE_TYPE` | `local` | `db` / `s3` | File storage provider (`local`, `db`, or `s3`). |
| `API_KEY` | (empty) | (secret key) | Enforces `X-API-Key` header authentication on API endpoints. |
| `CORS_ORIGINS` | `http://localhost:5173` | `https://get-report.vercel.app` | Allowed CORS origins. |
| `GEMINI_API_KEY` | (optional) | (recommended) | API Key for Google Antigravity Agent and vector embeddings. |
| `MAX_UPLOAD_SIZE_MB` | `50` | `50` | Maximum file upload limit in megabytes. |
| `RATE_LIMIT_ENABLED` | `True` | `True` | Global rate limiter toggle. |

---

## License
MIT License. See [LICENSE](LICENSE) for details.
