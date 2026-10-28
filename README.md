# GetReport

## Motto
**Turn Your Data Into Professional Reports in Seconds.**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev/)
[![UI](https://img.shields.io/badge/UI-Obsidian%20Executive%20%7C%20BeUI-000000?logo=react)](https://beui.dev/)
[![BI](https://img.shields.io/badge/BI-Apache%20ECharts%206.1-AA344D?logo=apacheecharts)](https://echarts.apache.org/)
[![Warehouse](https://img.shields.io/badge/Warehouse-Snowflake%20Arrow%20Stream%20%28Beta%29-29B5E8?logo=snowflake)](https://www.snowflake.com/)
[![Google Antigravity SDK](https://img.shields.io/badge/Agent-Google%20Antigravity%20SDK-4285F4?logo=google)](https://ai.google.dev/)
[![Polars](https://img.shields.io/badge/Engine-Polars%20Rust-CD412B?logo=rust)](https://pola.rs/)
[![DuckDB](https://img.shields.io/badge/OLAP-DuckDB%201.0%2B-FFF000?logo=duckdb)](https://duckdb.org/)
[![PostgreSQL](https://img.shields.io/badge/Database-Neon%20PostgreSQL-336791?logo=postgresql)](https://neon.tech/)
[![Typst](https://img.shields.io/badge/PDF_Engine-Typst%200.15%2B-239DAD)](https://typst.app/)
[![Security](https://img.shields.io/badge/Security-0%20Vulnerabilities%20%7C%20Audit%20Hardened-brightgreen?logo=shield)](https://github.com/)
[![Tests](https://img.shields.io/badge/Tests-446%20Passed-brightgreen)](https://github.com/)
[![ML Anomaly Detection](https://img.shields.io/badge/ML-HistGradientBoosting%20%7C%20scikit--learn-F7931E?logo=scikitlearn)](https://scikit-learn.org/)

---

## Overview
**GetReport** is an enterprise-grade automated data intelligence, exploratory data analysis (EDA), and interactive Business Intelligence (BI) platform. It transforms raw, multi-format datasets into publication-ready executive PDF reports, machine-learning-ready engineered feature pipelines, interactive visual analytics dashboards, and grounded conversational intelligence.

GetReport unifies high-performance Rust/Python data processing, mathematical profiling, and agentic AI:
- **Zero-Copy Hybrid Engine**: Combines **Polars (Rust)** for single-pass streaming aggregations with **DuckDB** for in-process SQL OLAP queries directly over Apache Arrow tables.
- **Interactive Business Intelligence Studio (Apache ECharts)**: A Superset-grade exploratory visual canvas supporting dynamic multidimensional aggregations, multi-series breakdowns, custom color palettes, BigNumber KPI metric cards, and responsive mobile-first query controls (`/analytics` and `/bi`).
- **Direct Snowflake Warehouse Ingestion (Beta)**: Direct cloud data warehouse connectivity supporting RSA Key-Pair and password authentication, dedicated read-only role enforcement, in-memory AES-256 session encryption, and zero-copy Arrow streaming into parquet.
- **Row Intelligence & Self-Supervised Anomaly Detection**: 4-tier row triage hierarchy (`PROTECT`, `REVIEW`, `FIX`, `EXCLUDE`) ensuring bit-identical preservation of clean records, automatic pruning of non-data rows (headers, totals, footers) before profiling, gated smart imputation against synthetic data hallucination, and cross-column relationship violation detection via out-of-fold `HistGradientBoostingRegressor`.
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
                                  │ Polars Analytical Engine      │
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
     │ Business Intelligence Studio  │                         │ Conversational Agent & Charts │
     │ (ECharts Multidimensional BI) │                         │ (AST Sandbox + Matplotlib Agg)│
     └───────────────┬───────────────┘                         └───────────────┬───────────────┘
                     │                                                         │
                     └────────────────────┬────────────────────────────────────┘
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Publication-Ready Deliverable │
                         │ (Executive PDF + GX Contracts)  │
                         └─────────────────────────────────┘
```

### 1. Interactive Business Intelligence (BI) Studio
- **Superset-Grade Exploration**: Full-featured interactive visual analytics engine built directly into GetReport (`/analytics` and `/bi`), powered by Apache ECharts.
- **Dynamic Multidimensional Aggregations**: Compute grouped sums, averages, counts, minimums, and maximums across any combination of categorical dimensions and numerical metrics.
- **Multi-Series Breakdowns**: Split metrics across secondary categorical dimensions for comparative stacked and grouped visualizations.
- **Responsive Mobile Layout**: Features an adaptive segmented dual-tab switcher (`Chart & Visuals` vs `Query Controls`) with viewport-aware auto-resizing, ensuring charts render smoothly on mobile, tablet, and desktop form factors.
- **Real-Time BigNumber KPIs**: Live cards surfacing total records, aggregated metrics, peak groups, and group cardinality.
- **Executive Color Themes**: Instant palette switching across Obsidian Dark, Emerald Finance, Amber Warmth, and Minimalist Monochrome.

### 2. In-Process DuckDB OLAP & Golden Query Store
- **Hardware Zero-Copy Execution**: Maps cleaned Polars DataFrames directly to DuckDB views via in-memory Apache Arrow record batches, querying millions of rows at memory-bus speeds without serialization overhead.
- **Engine-Level Sandboxing**: DuckDB instances enforce `SET enable_external_access = false;` at the native C++ level, preventing any filesystem access or socket connections from user queries.
- **Golden Query Catalog**: Curates verified natural-language-to-SQL query pairs with execution metrics, schema caching, and continuous regression testing.

### 3. Statistical Core & Advanced Data Profiling
- **Non-Parametric Dispersion and Shape**: Calculates Median Absolute Deviation (MAD), Interquartile Range (IQR), 5% Trimmed Mean, Coefficient of Variation (CV), and Bowley quartile skewness alongside Pearson moments.
- **Non-Linear Bivariate Correlation ($\Phi_K$)**: Evaluates non-linear and mixed-type associations across categorical, ordinal, and continuous intervals simultaneously.
- **Extended Profiling Alert Taxonomy**: Identifies constant columns, high cardinality, negative monetary values, bimodal distributions, uniform densities, and zero-inflation patterns.
- **Skewness-Adjusted Outlier Fences**: Dynamically adjusts Tukey multiplier bounds according to sample skewness, preventing false alarms on heavy-tailed or monetary distributions.
- **Missingness Structure Diagnostics**: Evaluates pairwise Phi-coefficient missingness matrices to classify MCAR, MAR, and MNAR structures, providing listwise deletion row-survival predictions.

### 4. Row Intelligence & Self-Supervised Anomaly Detection
- **4-Tier Verdict Hierarchy (`RowVerdict`)**: Categorizes every row into a single explicit verdict before cleaning begins:
  - `PROTECT` (Severity 0): Passes all checks; bit-identical guarantee through downstream cleaning.
  - `REVIEW` (Severity 1): Suspicious or multi-column relationship violations; flagged with natural-language drivers for human audit, never mutated or dropped.
  - `FIX` (Severity 2): Safe, deterministic cell repairs (masked null placeholders like `"N/A"`, imputable nulls with $\le 30\%$ column missingness).
  - `EXCLUDE` (Severity 3): Non-data artifacts (blank rows, repeated headers, subtotal/total rows, export footers).
- **Structural Sanitation & Statistical Isolation**: Purges non-data rows in Step 1b so summary totals never double revenue metrics or distort averages, and header strings never corrupt numeric column types. All summary statistics and quartiles run solely across valid data rows.
- **Gated Smart Imputation**: Sparse rows ($\le 10\%$ populated) and heavily missing columns ($>30\%$ nulls) are strictly blocked from median/mode imputation, completely preventing the pipeline from hallucinating synthetic records.
- **Self-Supervised Cross-Column ML Layer (`HistGradientBoostingRegressor`)**: Trains out-of-fold regression models natively handling NaNs. Evaluates columns with $R^2 \ge 0.30$, applies two-pass outlier decontamination, scales residuals by missingness pattern MAD ($\text{MAD} \times 1.4826$), and flags extreme tail deviations ($Z \ge 6.0$, $\ge 2.5 \times 99\text{th percentile}$, $\le 1\%$ cap).
- **Contrastive Driver Explanations**: Generates plain-language diagnostics for flagged rows (e.g., `salary=95000 (expected about 35000 given the rest of the row)`).
- **DAG Lineage Provenance**: Records all pruned non-data rows under the `"drop_non_data_rows"` node with partial reversibility and stored row IDs.

### 5. Forensic Confidence Scoring & Integrity Auditing
- **Benford's Law Forensic Audit**: Analyzes leading digit distributions against the logarithmic first-digit law using Pearson Chi-Square goodness-of-fit to uncover fabricated, synthetic, or manipulated numbers.
- **Role-Adaptive Confidence Weighting**: Calibrates Completeness, Consistency, Validity, and Stability scores according to inferred column roles (identifiers, metrics, categories, dates).
- **Population Stability Index (PSI)**: Monitors distribution drift and concept shifts across dataset versions.

### 6. Conversational Sandboxing & Visual Concept Synthesis
- **AST-Sandboxed Python Analyst Agent**: Parses generated code into an AST before execution. Blocks all forbidden imports (`os`, `sys`, `subprocess`, `socket`, `requests`, `pathlib`), reflection dunder attributes (`__class__`, `__subclasses__`, `__dict__`), and dangerous calls (`eval`, `exec`, `open`, `compile`).
- **Headless Matplotlib Generation**: Captures high-resolution plots via Matplotlib's `Agg` headless backend, outputting Base64-encoded PNG cards directly into the conversational UI.
- **Virtual Concept Synthesizer**: Translates natural-language intent or raw formulas (e.g. `margin = (revenue - cost) / revenue`) into valid, optimized Polars expressions, compiling them to augmented columns and recording provenance in the `TransformationDAG`.

### 7. Data Governance, Contracts & Lineage
- **Issue Ledger ("Jira for Dirty Data")**: Identifies quality defects, produces automated remediation code, and enforces an approve/reject/modify lifecycle before applying fixes in a restricted Python scope.
- **Great Expectations (GX) Contract Exporter**: Translates profile constraints into production-ready `ExpectationSuite` specifications (`.json` and standalone `.py` scripts) with column type, null percentage, and value range checks.
- **Transformation DAG**: Tracks all dataset mutations with cryptographic data hashes, parent/child node linkages, execution durations, and automated reversibility drop hints.

### 8. Time-Series Intelligence & High-Dimensional Analytics (Kats & HiPlot)
- **Kats Time-Series Forecasting & Anomaly Engine**: Performs monotonic trend analysis via Mann-Kendall tests, computes seasonal decomposition (additive/multiplicative STL) across temporal columns, and pinpoints structural changepoints.
- **HiPlot Parallel Coordinates**: High-dimensional multi-parameter visual analytics rendering continuous and categorical axes with interactive brush selection and sub-population slicing.
- **LightRAG Dual-Level Knowledge Graph**: Combines low-level entity-relationship extractions with high-level global dataset themes for grounded exploratory Q&A.

### 9. Zero-Vulnerability Security Architecture (Audit-Hardened)
GetReport enforces defense-in-depth security across every ingestion, execution, and rendering boundary:
- **Strict Polars AST Whitelisting (`VULN-01`)**: The concept synthesis AST validator strictly allows only safe root tokens (`ALLOWED_POLARS_ROOTS` = `{"pl", "col", "lit", "when"}`), blocking arbitrary attribute access, file I/O operations (`read_*`, `scan_*`, `pipe`, `map_elements`), functions, and lambda expressions.
- **Zero-Bypass HTML/SVG Escaping (`VULN-02`)**: Complete neutralization of stored/reflected XSS in executive report PDF rendering. AI insights, executive summaries, and recommendations are strictly autoescaped. Vector SVG charts are guarded against `<script>` and `<foreignObject>` tags.
- **CSV Formula Injection (CWE-1236) Neutralization (`VULN-03`)**: Neutralizes Dynamic Data Exchange (DDE) and formula code execution in spreadsheet applications (Excel, Calc) by auto-sanitizing text fields starting with `=, +, -, @, \t, \r` with prepended single quotes (`'`) across all CSV export endpoints.
- **DuckDB OOM DoS & Memory Hardening (`VULN-04`)**: Prevents out-of-memory denial of service by strictly capping analytical query materialization via Polars `.head(max_rows)` before `.to_dicts()` conversion, and constraining the API `limit` parameter with Pydantic validation (`1 <= limit <= 1000`).
- **Ordered Parameter Binding & Identifier Escaping (`VULN-05`)**: DuckDB parameterized query parsing uses an ordered, single-pass scanner that preserves exact lexical parameter appearance order, supports repeated placeholders (`{{ x }} + {{ x }}`) without parameter count mismatch, and escapes double-quote identifiers in column profiling.
- **Watchdog Thread Termination & Resource Bounds (`VULN-06`)**: Python analyst execution threads are monitored by an active watchdog timer; runaway or spinning threads are asynchronously terminated via `PyThreadState_SetAsyncExc(tid, SystemExit)`. Sandbox request schemas strictly cap code payloads to 50,000 characters and execution timeouts to 1–30 seconds.
- **Fail-Closed WebSocket Authentication (`VULN-07`)**: Protocol-level WebSocket authentication executes seamlessly in public free-to-use mode and immediately fails closed (status 4001) if explicit authentication is required.
- **Trusted Proxy Rate Limit Evasion Defense (`VULN-08`)**: The global rate limiter extracts real client IPs while validating that proxy headers (`CF-Connecting-IP`, `X-Forwarded-For`) are only honored when incoming connections originate from verified internal/loopback proxies (Render, Docker, or Cloudflare edge), preventing direct IP spoofing.
- **CRLF & Header Injection Defense (CWE-113) (`VULN-09`)**: Request ID middleware validates all incoming `X-Request-ID` headers against a strict whitelist regex (`^[a-zA-Z0-9_-]{1,64}$`), discarding any newline (`\r`, `\n`) or delimiter injection payloads.
- **DuckDB Native Sandbox**: Native C++ external access disabled (`SET enable_external_access = false;`), preventing local file inclusion (LFI), network calls, or unauthorized file reads.

### 10. Direct Snowflake Cloud Warehouse Ingestion (Beta)
- **Dedicated Read-Only Enforcement**: Pure read-only extract execution (`GRANT SELECT` role validation), guarded by strict AST query verification preventing any mutation, DDL, or data alteration statements.
- **Dual Enterprise Authentication**: Full support for both encrypted/unencrypted PKCS#8 RSA Key-Pair authentication (`.pem`, `.key` with passphrase) and standard database user credentials.
- **In-Memory Encrypted Session Vault**: Ephemeral storage of warehouse credentials using HKDF key derivation and AES-256-GCM authenticated encryption. Sensitive private keys and passwords never touch disk storage or server logs, expiring automatically via configurable TTL.
- **Live Warehouse Object Explorer**: Real-time hierarchy traversal of databases, schemas, tables, and views with metadata reflection (row estimates, storage bytes, and Arrow data types).
- **Out-of-Core Streaming Ingestion**: Directly streams query results into high-performance Apache Arrow batches and compressed Parquet chunks, bounded by cooperative query cancellation, configurable row ceilings, and Bernoulli table sampling.

### 11. Modern Frontend & Executive Obsidian Dark Architecture
GetReport pairs its backend engine with an agency-grade, accessible user interface:
- **Obsidian Dark & Linear-Grade Aesthetic**: Built on high-contrast, calm dark architecture with deep slate backdrops (`#07080a`, `#0c0d12`, `#121215`), micro-dot technical grids, and zero-distraction ambient surfaces.
- **Anti-Slop Utilitarian Philosophy**: Completely purged of sparkles, decorative AI gimmick shapes, and shifting hue-rotations. Pure focus on high information density, sharp contrast, and executive clarity.
- **BeUI Animated Interaction System**: Interactive elements upgraded with BeUI motion components for fluid state transitions, tactile press physics, and keyboard accessibility.
- **Respectable Micro-Badge Architecture**: Understated, razor-sharp graphite micro-badges (such as the refined `Beta` indicator) providing clear status without visual clutter.
- **Double-Bezel Hardware Architecture**: Concentric dual-layer frosted glass enclosures featured on both the floating desktop navigation header and the mobile bottom action bar.
- **Ergonomic Semantic Button System**: Custom 12px squircle geometry (`rounded-xl`) with specialized semantic states and tactile active press feedback (`active:scale-95`).
- **Complete Suite of 13 Dedicated Pages**: Fully routed single-page architecture covering `Index`, `Workspace`, `Features`, `HowItWorks`, `Pricing`, `Examples`, `Documentation`, `Dashboard`, `Analytics` (BI Studio), `Contact`, `PrivacyPolicy`, `TermsOfService`, and `NotFound`.

---

## Tech Stack

### Frontend
| Component | Technology |
|---|---|
| **Framework** | React 18 + Vite |
| **Language** | TypeScript 5.8 |
| **UI Components** | BeUI Animated Components, Shadcn/UI primitives, Radix UI |
| **Data Visualization & BI** | Apache ECharts 6.1 (`echarts-for-react`), Recharts |
| **Typography** | Plus Jakarta Sans, Mont, JetBrains Mono |
| **Styling** | Tailwind CSS 3.4, Vanilla CSS |
| **Design Architecture** | Obsidian Executive Dark, Linear-grade Micro-Badges, Double-Bezel Frosted Glass |
| **Sanitization** | DOMPurify |
| **Motion & Icons** | Framer Motion, Lucide React |
| **State & Data** | TanStack Query v5 |
| **Routing** | React Router v6 (with `/bi` route alias) |

### Backend
| Component | Technology |
|---|---|
| **Framework** | FastAPI (Python 3.12+) with Pydantic v2 |
| **OLAP Engine** | DuckDB 1.0+ (In-Process, Apache Arrow Zero-Copy) |
| **Warehouse** | Snowflake Connector (`snowflake-connector-python` with PyArrow engine) |
| **Session Security** | In-Memory HKDF + AES-256-GCM Credential Vault |
| **Data & ML Engines** | Polars (Rust Core), DuckDB, NumPy, SciPy, Scikit-Learn |
| **AI Agent** | Google Antigravity SDK (`google-antigravity`) |
| **LLM Providers** | Google Gemini (2.5/3.7 Flash), OpenRouter, OpenAI |
| **Database** | Neon Serverless PostgreSQL / SQLite (Local) |
| **Contracts** | Great Expectations (GX) |
| **Task Queue** | ARQ / Celery + Redis |
| **PDF Engines** | Typst (Ultra-fast, <25MB RAM Rust Engine) / WeasyPrint / ReportLab |
| **Storage** | Sandboxed Local Disk / PostgreSQL BYTEA / AWS S3 |

---

## API Reference (Key Endpoints)

### Ingestion & Jobs
- `POST /api/upload`: Single-pass streaming ingestion with magic byte signature verification.
- `POST /api/upload/batch`: Multi-file ingestion under a unified batch identifier.
- `POST /api/upload/join`: Multi-dataset relational joins (`inner`, `left`, `full`, `anti`, `semi`).
- `GET /api/status/{task_id}`: Polling endpoint for job progress and structured results.
- `WS /api/ws/status/{task_id}`: Real-time WebSocket stream with 15s heartbeats and Redis PubSub.

### Business Intelligence & Analytics
- `GET /api/analytics/datasets`: Retrieves all available analyzed datasets eligible for interactive exploration.
- `POST /api/analytics/query`: Executes high-speed grouped aggregations, generating ECharts-ready option payloads, KPI summaries, and tabular records.
- `GET /api/analytics/schema/{task_id}`: Inspects dataset column types, categorical attributes, and numerical measures for BI builder interfaces.

### Snowflake Warehouse (Beta)
- `POST /api/snowflake/connect`: Validates credentials and initializes an encrypted in-memory session.
- `GET /api/snowflake/databases`: Lists accessible databases for the authenticated session.
- `GET /api/snowflake/schemas`: Lists schemas in a given database (`?database=NAME`).
- `GET /api/snowflake/tables`: Lists tables and views with metadata (`?database=NAME&schema=NAME`).
- `POST /api/snowflake/preview`: Previews the top 100 rows with Arrow data types and null counts.
- `POST /api/snowflake/import`: Queues asynchronous streaming extraction into an analytical parquet dataset.
- `POST /api/snowflake/cancel/{task_id}`: Cooperatively cancels an active extraction query.
- `DELETE /api/snowflake/session`: Immediately purges session credentials from the memory vault.
- `DELETE /api/snowflake/extract/{task_id}`: Safely cleans up a completed warehouse extract file.

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
- `POST /api/auth/stream-ticket`: Issues a single-use, 60-second cryptographically secure ticket for SSE/WebSocket streams.

---

## Getting Started

### Prerequisites
- **Python 3.12+**
- **Node.js 18+ & npm**
- **Redis** (optional for local development, required for Celery / ARQ task queuing)

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

Open your browser at `http://localhost:5173` (or `http://localhost:8080`).

---

## Verification & Testing

GetReport maintains an exhaustive automated test suite verifying mathematical accuracy, security constraints, and UI compilation:

```bash
cd Backend
pytest -q
```

```
================== 446 passed in 74.15s (100%) ==================
```

### Test Suite Highlights
- **15 Snowflake Warehouse Integration Tests** (`test_snowflake_integration.py`): Validates account identifier format sanitization, double-quote identifier SQL safety, Arrow schema compilation, batch normalization, PKCS#8 RSA private key DER serialization, HKDF/AES-256 encrypted session vault lifecycle, driver metadata discovery, streaming parquet extraction, cooperative query cancellation, and session teardown.
- **22 Row Intelligence & Machine Learning Tests** (`test_row_intelligence.py`): Validates 4-tier triage exclusivity, bit-identical immutability on clean records, safe non-data row pruning, gated imputation on sparse rows, and self-supervised multi-column anomaly detection (`HistGradientBoostingRegressor`) with contrastive driver explanations.
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
- **Frontend Production Build & Typecheck**: Verified with `npm run build` and `tsc --noEmit` (0 TypeScript errors, clean bundle compilation, React Doctor score 96/100).

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
| `PDF_ENGINE` | `typst` | `typst` | PDF engine (`typst` ultra-fast Rust engine <25MB RAM, or `reportlab` fallback). |
| `DATABASE_URL` | (empty) -> SQLite | `postgresql://...` | Neon Serverless PostgreSQL connection string. |
| `REDIS_URL` | `redis://localhost:6379/0` | `redis://...` | Redis broker for background worker and WebSocket PubSub. |
| `STORAGE_TYPE` | `local` | `db` / `s3` | File storage provider (`local`, `db` for Neon BYTEA, or `s3`). |
| `API_KEY` | (empty) | (empty/optional) | Optional `X-API-Key` authentication (leave empty for public access). |
| `CORS_ORIGINS` | `http://localhost:5173` | `https://get-report.vercel.app` | Allowed CORS origins. |
| `GEMINI_API_KEY` | (optional) | (recommended) | API Key for Google Antigravity Agent and vector embeddings. |
| `MAX_UPLOAD_SIZE_MB` | `50` | `50` | Maximum file upload limit in megabytes. |
| `RATE_LIMIT_ENABLED` | `True` | `True` | Global rate limiter toggle. |

---

## License
MIT License. See [LICENSE](LICENSE) for details.
