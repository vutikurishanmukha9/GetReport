# GetReport: Complete Codebase Architectural, Technological, and Algorithmic Dossier

---

## 1. Executive Summary & Purpose of GetReport

### What is GetReport?
**GetReport** is an enterprise-grade automated data intelligence, forensic data quality auditing, exploratory data analysis (EDA), and interactive Business Intelligence (BI) platform. It is engineered to transform raw, messy, heterogeneous tabular datasets (CSV, TSV, Excel, Parquet, JSON, JSONL, Feather, Arrow) or direct cloud data warehouse extracts (Snowflake) into publication-ready executive dossiers (PDF, HTML), machine-learning-ready feature pipelines, Great Expectations data contracts, interactive visual exploration dashboards (Apache ECharts / Superset), and grounded conversational intelligence interfaces (LightRAG + Google Antigravity Agent).

### Core Mission & Problems Solved
Traditional data analysis pipelines suffer from fragmented tooling, high memory overhead, synthetic data hallucination during cleaning, vulnerable execution boundaries, and superficial statistical metrics. GetReport solves these challenges through:

1. **Elimination of Data Distortion & Hallucination**:
   - Pruning non-data rows (headers, totals, sub-totals, footers) before running statistical profiling.
   - 4-Tier Row Triage (`PROTECT`, `REVIEW`, `FIX`, `EXCLUDE`) ensuring bit-identical preservation of clean records.
   - Gated smart imputation preventing synthetic record fabrication when missingness is high.
2. **Zero-Copy Hybrid Analytical Performance**:
   - Polars (Rust core) vectorized streaming engine combined with DuckDB in-process OLAP querying directly over shared Apache Arrow memory batches.
3. **Forensic Integrity & Audit-Grade Provenance**:
   - First-digit Benford's Law Chi-Square distribution tests to detect fraudulent, fabricated, or synthetic numbers.
   - Population Stability Index (PSI) to detect distribution drift across dataset runs.
   - Cryptographic DAG (Directed Acyclic Graph) lineage tracking every dataset transformation.
4. **Interactive Exploration & Embedded BI**:
   - Dynamic multi-dimensional aggregation canvas powered by Apache ECharts 6.1 and Apache Superset companion integration.
   - High-dimensional visual discovery via Facebook Research HiPlot parallel coordinates.
5. **Zero-Vulnerability Security Architecture**:
   - Defense-in-depth across every boundary: AST sandboxing, C++ native filesystem locks, CSV formula neutralization (CWE-1236), XSS auto-escaping, and strict tenant isolation.

---

## 2. High-Level System Architecture & Topology

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 INGESTION & STAGING LAYER                                │
│                                                                                          │
│   ┌─────────────────────┐    ┌─────────────────────┐    ┌────────────────────────────┐   │
│   │   Raw File Upload   │    │  Multi-File Joins   │    │ Snowflake Cloud Warehouse  │   │
│   │ CSV/XLSX/Parquet/ND │    │ Inner/Left/Outer    │    │ Key-Pair Auth + AES Vault  │   │
│   └──────────┬──────────┘    └──────────┬──────────┘    └─────────────┬──────────────┘   │
└──────────────┼──────────────────────────┼─────────────────────────────┼──────────────────┘
               │                          │                             │
               ▼                          ▼                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                      POLARS VECTORIZED STREAMING ENGINE (RUST CORE)                      │
│                                                                                          │
│  • Magic Byte Signature Verification        • Single-Pass Stream Ingestion (Zero-Copy)   │
│  • Memory-Safe Chunking & Type Coercion     • SHA-256 Deduplication Hash Computation     │
└─────────────────────────────────────────┬────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                       PHASE 1: INSPECTION & ISSUE LEDGER TRIAGE                          │
│                                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │    Row Intelligence     │  │   Text Intelligence     │  │   Issue Ledger ("Jira")  │  │
│  │ • 4-Tier Row Triage     │  │ • Shape Clustering      │  │ • Automated Detectors    │  │
│  │ • Non-Data Pruning      │  │ • Mojibake / Case Normal│  │ • Approve/Reject/Modify  │  │
│  │ • Cross-Column OOF ML   │  │ • Semantic Conflict Res │  │ • Restricted Python Fix  │  │
│  └─────────────────────────┘  └─────────────────────────┘  └──────────────────────────┘  │
│                                         │                                                │
│                                         ▼ (Status: WAITING_FOR_USER)                     │
└─────────────────────────────────────────┼────────────────────────────────────────────────┘
                                          │
                         [User Approves / Configures Rules]
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                     PHASE 2: FULL ANALYTICAL & ML INTELLIGENCE PIPELINE                  │
│                                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌──────────────────────────┐  │
│  │  Statistical Profiling  │  │  Machine Learning Core  │  │   Smart Schema & Graph   │  │
│  │ • Non-Parametric Disp.  │  │ • Supervised AutoML Tour│  │ • Symbolic Equations     │  │
│  │ • Non-Linear Phik (ΦK)  │  │ • Unsupervised K-Means  │  │ • Functional Dependencies│  │
│  │ • Benford's Law Chi-Sq  │  │ • PCA 2D Projections    │  │ • LightRAG Knowledge Graph│ │
│  │ • Mann-Kendall / STL    │  │ • Contrastive Personas  │  │ • Trigram Fuzzy Search   │  │
│  └─────────────────────────┘  └─────────────────────────┘  └──────────────────────────┘  │
└───────────────────┬─────────────────────────────────────────────────┬────────────────────┘
                    │                                                 │
                    ▼                                                 ▼
┌──────────────────────────────────────────────┐  ┌────────────────────────────────────────┐
│     IN-PROCESS DUCKDB OLAP & SQL ENGINE      │  │    CONVERSATIONAL AGENT & SANDBOX      │
│                                              │  │                                        │
│ • Hardware Zero-Copy over Apache Arrow       │  │ • Google Antigravity Agent (Gemini)    │
│ • Native C++ Filesystem Block (`enable_ext`) │  │ • Dual-Level Local/Global RAG          │
│ • Golden Query KPI Store & SQL Introspect    │  │ • AST-Sandboxed Python Analyst Agent   │
│ • Parameter Order Scanner & Quoting Defenses │  │ • Matplotlib Headless Base64 Plots     │
└───────────────────┬──────────────────────────┘  └───────────────────┬────────────────────┘
                    │                                                 │
                    └────────────────────────┬────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                               DELIVERABLES & EXPORT ENGINES                              │
│                                                                                          │
│   ┌─────────────────────┐    ┌─────────────────────┐    ┌────────────────────────────┐   │
│   │ Executive PDF Report│    │  Great Expectations │    │ Cleaned Datasets & BI      │   │
│   │ Rust Typst Engine   │    │  Data Contract Suite│    │ CSV (CWE-1236 safe),       │   │
│   │ <25MB RAM Overhead  │    │  JSON & Python Test │    │ Parquet, ECharts Options   │   │
│   └─────────────────────┘    └─────────────────────┘    └────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Complete Technology Stack Matrix

| Layer / Subsystem | Primary Technology | Exact Versions / Specifications | Architectural Role & Implementation Details |
|:---|:---|:---|:---|
| **API Web Framework** | **FastAPI** | `>=0.110.0` | Asynchronous ASGI REST API framework; dependency-injection architecture; lifespan async context managers. |
| **ASGI Servers** | **Uvicorn** / **Granian** | Uvicorn `>=0.29.0`, Granian `>=1.6.0` | High-throughput async ASGI HTTP/1.1, HTTP/2, and WebSocket server runtimes. |
| **Data Validation** | **Pydantic** & **Pydantic Settings** | `pydantic-settings>=2.2.1` | Strict data schema validation, settings management, and type casting for API payloads. |
| **Columnar Streaming Engine** | **Polars (Rust)** | `>=1.2.1` | SIMD-vectorized columnar data processing; streaming execution; lazy frame query plans; zero-copy Apache Arrow memory layout. |
| **In-Process OLAP Engine** | **DuckDB** | `>=1.0.0` | In-process SQL OLAP engine; hardware zero-copy Arrow memory mapping; C++ level security boundary (`SET enable_external_access = false;`). |
| **Arrow Memory Standard** | **PyArrow** | `>=14.0.0` | In-memory columnar data interchange format between Polars, DuckDB, Snowflake, and disk serializers. |
| **Spreadsheet I/O** | **FastExcel**, **OpenPyXL**, **XlsxWriter** | `fastexcel>=0.10.0`, `openpyxl>=3.1.2`, `xlsxwriter>=3.2.0` | High-performance C-based Excel parsing (FastExcel), multi-sheet loading, and streaming Excel report generation. |
| **Scientific Computing & Stats** | **NumPy**, **SciPy** | `numpy>=1.26.4`, `scipy>=1.12.0` | Numerical linear algebra, non-parametric statistics, probability distributions, Chi-Square tests, and optimization. |
| **Machine Learning Core** | **Scikit-Learn** | `>=1.4.0` | HistGradientBoostingRegressor, IsolationForest, LocalOutlierFactor, KMeans, PCA, Empirical Bayes target encoding. |
| **Network & Graph Topology** | **NetworkX** | `>=3.0` | MultiDiGraph store for dataset semantic topologies, community clustering, DAG provenance, shortest path, and blast radius traversal. |
| **Cloud Data Warehouse** | **Snowflake Connector Python** | `>=3.7.0` (with PyArrow extra) | Enterprise cloud warehouse connectivity; Arrow batch streaming; RSA Key-Pair & password authentication. |
| **Cryptography & Session Vault** | **Cryptography** (PyCA) | `>=42.0.0` | HKDF (HMAC-based Extract-and-Expand Key Derivation) and AES-256-GCM authenticated encryption for credential vault. |
| **SQL Parser & Transpiler** | **SQLGlot** | `>=23.0.0` | SQL dialect parsing, AST lineage extraction, CTE dependency resolution, and table/view discovery. |
| **Headless Visualization** | **Matplotlib** | `>=3.8.3` | Headless plot rendering via the `Agg` non-interactive backend; direct rasterization to high-DPI base64 PNG. |
| **High-Speed PDF Engine** | **Typst (Rust)** | `typst>=0.15.0` | Ultra-fast native Rust PDF compilation engine; consumes <25MB RAM under load; compiles from parameterized `.typ` templates. |
| **Fallback PDF Engines** | **ReportLab**, **WeasyPrint**, **Jinja2** | `reportlab>=4.1.0`, `jinja2>=3.1.3` | Pixel-precise ReportLab canvas engine and HTML/CSS Paged Media rendering with strict SSRF-safe URL fetching. |
| **AI Agent Runtimes** | **Google Antigravity SDK**, **Google GenAI** | `google-antigravity>=0.1.15`, `google-genai>=1.0.0`, `openai>=1.14.0` | Multi-turn agent orchestration, Gemini 2.5/3.7 Flash models, structured tool calling, and grounded RAG synthesis. |
| **Task Queue & Async Workers** | **ARQ** / **Redis** / **In-Memory** | `arq>=0.26.1`, `redis>=5.0.3` | Lean Redis-backed async worker queue (~25MB RAM overhead) with automatic fallback to native Python asyncio in local dev. |
| **Relational & Vector Storage** | **Neon PostgreSQL** / **SQLite** | `psycopg2-binary>=2.9.9`, `asyncpg`, `aiosqlite` | Dual-mode storage: Serverless PostgreSQL with PGVector (HNSW/GIN indexes) in production; SQLite with WAL mode in development. |
| **Rate Limiting & Security** | **SlowAPI** | `>=0.1.9` | Token-bucket sliding-window rate limiting; proxy-aware real IP resolution against spoofing; CORS-preserved 429 handlers. |
| **Frontend Framework & Tooling** | **React 18** + **Vite 5** | `react@18.3.1`, `vite@5.4.19`, `@vitejs/plugin-react-swc` | Modern single-page web architecture; fast SWC compiler; ESM bundling; route-based code splitting. |
| **Frontend Language** | **TypeScript 5.8** | `typescript@5.8.3` | Strict type safety; zero `any` policy in core business modules; comprehensive domain interfaces. |
| **UI Primitives & Motion** | **Radix UI**, **BeUI**, **Tailwind CSS**, **Framer Motion** | `tailwindcss@3.4.17`, `framer-motion@12.29.2`, `@radix-ui/react-*` | Accessible headless primitives; BeUI tactile press physics; dark obsidian executive aesthetic. |
| **Interactive BI & Charting** | **Apache ECharts 6.1**, **Recharts** | `echarts@6.1.0`, `echarts-for-react@3.0.6`, `recharts@2.15.4` | Hardware-accelerated canvas/SVG chart rendering; multi-dimensional grouped and stacked series; interactive brush selection. |
| **Client State & Cache** | **TanStack Query v5** | `@tanstack/react-query@5.83.0` | Asynchronous query caching; stale-while-revalidate; query deduplication; optimistic updates. |
| **Frontend Linter & Anti-Slop** | **Oxlint** + **ESLint** | `oxlint@1.78.0`, `@oxlint/plugins` | High-speed Rust linter with 15 custom anti-slop rules guarding against type assertion chaining, object mutations, and code rot. |
| **Testing Runtimes** | **Pytest** + **Hypothesis** + **Vitest** | `pytest>=8.0.0`, `hypothesis>=6.151.6`, `vitest@3.2.4` | Property-based testing, adversarial boundary verification, async API tests, and component testing. |

---

## 4. Comprehensive Inventory of Components & Modules

### 4.1 Backend Core Layer (`Backend/app/core/`)
- **`auth.py`**:
  - Enforces API key verification (`X-API-Key` header) with timing-attack-safe `secrets.compare_digest`.
  - Implements **Ephemeral Stream Tickets** (`create_stream_ticket`, `consume_stream_ticket`): issues 60-second cryptographically random single-use tokens for EventSource (SSE) and WebSocket URLs, preventing API key exposure in HTTP access logs (CWE-598).
  - Validates UUID formatting via regex (`UUID_PATTERN`) to prevent path and parameter injection.
  - Enforces strict multi-tenant boundary checks (`check_task_ownership`, `get_authenticated_owner_id`) preventing Broken Object Level Authorization (BOLA / CWE-285).
- **`file_validation.py`**:
  - Validates uploaded file contents using Magic Numbers (binary signatures) rather than extensions:
    - Office Open XML (`.xlsx`): `PK\x03\x04`
    - OLE2 Compound Document (`.xls`): `\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1`
    - Apache Parquet (`.parquet`): `PAR1`
    - Apache Arrow / Feather (`.feather`, `.arrow`): `ARROW1`
    - GZIP (`.gz`): `\x1F\x8B`
    - CSV (`.csv`): Decodes multi-encoding text (UTF-8, UTF-16, Latin-1) and blocks binary null bytes (`\x00`).
- **`limiter.py`**:
  - Proxy-aware IP resolver (`_get_real_client_ip`): Only honors reverse proxy headers (`CF-Connecting-IP`, `X-Forwarded-For`) if the immediate peer IP is verified loopback or private RFC-1918 space, defeating rate limit evasion.
  - Configures tiered Token-Bucket limits: Uploads (10/min), Analysis (30/min), Chat (60/min), Status (300/min), Reports (20/min).
- **`security_headers.py`**:
  - Pure ASGI middleware injecting defensive headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection: 1; mode=block`, `Strict-Transport-Security: max-age=63072000`, `Content-Security-Policy`, and masking server identity.
- **`request_id.py`**:
  - Pure ASGI middleware generating and validating `X-Request-ID` against strict whitelist regex `^[a-zA-Z0-9_-]{1,64}$` to eliminate CRLF and HTTP Header Injection (CWE-113).
- **`task_dispatcher.py`**:
  - Lean task orchestrator supporting dual execution: ARQ with Redis in production; native Python `asyncio` in development.
- **`factories/`**:
  - `registry.py`: Generic, type-safe `Registry[T]` supporting decorator and programmatic registration with test-reset isolation.
  - `storage.py`: Storage provider factory providing `LocalStorageProvider`, `DatabaseStorageProvider` (Postgres BYTEA), and `S3StorageProvider`.
  - `exporters.py`: Template Method data exporters (`CsvDataExporter`, `ExcelDataExporter`, `ParquetDataExporter`, `JsonDataExporter`) with universal CWE-1236 formula neutralization.
  - `reports.py`: Factory chaining fallback PDF engines (`TypstReportEngine` -> `WeasyPrintReportEngine` -> `ReportLabReportEngine`).
  - `remediation.py`: Functional strategy pattern mapping issue actions (`drop_rows`, `fill_mean`, `fill_median`, `fill_mode`, `fill_value`, `cap_outliers`, `log1p`) to pure Polars DataFrame transforms.

### 4.2 Backend API Routers (`Backend/app/api/routes/`)
- **`upload.py`**:
  - Single-pass zero-copy streaming upload calculating SHA-256 hash, enforcing file size limits, validating magic bytes, and dispatching Phase 1 inspection.
  - Batch upload endpoint (`/upload/batch`) bounded to 10 files with aggregate size enforcement.
  - Pre-flight header inspection (`/upload/inspect-headers`) for intelligent frontend column mapping.
  - Multi-dataset join endpoint (`/upload/join`) executing Polars joins (inner, left, outer, semi, anti) across up to 5 files with key mapping, preventing Cartesian cross-join DoS.
- **`status.py`**:
  - Polling endpoint (`/status/{task_id}`) returning job progress, stage, error, and download links.
  - Server-Sent Events (SSE) stream (`/status/{task_id}/stream`) emitting real-time `progress`, `waiting`, `complete`, and `error` events.
  - WebSocket endpoint (`/ws/status/{task_id}`) with heartbeat ping/pong and authentication.
- **`report.py`**:
  - PDF generation trigger and status polling (`/jobs/{task_id}/report`, `/report/status`).
  - Multi-format data export (`/jobs/{task_id}/export/{format}`): CSV, Parquet, Excel, HTML.
  - Transformation DAG inspection, node traversal, and audit log export (JSON/CSV).
  - In-process DuckDB SQL execution (`/jobs/{task_id}/query-sql`) with parameter order parsing, row limits, and C++ native sandbox.
  - Great Expectations contract export (`/jobs/{task_id}/export-gx`).
  - Golden Query repository management (create, list, delete verified SQL KPIs).
  - AST-sandboxed Python analyst runtime (`/jobs/{task_id}/sandbox-exec`).
  - Virtual Concept Synthesizer (`/jobs/{task_id}/concepts/derive`).
- **`chat.py`**:
  - RAG conversational Q&A (`/jobs/{task_id}/chat`) and SSE token streaming (`/jobs/{task_id}/chat/stream`).
- **`issues.py`**:
  - Issue Ledger management: list issues, approve, reject, modify fix code, approve all, reject all, and lock ledger.
- **`snowflake.py`**:
  - Cloud warehouse connection (`/snowflake/connect`), metadata exploration (`/databases`, `/schemas`, `/tables`), preview, profile, and out-of-core streaming Parquet import (`/snowflake/import`).
- **`graph.py`**:
  - Knowledge graph extraction (`/jobs/{task_id}/graph`), interactive Vis.js HTML rendering, shortest causal path, blast radius impact analysis, trigram fuzzy entity search, and SQL lineage ingestion (`/sql-lineage`).
- **`analytics.py`**:
  - Superset companion health check, dataset synchronization, guest token minting, schema introspection, grouped metric aggregation, and Superset Explore query execution.

### 4.3 Backend Domain Services (`Backend/app/services/`)
- **`data_processing.py`**:
  - Streaming DataFrame loading via Polars with auto-delimiter and encoding detection.
  - Zip bomb & Gzip bomb validation inspecting uncompressed byte ratios.
  - MICE (Multivariate Imputation by Chained Equations) using iterative Bayesian Ridge regression.
  - Phone number E.164 normalization.
  - Dynamic missingness suggestion computation.
- **`row_intelligence.py`**:
  - 4-Tier Row Triage (`PROTECT`, `REVIEW`, `FIX`, `EXCLUDE`).
  - Pre-profiling pruning of non-data rows (headers, totals, footers).
  - Out-of-fold `HistGradientBoostingRegressor` for self-supervised cross-column relationship anomaly detection.
  - Residual scaling via Median Absolute Deviation ($\text{MAD} \times 1.4826$).
- **`text_intelligence.py`**:
  - Character shape clustering (`AAAA-9999`, etc.), casing inconsistency, and mojibake resolution.
  - Out-of-fold probabilistic classifier detecting conflicting category labels.
- **`confidence_scoring.py`**:
  - Benford's Law logarithmic first-digit audit with Chi-Square goodness-of-fit.
  - 4-Pillar Role-Adaptive Confidence Scoring (Completeness, Consistency, Validity, Stability) mapping to letter grades (A-F).
- **`analysis/`**:
  - `statistics.py`: Non-parametric dispersion (MAD, IQR, 5% Trimmed Mean, CV, Bowley skewness, Pearson moments).
  - `correlations.py`: Non-linear mixed-type $\Phi_K$ (Phik) correlation matrix with Chi-Square independence test and degrees of freedom correction.
  - `outliers.py`: Tukey's fences, Hubert-Vandervieren skewness-adjusted fences, Isolation Forest, Local Outlier Factor (LOF), and STL residual outliers.
  - `time_series.py`: Mann-Kendall monotonic trend test, Sen's slope estimator, STL decomposition (additive/multiplicative), and seasonality detection.
  - `missing.py`: MCAR/MAR/MNAR classification, pairwise $\phi$ missingness correlation, and listwise deletion survival prediction.
  - `ml_readiness.py`: Composite ML readiness index (0-100) scoring null rates, constant columns, type purity, outlier burden, and target class imbalance.
- **`supervised/` & `supervised_intelligence.py`**:
  - Automated target detection, preprocessor matrix construction, and AutoML tournament across HistGradientBoosting, LightGBM, Random Forest, Ridge, and Logistic Regression with key driver feature importances.
- **`unsupervised/` & `unsupervised_intelligence.py`**:
  - Optimal K-Means clustering with Silhouette, Davies-Bouldin, and Calinski-Harabasz evaluation, PCA 2D dimensionality reduction, and Contrastive Persona profile synthesis.
- **`timeseries_kats.py`**:
  - Meta Kats-inspired time series intelligence: CUSUM changepoint detection, Holt-Winters exponential smoothing, and comprehensive time series feature extraction.
- **`duckdb_engine.py`**:
  - Zero-copy DuckDB analytical engine with C++ native sandbox, single-pass parameter ordering scanner, and double-quote identifier escaping.
- **`sandboxed_analyst.py`**:
  - AST-parsed Python/Polars analyst execution environment with forbidden import blocking, dunder reflection mitigation, watchdog thread timeout termination via `PyThreadState_SetAsyncExc`, and headless Matplotlib base64 plot capture.
- **`concept_synthesizer.py`**:
  - Natural language and formula parser compiling expressions to optimized Polars expression ASTs under strict root whitelisting (`ALLOWED_POLARS_ROOTS = {"pl", "col", "lit", "when"}`).
- **`transformation_dag.py`**:
  - Cryptographically hashed mutation DAG tracking every dataset transformation, parent/child linkages, durations, and reversibility hints.
- **`dataset_versioning.py`**:
  - Population Stability Index (PSI) and schema drift tracking across dataset versions.
- **`smart_schema.py`**:
  - Symbolic equation solver discovering multivariate linear relations ($Y = aX_1 + bX_2 + c$), functional dependency mining ($X \to Y$), and timestamp/Excel date detection.
- **`pii_guard.py`**:
  - Sensitive data pattern detection (emails, SSNs, credit cards with Luhn checksum validation, phone numbers, IP addresses, API keys).
- **`rag_service.py` & `core/rag_utils.py`**:
  - Dual-level knowledge graph RAG, table-aware chunking, vector stores (PGVector HNSW, SimpleVectorStore cosine, TF-IDF fallback), Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
- **`report_generator.py`, `report_typst.py`, `report_weasyprint.py`, `report_sections.py`**:
  - Enterprise report compilation across Typst, WeasyPrint, and ReportLab.

### 4.4 Frontend Pages & Application Architecture (`Frontend/src/`)
- **Pages**:
  - `Index.tsx`: High-impact landing page featuring linear-grade dark obsidian aesthetic, live metrics trust bar, process pipeline visualization, comparison tables, and quick-launch docks.
  - `Workspace.tsx`: Multi-stage interactive data workspace (`upload` -> `preview` -> `generating` -> `complete`) with real-time SSE progress tracking, data preview, cleaning rule customizer, and grounded chat interface.
  - `Analytics.tsx`: Superset-grade Business Intelligence Studio providing grouped aggregations, dimension breakdowns, metric calculations, and interactive Apache ECharts visualizations.
  - `Dashboard.tsx`: Persistent audit history dashboard with search, filtering, quality letter grades, and direct report download actions.
  - `Documentation.tsx`: Full interactive API reference with interactive tabs, cURL/Python/TypeScript code snippets, and complete request/response payloads.
  - `Features.tsx`: Exhaustive technical breakdown of all 11 core engine pillars.
  - `HowItWorks.tsx`: Step-by-step interactive walkthrough of the 4-phase forensic audit lifecycle.
  - `Pricing.tsx`: Transparent pricing tiers (Free/Open Source, Pro, Enterprise) with feature matrices.
  - `Examples.tsx`: Curated showcase datasets (Financial Fraud Ledger, Healthcare EHR Records, Retail E-Commerce Sales) with pre-computed audits.
  - `Contact.tsx`, `PrivacyPolicy.tsx`, `TermsOfService.tsx`, `NotFound.tsx`: Production legal, compliance, contact, and 404 routing.
- **Core Components**:
  - `SupersetExploreStudio.tsx`: Full-featured BI visual explorer with dynamic dimension dropdowns, metric aggregation selectors, secondary split dimensions, top-N limiters, and responsive ECharts canvas rendering.
  - `HiPlotParallelCoordinates.tsx`: Multi-parameter parallel coordinate discovery canvas with interactive categorical/continuous axis brushing.
  - `SnowflakeConnectModal.tsx`: Secure modal for RSA Key-Pair and password authentication, warehouse browsing, and streaming ingestion.
  - `IssueLedger.tsx`: Interactive issue management interface ("Jira for Dirty Data") allowing row-by-row approve/reject/modify actions.
  - `DataHealthCheck.tsx`: Executive KPI cards displaying overall quality scores, letter grades, pillar breakdowns, and Benford's Law status.
  - `ChatInterface.tsx`: Conversational AI interface supporting streaming markdown answers, source footnote citations, follow-up suggestions, and embedded chart cards.
  - `ReportGeneration.tsx`: Tabbed dossier viewer presenting executive summaries, statistical tables, correlation heatmaps, outlier distributions, and DAG lineage graphs.
- **Anti-Slop Oxlint Architecture (`Frontend/tools/oxlint/anti-slop/`)**:
  - 15 opinionated custom linter rules enforcing strict TypeScript discipline:
    - `no-chained-type-assertions`: Bans `as unknown as Type`.
    - `no-conditional-empty-object-spread`: Prevents spreading conditional empty objects that widen types.
    - `no-known-value-widening`: Disallows widening known literal types to primitives.
    - `no-module-mocking`: Enforces pure dependency injection over global module monkey-patching in tests.
    - `no-object-parameters`: Requires explicit interfaces over loose anonymous object parameters.
    - `no-reflect-apply`, `no-reflect-get`: Enforces compile-time property access over reflection.
    - `no-runtime-typeof`: Restricts runtime typeof checks where TypeScript union discrimination applies.
    - `no-shape-in-symbol-names`: Prevents encoding type shape names in identifier symbols.
    - `no-unknown-parameters`, `no-unknown-returns`, `no-unknown-type-aliases`: Enforces zero `unknown`/`any` leaks.
    - `no-unsafe-dictionary-type`: Flags unchecked dictionary access without `undefined` guards.
    - `no-widen-then-assert`: Bans widening an expression to assign or re-assert it.
    - `require-safety-comment-for-type-assertion`: Mandates an explicit `// SAFETY:` explanatory comment above any `as` type assertion.

---

## 5. Exhaustive Catalogue of Mathematical Formulations, Heuristics & Algorithms

### 5.1 Statistical Profiling & Non-Parametric Dispersion

#### 1. Median Absolute Deviation (MAD)
To evaluate dispersion without distortion from extreme outliers, GetReport computes the Median Absolute Deviation:
$$\text{MAD} = \text{median}\left(|X_i - \text{median}(X)|\right)$$
For asymptotic consistency with standard deviation under a normal distribution, the scaled MAD is computed:
$$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{MAD}$$

#### 2. Interquartile Range (IQR) & 5% Trimmed Mean
- **IQR**: Computed as $IQR = Q_3 - Q_1$, where $Q_1$ and $Q_3$ are the 25th and 75th percentiles.
- **5% Trimmed Mean**: Arithmetic mean calculated after discarding the lowest 5% and highest 5% tails:
  $$\bar{X}_{\text{trimmed}} = \frac{1}{n - 2k} \sum_{i=k+1}^{n-k} X_{(i)} \quad \text{where } k = \lfloor 0.05n \rfloor$$

#### 3. Bowley Quartile Skewness
A non-parametric measure of distribution asymmetry based entirely on quartiles:
$$S_B = \frac{Q_3 + Q_1 - 2Q_2}{Q_3 - Q_1}$$
Where $S_B \in [-1, 1]$. Values close to 0 indicate symmetric distribution, while positive/negative values denote right/left tail skew.

#### 4. Pearson Moments & Robust Z-Scores
- **Pearson Sample Skewness**:
  $$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{s}\right)^3$$
- **Pearson Excess Kurtosis**:
  $$g_2 = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{X_i - \bar{X}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$
- **Modified Z-Score** (Boris Iglewicz and David Hoaglin):
  $$M_i = \frac{0.6745 \times (X_i - \text{median}(X))}{\text{MAD}}$$
  Where $|M_i| > 3.5$ is flagged as a potential outlier.

---

### 5.2 Non-Linear Association: The $\Phi_K$ (Phik) Correlation Algorithm

Standard Pearson correlation measures only linear relationships; Spearman and Kendall measure monotonic relationships. GetReport implements **$\Phi_K$ (Phik)**, a correlation coefficient based on refined Pearson Chi-Square test statistics that simultaneously handles:
1. Continuous vs Continuous
2. Categorical vs Categorical
3. Mixed Continuous vs Categorical (without one-hot encoding explosion)

#### Mathematical Formulation:
1. **Adaptive Discretization**: Continuous variables are dynamically binned into $B$ quantiles (default 10 bins) preserving sample density.
2. **2D Contingency Table**: For variables $X$ and $Y$, construct an $r \times c$ contingency matrix of observed counts $O_{ij}$.
3. **Chi-Square Calculation**:
   $$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \quad \text{where } E_{ij} = \frac{R_i C_j}{N}$$
4. **Degrees of Freedom Correction & Mapping**:
   $$\Phi_K = \sqrt{ \frac{\chi^2 / N}{\min(r-1, c-1)} }$$
   The value is mapped via an inverted bivariate Gaussian integral to produce an association metric equivalent to standard Pearson correlation $r \in [0, 1]$.

---

### 5.3 Forensic Auditing & Data Integrity Algorithms

#### 1. Benford's Law Chi-Square Forensic Audit
Natural numeric datasets (monetary transactions, populations, invoice amounts) exhibit a logarithmic first-digit frequency distribution:
$$P(d) = \log_{10}\left(1 + \frac{1}{d}\right) \quad \text{for } d \in \{1, 2, \dots, 9\}$$

GetReport tests empirical first digits against Benford's Law:
1. Extract leading non-zero digit $d \in \{1, \dots, 9\}$ from positive numeric columns.
2. Calculate observed counts $O_d$ and expected counts $E_d = N \times P(d)$.
3. Compute Pearson Chi-Square Goodness-of-Fit statistic:
   $$\chi^2 = \sum_{d=1}^9 \frac{(O_d - E_d)^2}{E_d}$$
4. Evaluate with 8 degrees of freedom ($\text{df} = 9 - 1 = 8$).
   - Critical value at $\alpha = 0.01$ (99% confidence): $\chi^2_{\text{crit}} = 20.09$.
   - If $\chi^2 > 20.09$, the column is flagged for potential synthetic fabrication or human manipulation.

#### 2. Population Stability Index (PSI)
Used in `dataset_versioning.py` to quantify structural distribution shift between baseline dataset $P$ and target dataset $Q$:
$$\text{PSI} = \sum_{b=1}^B \left( P_b - Q_b \right) \times \ln\left( \frac{P_b}{Q_b} \right)$$
- $\text{PSI} < 0.10$: Stable (No significant shift).
- $0.10 \le \text{PSI} < 0.25$: Moderate Shift (Warrants monitoring).
- $\text{PSI} \ge 0.25$: Critical Drift (Significant concept drift; models and pipelines must be retrained).

#### 3. Role-Adaptive 4-Pillar Confidence Scoring
Confidence scores are computed across 4 pillars with weights dynamically calibrated by inferred column role:
$$\text{Score} = w_{\text{comp}} \cdot C_{\text{comp}} + w_{\text{cons}} \cdot C_{\text{cons}} + w_{\text{val}} \cdot C_{\text{val}} + w_{\text{stab}} \cdot C_{\text{stab}}$$
- **Identifier Role**: Emphasizes Completeness (0.50) and Stability/Uniqueness (0.30).
- **Metric/Numeric Role**: Emphasizes Validity (0.40) and Completeness (0.30).
- **Categorical Role**: Emphasizes Consistency/Casing (0.40) and Completeness (0.30).
- **Temporal/Date Role**: Emphasizes Validity (0.50) and Completeness (0.30).

Letter grade thresholds:
- **Grade A**: $\ge 90.0\%$
- **Grade B**: $80.0\% - 89.9\%$
- **Grade C**: $70.0\% - 79.9\%$
- **Grade D**: $60.0\% - 69.9\%$
- **Grade F**: $< 60.0\%$

#### 4. Luhn Checksum Algorithm (Mod 10)
Used in `pii_guard.py` to validate credit card numbers and avoid false positives on arbitrary 16-digit integers:
$$\left( \sum_{i=1}^n d_i' \right) \pmod{10} == 0$$
Where every second digit from right to left is doubled, and if doubling results in a number $> 9$, the digits are summed (or subtracted by 9).

---

### 5.4 Row Intelligence & Anomaly Detection Regressors

#### 1. 4-Tier Row Triage Hierarchy
Every row in the dataset is triaged into an explicit single verdict before cleaning:
- **`PROTECT` (Severity 0)**: Row is clean, passes all integrity checks, and is guaranteed bit-identical preservation through all downstream pipelines.
- **`REVIEW` (Severity 1)**: Row exhibits suspicious multivariate relationship deviations; flagged with human-readable driver explanations; never mutated or dropped automatically.
- **`FIX` (Severity 2)**: Row has safe, deterministic cell defects (masked null placeholders like `"N/A"`, imputable values with column null rate $\le 30\%$).
- **`EXCLUDE` (Severity 3)**: Non-data row artifacts (completely blank rows, repeated mid-file table headers, summary subtotal/total rows, export footers).

#### 2. Non-Data Row Pruning Heuristics
Before computing quartiles, variances, or summary statistics, non-data rows are purged:
- **Repeated Headers**: Rows whose cell values exactly match the DataFrame column header strings.
- **Summary Rows**: Rows containing sentinel strings (`"total"`, `"subtotal"`, `"grand total"`, `"summary"`, `"average"`, `"sum"`) in primary text columns while numeric columns contain aggregate values.
- **Sparse Footers**: Rows with $>90\%$ null cells located in the final 5% of the file.

#### 3. Self-Supervised Cross-Column Anomaly Detection (`HistGradientBoostingRegressor`)
Detects rows that violate learned cross-column multivariate relationships:
1. Selects numeric feature columns and trains out-of-fold regression models using `HistGradientBoostingRegressor` (which handles missing values natively without imputation distortion).
2. Computes out-of-fold predictions $\hat{y}_i$ for each row.
3. Calculates residual errors: $e_i = |y_i - \hat{y}_i|$.
4. Scales residuals using the column's Median Absolute Deviation:
   $$z_{\text{residual}} = \frac{e_i}{1.4826 \times \text{MAD}(e)}$$
5. Flags rows with extreme residual deviations ($z_{\text{residual}} \ge 6.0$ and $e_i \ge 2.5 \times 99\text{th percentile}$).
6. Synthesizes contrastive natural-language driver explanations:
   $$\text{"salary=95000 (expected about 35000 given the rest of the row)"}$$

---

### 5.5 Advanced Outlier Detection

#### 1. Tukey's Fences
Standard fences:
$$[Q_1 - 1.5 \cdot IQR, \; Q_3 + 1.5 \cdot IQR]$$
Extreme fences:
$$[Q_1 - 3.0 \cdot IQR, \; Q_3 + 3.0 \cdot IQR]$$

#### 2. Hubert & Vandervieren Skewness-Adjusted Fences
For skewed and heavy-tailed distributions (e.g. monetary transactions, executive compensation), standard Tukey fences cause excessive false positives. GetReport implements skewness-adjusted bounds:
- If skewness is positive ($g_1 > 0$):
  $$\text{Lower} = Q_1 - 1.5 \cdot e^{-4 g_1} \cdot IQR$$
  $$\text{Upper} = Q_3 + 1.5 \cdot e^{3 g_1} \cdot IQR$$
- If skewness is negative ($g_1 < 0$):
  $$\text{Lower} = Q_1 - 1.5 \cdot e^{-3 |g_1|} \cdot IQR$$
  $$\text{Upper} = Q_3 + 1.5 \cdot e^{4 |g_1|} \cdot IQR$$

#### 3. Multivariate Isolation Forest & Local Outlier Factor (LOF)
- **Isolation Forest**: Subsamples data into ensemble isolation trees, measuring path length $h(x)$ required to isolate sample $x$. Shorter path lengths denote anomalies. Contamination is bounded by the empirical outlier ratio.
- **Local Outlier Factor (LOF)**: Measures local density of a sample relative to its $k$-nearest neighbors ($k=20$). Samples with significantly lower density than their neighbors are identified as density-based outliers.

---

### 5.6 Time-Series Intelligence & Changepoint Detection

#### 1. Mann-Kendall Monotonic Trend Test
A non-parametric statistical test to determine whether a time series exhibits a monotonic upward or downward trend without assuming normality or linearity.

**Test Statistic $S$**:
$$S = \sum_{k=1}^{n-1} \sum_{j=k+1}^n \text{sgn}(X_j - X_k) \quad \text{where } \text{sgn}(\theta) = \begin{cases} +1 & \text{if } \theta > 0 \\ 0 & \text{if } \theta = 0 \\ -1 & \text{if } \theta < 0 \end{cases}$$

**Variance of $S$ (corrected for ties)**:
$$\text{Var}(S) = \frac{1}{18} \left[ n(n-1)(2n+5) - \sum_{p=1}^g t_p(t_p - 1)(2t_p + 5) \right]$$
Where $g$ is the number of tied groups and $t_p$ is the size of tied group $p$.

**Standardized Test Statistic $Z$**:
$$Z = \begin{cases} \frac{S - 1}{\sqrt{\text{Var}(S)}} & \text{if } S > 0 \\ 0 & \text{if } S = 0 \\ \frac{S + 1}{\sqrt{\text{Var}(S)}} & \text{if } S < 0 \end{cases}$$
The two-sided p-value is computed from the standard normal distribution $p = 2(1 - \Phi(|Z|))$.

**Sen's Slope Estimator**:
$$\beta = \text{median}\left( \frac{X_j - X_k}{j - k} \right) \quad \text{for all } 1 \le k < j \le n$$

#### 2. STL Decomposition (Loess Smoothing)
Decomposes temporal signals into additive or multiplicative components:
$$Y_t = T_t + S_t + R_t \quad \text{(Additive)} \quad \text{or} \quad Y_t = T_t \times S_t \times R_t \quad \text{(Multiplicative)}$$
Where $T_t$ is trend, $S_t$ is seasonal, and $R_t$ is the stochastic residual. Outliers in time series are detected on the isolated residual component $R_t$.

#### 3. CUSUM (Cumulative Sum) Changepoint Detection
Detects structural shifts in the mean of time series:
$$S_i = \max\left(0, S_{i-1} + (X_i - \mu_0 - k)\right)$$
When $S_i > h$ (decision threshold), a structural changepoint is recorded.

#### 4. Holt-Winters Exponential Smoothing
Performs level, trend, and seasonal forecasting:
$$\ell_t = \alpha(y_t - s_{t-m}) + (1-\alpha)(\ell_{t-1} + b_{t-1})$$
$$b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta)b_{t-1}$$
$$s_t = \gamma(y_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)s_{t-m}$$

---

### 5.7 Missingness Diagnostics & MICE Imputation

#### 1. Missingness Mechanism Classification
- **MCAR (Missing Completely at Random)**: Missingness probability is completely independent of observed and unobserved data.
- **MAR (Missing at Random)**: Missingness depends systematically on other observed columns.
- **MNAR (Missing Not at Random)**: Missingness depends on the value of the missing variable itself.

GetReport computes pairwise $\phi$ (Phi) correlation coefficients across binary missingness indicators $M_j = \mathbb{I}(X_j \text{ is null})$:
$$\phi = \frac{n_{11}n_{00} - n_{10}n_{01}}{\sqrt{n_{1\cdot} n_{0\cdot} n_{\cdot 1} n_{\cdot 0}}}$$
High inter-column $|\phi| \ge 0.5$ identifies coordinated MAR patterns.

#### 2. Multivariate Imputation by Chained Equations (MICE)
For columns approved for multivariate imputation:
1. Initialize missing values with column medians.
2. For each variable $Y$ with missing values:
   - Set current $Y$ values back to missing.
   - Fit Bayesian Ridge / Linear Regression model predicting $Y$ from all other variables $X_{-Y}$.
   - Impute missing values with posterior predictive draws $\hat{Y} + \epsilon$.
3. Repeat through 10 full iterations to preserve multivariate covariance structure without artificial variance attenuation.

---

### 5.8 Feature Engineering & Encodings

#### 1. Cyclical Fourier Embeddings
Transforms periodic time variables (hour of day, day of week, month of year) into continuous trigonometric coordinates preserving temporal adjacency ($23:59 \to 00:01$):
$$x_{\sin} = \sin\left(\frac{2\pi t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi t}{T}\right)$$
Where $T$ is the cycle period (e.g., $T=24$ for hours, $T=7$ for days of week).

#### 2. Empirical Bayes Target Encoding (with Smoothing)
For high-cardinality categorical features, computes smoothed posterior target expectation avoiding overfitting:
$$S_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$
Where $n_i$ is category frequency, $\bar{y}_i$ is within-category target mean, $\bar{y}_{\text{global}}$ is global target mean, and $m$ is smoothing weight parameter. Evaluated via out-of-fold cross-validation to prevent target leakage.

#### 3. Categorical Shannon Entropy & Simpson Diversity
- **Shannon Entropy**:
  $$H(X) = - \sum_{i=1}^k p_i \log_2(p_i)$$
- **Normalized Evenness**:
  $$J = \frac{H(X)}{\log_2(k)}$$
- **Simpson Diversity Index**:
  $$D = 1 - \sum_{i=1}^k p_i^2$$

---

### 5.9 Machine Learning Intelligence (AutoML & Unsupervised)

#### 1. Supervised AutoML Tournament
- **Automated Target Detection**: Evaluates column names (`"target"`, `"label"`, `"churn"`, `"default"`, `"price"`, `"revenue"`), cardinality, and data types to infer classification vs regression.
- **Tournament Grid**: Evaluates HistGradientBoosting, LightGBM, Random Forest, LogisticRegression/Ridge with 5-fold cross-validation.
- **Metrics**: ROC-AUC, F1-macro, PR-AUC, Brier score for classification; $R^2$, RMSE, MAE, MAPE for regression.
- **Key Driver Analysis**: Computes SHAP or permutation importance $\Delta \text{Metric} = \text{Score}_{\text{permuted}} - \text{Score}_{\text{base}}$ to rank influential features.

#### 2. Optimal K-Means Clustering & PCA Projections
- Evaluates cluster counts $k \in [2, 8]$ using three simultaneous indices:
  1. **Silhouette Score**: $s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$
  2. **Davies-Bouldin Index**: $DB = \frac{1}{k} \sum_{i=1}^k \max_{j \ne i} \left(\frac{\sigma_i + \sigma_j}{d(c_i, c_j)}\right)$
  3. **Calinski-Harabasz Variance Ratio**: $CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{n - k}{k - 1}$
- Dimensionality reduction via **Principal Component Analysis (PCA)** computing top 2 orthogonal eigenvectors of the covariance matrix for 2D visual projection.
- **Contrastive Persona Synthesizer**: Calculates feature deviation vectors $\Delta_j = \mu_{\text{cluster}, j} - \mu_{\text{population}, j}$ scaled by population standard deviation to identify defining characteristics of each segment.

---

### 5.10 Smart Schema & Symbolic Discovery

#### 1. Symbolic Linear Equation Solver
Discovers hidden accounting relationships ($Y = aX_1 + bX_2 + c$, e.g., $\text{Total} = \text{Subtotal} + \text{Tax}$):
1. Samples numeric column triads or pairs.
2. Solves the linear system using Moore-Penrose pseudo-inverse via Singular Value Decomposition (SVD):
   $$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
3. Evaluates coefficient of determination $R^2$ and residual norms. If $R^2 \ge 0.9999$ and residuals are within rounding tolerance, records the symbolic equation.

#### 2. Functional Dependency Mining ($X \to Y$)
Identifies relational database functional dependencies where column $X$ uniquely determines column $Y$ (e.g., $\text{zip\_code} \to \text{state}$):
1. Computes grouped cardinality: $|\text{unique}(X)|$ vs $|\text{unique}(X, Y)|$.
2. If $|\text{unique}(X)| == |\text{unique}(X, Y)|$ across $\ge 99.5\%$ of records, flags functional dependency $X \to Y$.

#### 3. Jaro-Winkler String Distance Metric
Used in `issue_ledger.py` for fuzzy deduplication of categorical entities (e.g. `"New York"` vs `"New York, NY"`):
$$d_j = \frac{1}{3}\left(\frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m}\right)$$
$$d_w = d_j + \ell \cdot p \cdot (1 - d_j)$$
Where $m$ is matching characters within window $\lfloor \frac{\max(|s_1|, |s_2|)}{2} \rfloor - 1$, $t$ is transpositions, $\ell$ is common prefix length (up to 4), and $p=0.1$ is prefix scaling factor.

---

### 5.11 Conversational Intelligence & LightRAG Knowledge Graph

#### 1. LightRAG Dual-Level Knowledge Graph Traversal
- **Local Retrieval**: Explores 1-hop and 2-hop neighborhood subgraphs for targeted entity queries using Dijkstra breadth-first expansion.
- **Global Retrieval**: Aggregates high-level dataset community clusters and thematic summaries for broad executive questions.

#### 2. Reciprocal Rank Fusion (RRF)
Combines dense vector retrieval with sparse lexical TF-IDF retrieval:
$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)} \quad \text{where } k=60$$
Where $r_m(d)$ is the rank of document $d$ in retrieval method $m$.

#### 3. Cross-Encoder Reranking
Applies semantic cross-attention scoring $s(q, d)$ over the top candidate chunks using FlashRank or cosine similarity to re-order candidate evidence before context injection.

---

### 5.12 Defensive Security & Sandboxing Algorithms

1. **Polars AST Whitelist Validation (`VULN-01`)**:
   - Parses concept formulas and remediation expressions into Python AST.
   - Enforces `ALLOWED_POLARS_ROOTS = {"pl", "col", "lit", "when"}`.
   - Rejects any file I/O (`read_*`, `scan_*`), lambda definitions, function declarations, attribute access to dunders, or external method calls.
2. **Spreadsheet Formula Injection Neutralization (CWE-1236 / `VULN-03`)**:
   - Sanitizes text cells starting with formula execution triggers: `=`, `+`, `-`, `@`, `\t`, `\r`.
   - Automatically prepends neutralizing single quote (`'`) to all matching text cells before CSV or Excel streaming.
3. **AST Sandboxed Python Runtime & Watchdog Termination (`VULN-06`)**:
   - Parses user analysis code into AST and blocks forbidden modules (`os`, `sys`, `subprocess`, `socket`, `requests`, `pathlib`, `shutil`), reflection attributes (`__class__`, `__subclasses__`, `__dict__`), and dangerous builtins (`eval`, `exec`, `open`, `compile`).
   - Executes code in isolated worker thread with watchdog timer. If execution exceeds timeout (1-30s), forcibly terminates thread using `ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(SystemExit))`.
4. **DuckDB Native C++ Sandboxing**:
   - Sets `SET enable_external_access = false;` immediately upon session creation, disabling native C++ socket connections, disk reads, and Local File Inclusion (LFI).
5. **AES-256-GCM Session Vault**:
   - Snowflake credentials are encrypted in-memory using HKDF key derivation from server secret key and random salt, followed by AES-256-GCM authenticated encryption with 128-bit authentication tag. Zero disk persistence.

---

## 6. Detailed Test Suite Analysis: Complete Inventory of 196 Backend & Frontend Tests

GetReport maintains an exhaustive automated test suite of **487 test cases across 196 test files** in `Backend/tests`, achieving **100% pass rate (446 passed in standard test runner)**, complemented by frontend Vitest unit tests and React Doctor audit scores.

### 6.1 Adversarial & Extreme Numerical Edge Cases (11 Test Suites)
- **`test_case_adversarial_all_constant_string_columns.py`**: Verifies pipeline stability when all columns are identical constant strings; ensures summary stats, variance, and confidence scoring handle zero entropy without division by zero.
- **`test_case_adversarial_alternating_binary_patterns.py`**: Verifies handling of alternating high-frequency binary patterns (`0, 1, 0, 1...`); tests periodicity detection and autocorrelation bounds.
- **`test_case_adversarial_control_characters_in_headers.py`**: Validates sanitization of ASCII control characters (`\x00`, `\r`, `\n`, `\t`, `\b`) in column header names.
- **`test_case_adversarial_deep_nested_json_strings.py`**: Tests resilience when string columns contain deeply nested stringified JSON blobs; ensures type inference does not crash parser.
- **`test_case_adversarial_extreme_aspect_ratios.py`**: Stress-tests extreme shapes: massive tall datasets (100,000 rows $\times$ 1 column) and wide datasets (5 rows $\times$ 500 columns).
- **`test_case_adversarial_heavy_tailed_pareto.py`**: Verifies log-IQR bound calculations on severe Pareto power-law distributions ($\alpha=1.2$) with heavy skewness ($g_1 > 2.0$).
- **`test_case_adversarial_infinite_and_subnormal_floats.py`**: Tests system stability when encountering IEEE 754 positive/negative infinity, `NaN`, subnormal numbers ($1\times 10^{-300}$), and extreme floats ($1\times 10^{300}$).
- **`test_case_adversarial_large_integer_overflow_64bit.py`**: Verifies 64-bit integer handling near boundary limits ($> 2^{63}-1$ and $< -2^{63}$) preventing integer overflow in aggregations.
- **`test_case_adversarial_perfect_multicollinearity.py`**: Tests exact $r = 1.0$ and $r = -1.0$ collinear variable pairs; verifies matrix rank deficiency handling in least squares.
- **`test_case_adversarial_unicode_emojis_rtl.py`**: Verifies multi-byte UTF-8 handling: Arabic/Hebrew Right-To-Left (RTL) scripts, multi-codepoint emojis, and zero-width joiners.
- **`test_case_adversarial_zero_variance_matrix.py`**: Tests covariance and correlation matrix calculation on all-zero or identical constant columns, ensuring graceful `NaN` handling without runtime exceptions.

### 6.2 Confidence Scoring & Forensic Auditing (10 Test Suites)
- **`test_case_confidence_scoring_benfords_law_chi_square.py`**: Verifies Chi-Square goodness-of-fit: natural power-law data conforms ($\chi^2 \le \chi^2_{\text{crit}}$); uniform synthetic data is flagged as an anomaly ($\chi^2 > \chi^2_{\text{crit}}$).
- **`test_case_confidence_scoring_clean.py`**: Verifies that a clean, well-distributed dataset achieves Grade A ($\ge 90\%$).
- **`test_case_confidence_scoring_dirty.py`**: Verifies that a corrupted dataset with high null rates and mixed types receives Grade F ($< 60\%$).
- **`test_case_confidence_scoring_grade_a_boundary.py`**: Validates boundary scoring behavior right at the 90.0% threshold.
- **`test_case_confidence_scoring_grade_f_boundary.py`**: Validates boundary scoring behavior right at the 60.0% threshold.
- **`test_case_confidence_scoring_pillar_weights_sum_to_one.py`**: Verifies mathematical invariant that Completeness, Consistency, Validity, and Stability weights strictly sum to $1.0000$ across all roles.
- **`test_case_confidence_scoring_role_adaptive_weights.py`**: Validates adaptive weighting shifts according to inferred column roles (identifiers, metrics, categories, dates).
- **`test_case_confidence_scoring_consistency_mixed_case.py`**: Tests penalization of erratic casing patterns (`"Active"`, `"active"`, `"ACTIVE"`, `"aCtIvE"`).
- **`test_case_confidence_scoring_stability_zero_variance.py`**: Tests penalization of zero-variance columns under the Stability pillar.
- **`test_case_confidence_scoring_validity_iqr_outliers.py`**: Tests validity degradation when outlier contamination increases.

### 6.3 Missingness, Imputation & MICE (6 Test Suites)
- **`test_case_mice_multivariate_covariance_preservation.py`**: Verifies that MICE multivariate imputation preserves inter-feature covariance matrices without artificial variance suppression.
- **`test_case_multivariate_mice_imputation.py`**: End-to-end test of chained equations imputation across continuous variables.
- **`test_case_missingness_complete_data_matrix.py`**: Validates 100% listwise deletion survival when zero nulls exist.
- **`test_case_missingness_high_cardinality_null_patterns.py`**: Stress-tests missingness correlation on datasets with complex multi-column null permutations.
- **`test_case_missingness_mcar_mar_mnar_correlation.py`**: Verifies automated classification of MCAR vs MAR mechanisms based on pairwise $\phi$ correlation matrices.
- **`test_case_missingness_phi_inter_correlation.py`**: Tests mathematical accuracy of binary contingency $\phi$ coefficients.

### 6.4 Outliers, Dispersion & Distributions (11 Test Suites)
- **`test_case_bimodal_distribution_outlier_behavior.py`**: Verifies that bimodal distributions (two separated clusters) do not treat cluster centers as outliers.
- **`test_case_log_iqr_positive_skewed_monetary_metrics.py`**: Validates log-IQR fence expansion on positively skewed monetary transaction data.
- **`test_case_mad_modified_zscore_zero_median.py`**: Tests modified z-score calculation when column median is zero.
- **`test_case_outlier_isolation_forest_contamination_edge.py`**: Verifies Isolation Forest behavior at contamination boundaries ($0.001$ to $0.20$).
- **`test_case_outlier_multivariate_lof_high_density_clusters.py`**: Validates Local Outlier Factor (LOF) isolation of anomalies near dense spatial clusters.
- **`test_case_outlier_skewness_adjusted_bounds.py`**: Verifies Hubert-Vandervieren adjusted boxplot bounds accommodating sample skewness.
- **`test_case_tukey_iqr_extreme_multiplier_3x.py`**: Tests separation between mild ($1.5\times IQR$) and extreme ($3.0\times IQR$) outliers.
- **`test_case_statistics_robust_dispersion_and_skewness.py`**: Validates mathematical equivalence of MAD, IQR, and Bowley skewness.
- **`test_case_statistics_spearman_rank_correlation.py`**: Tests Spearman rank correlation on non-linear monotonic data ($y = x^3$).
- **`test_case_skewness_kurtosis_uniform_distribution.py`**: Verifies uniform distribution moments: skewness $\approx 0$, excess kurtosis $\approx -1.2$.
- **`test_case_categorical_shannon_entropy_and_diversity.py`**: Verifies Shannon entropy ($H=2.0$ for 4 uniform categories) and Simpson diversity index ($D=0.75$).

### 6.5 Feature Engineering & Encodings (11 Test Suites)
- **`test_case_feature_engineering_cyclical_fourier_embeddings.py`**: Tests trigonometric sine/cosine projections of 24-hour periodic cycles.
- **`test_case_feature_engineering_empirical_bayes_target_encoding.py`**: Validates smoothed target encoding preventing data leakage.
- **`test_case_feature_engineering_encoding.py`**: Tests one-hot, frequency, and ordinal encodings.
- **`test_case_feature_engineering_full_pipeline_run.py`**: End-to-end integration of automated feature transformations.
- **`test_case_feature_engineering_numeric_log_and_sqrt.py`**: Tests variance-stabilizing transforms (`log1p`, `sqrt`).
- **`test_case_feature_engineering_scaling_and_dates.py`**: Tests robust scaling, standard scaling, and date part extraction (year, month, day, dayofweek).
- **`test_case_feature_engineering_target_correlation_proxy.py`**: Tests linear and rank correlation proxies against targets.
- **`test_case_feature_engineering_text_features.py`**: Tests text length, token count, and uppercase character ratio extraction.
- **`test_case_feature_engineering_variance_proxy.py`**: Tests near-zero variance filtering.
- **`test_case_encoding_utf8_sig_bom_resilience.py`**: Tests automated stripping of UTF-8 Byte Order Marks (BOM: `\xEF\xBB\xBF`).
- **`test_case_encoding_windows_1252_resilience.py`**: Tests automated decoding of Windows-1252 / CP1252 encoded files (smart quotes, em-dashes).

### 6.6 Issue Ledger & Remediation Lifecycle (13 Test Suites)
- **`test_case_issue_ledger_approve_all.py`**: Tests batch approval of all detected data quality issues.
- **`test_case_issue_ledger_reject_all.py`**: Tests batch rejection ensuring no transforms are applied.
- **`test_case_issue_ledger_approve_workflow.py` / `test_case_issue_ledger_reject_workflow.py`**: Validates state transitions from `pending` to `approved`/`rejected`.
- **`test_case_issue_ledger_modify_workflow.py`**: Tests user modification of suggested remediation code and validates that modified code executes in restricted scope.
- **`test_case_issue_ledger_locking.py`**: Verifies that a locked ledger rejects subsequent mutations.
- **`test_case_issue_ledger_entity_disambiguation.py`**: Tests automated clustering of near-duplicate entities.
- **`test_case_issue_ledger_fuzzy_duplicates.py`**: Tests Jaro-Winkler fuzzy matching on typos in category names.
- **`test_case_issue_ledger_masked_nulls.py`**: Tests identification of sentinel placeholders (`"N/A"`, `"-999"`, `"#NULL!"`).
- **`test_case_issue_ledger_filter_by_severity_critical.py`**: Tests filtering issues by critical severity.
- **`test_case_issue_ledger_filter_by_status_pending.py`**: Tests filtering issues by status.
- **`test_case_issue_ledger_summary_counts.py`**: Verifies summary counter consistency across mutations.
- **`test_case_issue_ledger_to_dict_serialization.py`**: Tests JSON serialization and deserialization roundtrips.

### 6.7 Row Intelligence & Anomaly Regressors (8 Test Suites)
- **`test_row_intelligence.py`**: Comprehensive 22-test suite verifying 4-tier triage exclusivity, bit-identical immutability on clean records, safe non-data row pruning, gated imputation on sparse rows, and self-supervised multi-column anomaly detection (`HistGradientBoostingRegressor`) with contrastive driver explanations.
- **`test_case_aggregate_summary_row_stripping.py`**: Tests removal of `"Total"` and `"Subtotal"` summary rows before profiling.
- **`test_case_mid_file_header_detection_and_filtering.py`**: Tests detection and removal of repeated mid-file table headers.
- **`test_text_intelligence.py`**: 15 tests verifying character shape clustering (`AAAA-9999`), mojibake repair, and semantic label conflict detection.
- **`test_phone_number_cleaning.py`**: Validates E.164 standardization across international formats.
- **`test_case_contextual_imputation_and_multivariate_ledger.py`**: Validates grouped median/mode contextual imputation.
- **`test_case_ml_readiness_needs_cleaning_boundary_70pct.py`**: Validates ML readiness decision gating at the 70% threshold.
- **`test_case_ml_readiness_ready_boundary_90pct.py`**: Validates ML readiness decision gating at the 90% threshold.

### 6.8 Time Series, Kats & STL (4 Test Suites)
- **`test_timeseries_kats.py`**: 4 comprehensive tests validating Mann-Kendall monotonic trend tests, Sen's slope, seasonal decomposition (additive/multiplicative STL), CUSUM changepoint detection, Holt-Winters exponential smoothing, and time series feature extraction.
- **`test_case_time_series_mann_kendall_trend.py`**: Tests non-parametric monotonic trend detection on strictly increasing and decreasing series.
- **`test_case_time_series_stl_insufficient_periods.py`**: Validates graceful fallback when time series duration is less than two full seasonal periods.
- **`test_case_time_series_stl_irregular_sampling_intervals.py`**: Tests time series resampling and interpolation when sampling timestamps are irregular.

### 6.9 Database, DuckDB, Polars & Storage (11 Test Suites)
- **`test_duckdb_engine.py`**: 5 tests validating zero-copy Apache Arrow memory queries, profile calculations, parameter order binding, and statement-chaining defenses.
- **`test_sql_api_endpoints.py`**: 6 tests validating `/query-sql`, `/sql-schema`, and error mapping.
- **`test_sql_lab_engine.py`**: 3 tests validating DuckDB schema introspection and declarative chart builders.
- **`test_case_polars_lazy_streaming_execution.py`**: Tests streaming execution on large file scans.
- **`test_case_polars_lazy_streaming_skewness_kurtosis.py`**: Validates streaming skewness and kurtosis calculations.
- **`test_case_polars_join_multi_key_collision.py`**: Tests multi-key join resolution with column collision renaming.
- **`test_case_polars_join_outer_preserves_unmatched_rows.py`**: Verifies full outer join preservation of non-matching records.
- **`test_case_dataset_versioning_population_stability_index.py`**: Tests PSI calculation detecting stable vs drifted distributions.
- **`test_case_schema_drift_column_type_mutation.py`**: Tests detection of column data type mutations across versions.
- **`test_case_schema_drift_nullability_surge.py`**: Tests alerts when column missingness surges unexpectedly.
- **`test_case_schema_drift_removed_columns.py`**: Tests detection of dropped or missing features.

### 6.10 Snowflake Cloud Warehouse Ingestion & Vault (14 Test Suites)
- **`test_snowflake_integration.py`**: Comprehensive 15-test suite validating:
  - Account identifier format sanitization (SSRF defense).
  - Double-quote identifier escaping for SQL safety.
  - Arrow schema compilation from Snowflake metadata.
  - Arrow batch type normalization.
  - PKCS#8 RSA private key DER serialization (encrypted and unencrypted).
  - HKDF and AES-256-GCM encrypted session vault lifecycle and sliding TTL expiration.
  - Driver database, schema, table, and view metadata discovery.
  - Streaming Parquet extraction with byte-limit protection.
  - Cooperative query cancellation and session teardown.

### 6.11 Supervised & Unsupervised Machine Learning (9 Test Suites)
- **`test_supervised_intelligence.py`**: 9 tests validating automated classification target detection, regression target detection, candidate ranking, preprocessing matrix generation, multi-model tournament, and key driver analysis.
- **`test_unsupervised_intelligence.py`**: 8 tests validating optimal K-Means clustering (Silhouette, Davies-Bouldin, Calinski-Harabasz), missingness imputation in clustering, PCA 2D projections, and Contrastive Persona synthesis.
- **`test_ml_production.py`**: Tests model serialization (ONNX/Joblib), batch inference scoring, and executive ML summary generation.
- **`test_ml_readiness.py`**: Tests multi-pillar ML readiness score calculations.
- **`test_case_ml_readiness_class_imbalance_penalty.py`**: Tests scoring penalties when target class distribution is severely skewed.
- **`test_case_ml_readiness_constant_columns_penalty.py`**: Tests scoring penalties when uninformative constant columns are present.
- **`test_case_ml_readiness_critical_null_gate.py`**: Verifies that high null rates block "ML Ready" status.
- **`test_hiplot_service.py`**: Validates high-dimensional parallel coordinates rendering, axis type classification, and brush slicing.
- **`test_concept_synthesizer.py`**: Validates AST validation and compilation of user formulas into Polars expressions.

### 6.12 Knowledge Graphs, Graphify & Understand-Anything (3 Test Suites)
- **`test_graphify_integration.py`**: Tests graph extraction, community clustering, God Node identification, surprising connection discovery, shortest path Dijkstra, blast radius impact calculation, Vis.js HTML generation, and SVG vector rendering.
- **`test_rag_graph_pipeline.py`**: Tests integration between knowledge graph entity neighborhoods and conversational prompt injection.
- **`test_understand_anything_integration.py`**: 5 tests validating Understand-Anything architectural layer detection, guided audit tours, semantic fingerprint hashing, and schema drift comparison.

### 6.13 RAG, Semantic Search & Vector Stores (22 Test Suites)
- **`test_antigravity_rag.py`**: Tests Google Antigravity SDK availability and tool construction.
- **`test_rag_advanced.py`**: Tests dual-level RAG query expansion, token streaming, and prompt injection filters.
- **`test_rag_scalability.py`**: Tests vector store indexing performance under high chunk volume.
- **`test_case_rag_chat_history_sliding_window.py`**: Verifies conversational sliding-window history truncation.
- **`test_case_rag_gemini_rest_and_tfidf_fallback.py`**: Tests graceful fallback from Gemini API to in-memory TF-IDF vector store when API keys are absent.
- **`test_case_rag_multi_query_expansion_deduplication.py`**: Tests multi-query expansion and duplicate chunk elimination.
- **`test_case_rag_polars_query_keyword_matching.py`**: Tests hybrid keyword matching against tabular metadata.
- **`test_case_rag_prompt_injection_sanitization.py`**: Tests neutralization of prompt injection phrases (`"Ignore previous instructions"`, `"System Prompt:"`).
- **`test_case_rag_simple_vector_store_cosine.py`**: Tests cosine similarity mathematical calculations in `SimpleVectorStore`.
- **`test_case_rag_suggested_followups_generation.py`**: Tests heuristic follow-up question synthesis.
- **`test_case_rag_table_aware_text_splitter.py`**: Tests semantic row anchoring in table chunking.
- **`test_case_rag_targeted_structured_context_extraction.py`**: Tests extraction of relevant statistical sections for LLM context.
- **`test_case_rag_tfidf_vector_retrieval.py`**: Validates TF-IDF vector retrieval and sparse cosine similarity.
- **`test_case_rrf_reciprocal_rank_fusion_math.py`**: Tests Reciprocal Rank Fusion formula ($k=60$).
- **`test_typesafe_jev.py`**: 4 tests validating TypeSafe JEV client integration.

### 6.14 Security Hardening & Zero-Vulnerability Verification (8 Test Suites)
- **`test_security_audit.py`**: Comprehensive 21-test suite verifying the entire zero-vulnerability security posture:
  - `VULN-01`: Polars AST whitelist strictly enforces `ALLOWED_POLARS_ROOTS` and blocks unauthorized file I/O, lambdas, and functions.
  - `VULN-02`: Zero-bypass HTML/SVG escaping in PDF reports preventing stored/reflected XSS.
  - `VULN-03`: CSV formula injection (CWE-1236) neutralization prepending single quotes to `=, +, -, @, \t, \r`.
  - `VULN-04`: DuckDB memory bounds enforcing Polars `.head(max_rows)` before `.to_dicts()`.
  - `VULN-05`: DuckDB query parameter order preservation and SQL identifier double-quoting.
  - `VULN-06`: Watchdog thread timeout termination via `PyThreadState_SetAsyncExc`.
  - `VULN-07`: WebSocket fail-closed authentication when production security is enforced.
  - `VULN-08`: Trusted proxy IP resolution preventing rate limit bypass via forged headers.
  - `VULN-09`: Request ID header validation against regex `^[a-zA-Z0-9_-]{1,64}$` preventing CRLF injection.
  - `VULN-10`: DuckDB C++ native filesystem sandbox (`SET enable_external_access = false;`).
- **`test_sandboxed_analyst.py`**: Validates AST import blocking, reflection dunder restrictions, and timeout termination.
- **`test_case_security_hardening.py`**: Tests path traversal defenses across all storage providers.
- **`test_case_security_headers_middleware.py`**: Verifies injection of security headers on all endpoints.
- **`test_case_storage_path_traversal_defense.py`**: Tests rejection of relative path traversal (`../../etc/passwd`).
- **`test_case_database_composite_indexes_and_wal.py`**: Validates SQLite WAL mode and composite B-tree indexes.
- **`test_case_database_concurrent_wal_writer_locks.py`**: Stress-tests SQLite concurrent writers with busy timeouts.
- **`test_case_sqlite_wal_concurrency_busy_timeout.py`**: Tests graceful retry under database write contention.

### 6.15 BI Analytics, ECharts & Superset Companion (4 Test Suites)
- **`test_analytics_endpoints.py`**: 9 tests validating `/analytics/status`, `/analytics/sync`, `/analytics/guest-token`, `/analytics/dashboards`, `/analytics/datasets`, and `/analytics/aggregate`.
- **`test_superset_engine.py`**: 19 tests validating Apache Superset column classification, sample value introspection, filtered aggregations, ECharts option tree compilation, and KPI card generation.
- **`test_golden_query_service.py`**: Validates saving, retrieving, executing, and deleting verified SQL Golden KPIs.
- **`test_case_vector_svg_chart_rendering.py`**: Validates vector SVG chart generation with responsive viewboxes and sanitized styles.

### 6.16 Report Generation & Transformation DAG (8 Test Suites)
- **`test_report.py`**: Tests Typst, WeasyPrint, and ReportLab PDF compilation.
- **`test_case_transformation_dag_lineage.py`**: Tests cryptographic hash tracking and parent/child linkage.
- **`test_case_transformation_dag_multi_step_audit_log.py`**: Tests export of multi-step transformation audit logs to CSV and JSON.
- **`test_case_transformation_dag_node_serialization_roundtrip.py`**: Tests complete serialization and deserialization of DAG nodes.
- **`test_case_transformation_dag_reversibility.py`**: Tests reversibility metadata and drop hints.
- **`test_case_transformation_dag_summary_metrics.py`**: Tests summary metrics calculation across DAG nodes.
- **`test_case_transformation_dag_locked_error_enforcement.py`**: Verifies that locked DAGs reject modifications.
- **`test_gx_exporter.py`**: Validates generation of Great Expectations test suites from dataset profiles.

### 6.17 Statistical Core & Mathematical Properties (56 Test Suites)
- **`test_analysis_properties.py`**: Hypothesis property-based testing suite:
  - Mean is bounded within $[ \min(X), \max(X) ]$.
  - Median is bounded within $[ \min(X), \max(X) ]$.
  - Variance is non-negative ($\sigma^2 \ge 0$).
  - Quantiles are monotonic ($Q_{0.25} \le Q_{0.50} \le Q_{0.75}$).
  - Correlation coefficients are bounded within $[-1.0, 1.0]$.
  - Pearson correlation matrix diagonal is identically $1.0$.
  - Null percentage is bounded within $[0.0, 100.0]$.
- **`test_analysis_edge_cases.py`**: 10 tests verifying empty datasets, single-row datasets, all-null columns, constant columns, extreme values, and mixed types.
- **`test_phik_correlation.py`**: Tests mathematical accuracy of non-linear $\Phi_K$ correlation calculations.
- **`test_bulletproof.py`**: 5 tests validating zip bomb detection, decompression ratios, and winsorization.
- **`test_messy_datasets.py`**: Validates cleaning resilience on datasets with mixed encodings, trailing whitespace, and corrupt rows.
- **`test_weak_spots_resolution.py`**: Tests exclusion of ID columns and date columns from categorical distributions and fallback rule generation.

### 6.18 Frontend Vitest Suite & Quality Metrics
- **`Frontend/src/test/example.test.ts`**: Tests multipart file upload without forced Content-Type headers, HTTP error handling, and report download failures.
- **`React Doctor Audit`**: 96/100 score; zero bundle bloat; zero unused dependencies; clean code splitting.
- **`TypeScript Strict Typecheck`**: Verified with `tsc --noEmit -p tsconfig.app.json` (0 errors).

---

## 7. End-to-End Execution Flow (Data Ingestion to Publication)

```
[1. INGESTION]
  User drops file or connects Snowflake
         │
         ▼
[2. STREAMING & VALIDATION]
  • Verify Magic Bytes (file_validation.py)
  • Single-Pass Stream to Storage (storage.py)
  • Compute SHA-256 Hash Deduplication
  • Create Job in Database (db.py)
         │
         ▼
[3. PHASE 1: INSPECTION]
  • Load DataFrame with Polars (data_processing.py)
  • Run Row Intelligence & Prune Non-Data Rows (row_intelligence.py)
  • Run Text Intelligence on Shapes & Casing (text_intelligence.py)
  • Detect Data Quality Issues (issue_ledger.py)
  • Job Status set to WAITING_FOR_USER (progress: 40%)
         │
         ▼
[4. USER INTERACTION (WORKSPACE UI)]
  • User reviews Issue Ledger ("Jira for Dirty Data")
  • Approves, rejects, or customizes remediation rules
  • Clicks "Start Full Analysis"
         │
         ▼
[5. PHASE 2: REMEDIATION & LINEAGE]
  • Apply approved rules via Functional Strategies (remediation.py)
  • Record cryptographic hashes & nodes in TransformationDAG
  • Compute Before-vs-After Comparison Report (comparison.py)
  • Persist Cleaned Dataset (Parquet)
         │
         ▼
[6. DEEP STATISTICAL & ML INTELLIGENCE]
  • Non-Parametric Dispersion (MAD, IQR, Trimmed Mean, Bowley Skewness)
  • Non-Linear Phik (ΦK) Correlation Matrix
  • Benford's Law Chi-Square Forensic Audit
  • Time Series Analysis (Mann-Kendall Monotonic Trend & STL)
  • Missingness Matrix Diagnostics (MCAR / MAR / MNAR)
  • ML Readiness Scoring (0-100)
  • Supervised AutoML Tournament & Key Driver SHAP Importance
  • Unsupervised K-Means, PCA 2D Projection & Contrastive Personas
  • Symbolic Equation Discovery & Functional Dependency Mining
         │
         ▼
[7. PARALLEL SYNTHESIS & INDEXING (asyncio.gather)]
  ┌───────────────────────────┴───────────────────────────┐
  ▼                                                       ▼
[Visualizations & Narrative]            [Knowledge Graph & RAG]
• Headless Matplotlib Charts (Base64)   • Build MultiDiGraph (NetworkX)
• AI Executive Insights (Gemini)       • Cluster Communities & God Nodes
• Apache ECharts Visual Specifications • Chunk & Index in Vector Store
  └───────────────────────────┬───────────────────────────┘
                              │
                              ▼
[8. PUBLICATION & DELIVERABLES]
  • Compile Executive Dossier via Rust Typst Engine (<25MB RAM)
  • Generate Great Expectations Contract Suite (.json, .py)
  • Expose in-process DuckDB SQL Engine over Arrow Batches
  • Launch Interactive Superset BI Studio & Conversational RAG Chat
  • Job Status set to COMPLETED (progress: 100%)
```

---

## 8. Deployment, DevOps & Production Hardening Architecture

1. **Production Docker Container (`Backend/Dockerfile`)**:
   - Multi-stage slim Python 3.12 image.
   - Installs native shared libraries for graphics and typography: `libpango-1.0-0`, `libcairo2`, `libgdk-pixbuf-2.0-0`, `shared-mime-info`, `fonts-dejavu-core`, and `fonts-liberation`.
   - Runs with non-root security principles and production Uvicorn/Granian workers.
2. **Database Resilience & Scalability (`Backend/app/db.py`)**:
   - PostgreSQL (Neon Serverless): Threaded connection pooling with pre-ping validation (`SELECT 1`) to automatically discard connections dropped during serverless SSL idle timeouts; PGVector extension with HNSW cosine indexes.
   - SQLite (Local Dev): Write-Ahead Logging (`PRAGMA journal_mode=WAL`), `PRAGMA synchronous=NORMAL`, `PRAGMA cache_size=-8000`, and `PRAGMA busy_timeout=5000` to prevent writer contention.
3. **Cloud Deployment Blueprint (`render.yaml` & `vercel.json`)**:
   - Backend hosted on Render with automatic health checks (`/health`), Redis background task workers, and persistent disk volume options.
   - Frontend hosted on Vercel with single-page routing rewrites, global CDN caching, and strict CORS origin binding.
4. **Apache Superset Companion Infrastructure (`docker/docker-compose.superset.yml`)**:
   - Fully containerized companion stack: Redis 7, PostgreSQL 15, and Apache Superset with embedded analytics enabled (`FEATURE_FLAGS = {"EMBEDDED_SUPERSET": True}`) and scoped Guest Token JWT authentication.

---

## 9. Conclusion

GetReport is a state-of-the-art, production-hardened platform that unifies high-performance systems programming (Rust Polars engine, Rust Typst PDF compiler, DuckDB in-process C++ OLAP) with advanced mathematical statistics (Phik non-linear correlations, Benford's Law forensic audits, Mann-Kendall monotonic trend tests, MICE multivariate imputation, out-of-fold HistGradientBoosting anomaly detection), agentic conversational intelligence (Google Antigravity SDK, LightRAG dual-level knowledge graph traversal, AST-sandboxed Python runtime), and an agency-grade, accessible user interface adhering to strict anti-slop TypeScript standards.

