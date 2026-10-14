# Comprehensive Architectural, Code Quality, and Security Audit of the GetReport Platform

## Executive Architectural Overview and System Topology

GetReport is an enterprise-oriented automated data profiling, exploratory data analysis, and executive document compilation platform designed to ingest multi-format tabular datasets and transform them into publication-grade portable document deliverables, interactive visual dashboards, and conversational analytical sessions. Built atop a modern polyglot stack, the platform pairs a Python 3.12 and [FastAPI](https://www.google.com/search?q=https%3A%2F%2Ffastapi.tiangolo.com%2F) asynchronous backend (`Backend/app/main.py`) served via the Rust-based [Granian](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Femmett-framework%2Fgranian) ASGI server (`Backend/start.sh`) with a [React 18](https://www.google.com/search?q=https%3A%2F%2Freact.dev%2F), [TypeScript 5.8](https://www.google.com/search?q=https%3A%2F%2Fwww.typescriptlang.org%2F), and [Vite](https://www.google.com/search?q=https%3A%2F%2Fvitejs.dev%2F) single-page frontend (`Frontend/src/App.tsx`). At the data processing core, the system integrates [Polars](https://www.google.com/search?q=https%3A%2F%2Fpola.rs%2F) for vectorized and streaming dataframe operations, [DuckDB](https://www.google.com/search?q=https%3A%2F%2Fduckdb.org%2F) for in-process analytical SQL queries over zero-copy [Apache Arrow](https://www.google.com/search?q=https%3A%2F%2Farrow.apache.org%2F) record batches, and [Typst](https://www.google.com/search?q=https%3A%2F%2Ftypst.app%2F) alongside [WeasyPrint](https://www.google.com/search?q=https%3A%2F%2Fweasyprint.org%2F) and [ReportLab](https://www.google.com/search?q=https%3A%2F%2Fwww.reportlab.com%2F) for multi-engine document compilation.

A full-spectrum static and behavioral inspection of all 516 files across the repository reveals a platform with unusually strong statistical sophistication and thoughtful algorithmic design in its data science core, contrasted against significant structural duplication, latent runtime state defects, and several high-impact security vulnerabilities across its execution sandboxes and multi-tenant boundaries. While the engineering team has clearly invested substantial effort into defensive hardening—evidenced by dedicated security test suites (`Backend/tests/test_security_audit.py` and `Backend/tests/test_case_security_hardening.py`), magic-byte file verification, decompression bomb guards, and pure ASGI middleware—several critical security controls rely on incomplete abstract syntax tree (AST) denylists, unescaped string interpolations, and unverified client-supplied identity headers that can be bypassed under adversarial conditions.

| Subsystem Layer | Primary Technologies and Libraries | Key Source Modules | Architectural Role and Operational Status |
| :--- | :--- | :--- | :--- |
| ASGI API & Routing | FastAPI 0.110+, Granian 1.6+, SlowAPI, Pydantic v2 | `Backend/app/main.py`, `Backend/app/api/endpoints.py`, `Backend/app/api/routes/*` | Exposes REST, SSE, and WebSocket endpoints for ingestion, governance, SQL querying, and chat |
| Hybrid Analytical Engine | Polars 1.2+, DuckDB 1.0+, PyArrow 14+, FastExcel | `Backend/app/services/data_processing.py`, `Backend/app/services/duckdb_engine.py` | Performs streaming file ingestion, schema coercion, and zero-copy Arrow OLAP SQL execution |
| Cloud Warehouse Connector | Snowflake Connector Python 3.7+, SQLGlot 23+, Cryptography 42+ | `Backend/app/services/snowflake/driver.py`, `vault.py`, `executor.py`, `query_guard.py` | Manages RSA/password warehouse authentication, AES-256-GCM session storage, and Arrow batch extraction |
| Statistical & ML Profiling | NumPy, SciPy, Scikit-Learn 1.4+ (`HistGradientBoosting`, `IsolationForest`, `LOF`, `KMeans`, `PCA`) | `Backend/app/services/row_intelligence.py`, `text_intelligence.py`, `analysis/*`, `supervised/*`, `unsupervised/*` | Executes 4-tier row triage, non-parametric statistics, Phik correlations, Benford audits, and AutoML |
| Governance & Lineage | Custom Issue Ledger, Transformation DAG, Great Expectations | `Backend/app/services/issue_ledger.py`, `transformation_dag.py`, `gx_exporter.py` | Tracks data quality defects, human-in-the-loop remediation approvals, cryptographic hashes, and data contracts |
| Conversational RAG & Agent | Google Antigravity SDK, Google GenAI, OpenAI SDK, NetworkX 3.0 | `Backend/app/services/rag_service.py`, `core/rag_utils.py`, `core/graph_store.py`, `query_router.py` | Orchestrates LightRAG knowledge graph traversal, hybrid RRF vector search, and agentic tool execution |
| Document Compilation | Typst 0.15+, ReportLab 4.1+, WeasyPrint, Jinja2 3.1+, Matplotlib 3.8+ | `Backend/app/core/factories/reports.py`, `report_typst.py`, `report_generator.py`, `report_weasyprint.py` | Compiles multi-section executive PDF deliverables with automatic engine fallback chaining |
| Background Task Execution | ARQ 0.26+, Celery 5.3+, Redis 5.0+, Native Asyncio | `Backend/app/core/task_dispatcher.py`, `tasks_arq.py`, `tasks.py`, `services/task_manager.py` | Dispatches two-phase inspection and analysis workflows across Redis workers or in-memory loops |
| Persistence & Storage | AsyncPG, Psycopg2, AioSQLite, SQLite3, Boto3 | `Backend/app/db.py`, `Backend/app/services/storage.py` | Persists job state, golden queries, pgvector embeddings, and uploaded dataset blobs across local, DB, or S3 |
| Frontend Application | React 18.3, TypeScript 5.8, Vite 5.4, TanStack Query v5, Tailwind CSS 3.4, Radix UI, DOMPurify | `Frontend/src/App.tsx`, `services/api.ts`, `hooks/useTaskStatus.ts`, `components/*` | Renders the interactive workspace, issue ledger review UI, SQL studio, parallel coordinates, and chat interface |

The end-to-end data lifecycle follows a two-stage human-in-the-loop pipeline. During Stage 1 (`POST /api/upload`), an uploaded file is validated via magic header bytes and streamed to storage while computing a SHA-256 digest. A background worker (`inspect_file_task` in `Backend/app/tasks_arq.py`) loads the dataset via Polars, runs initial quality diagnostics, populates an interactive `IssueLedger`, and transitions the job state to `waiting_for_user`. During Stage 2 (`POST /api/jobs/{task_id}/analyze`), user-approved cleaning rules are applied, the cleaned dataset is materialized as a Snappy-compressed Parquet file, and the analytical pipeline executes statistical profiling, chart rendering, automated narrative synthesis, Typst PDF compilation, and Knowledge Graph construction concurrently.

## Engineering Strengths and Architectural Pros

### Zero-Copy Hybrid Analytical Processing and Memory-Bounded Ingestion

A standout architectural achievement of GetReport is its commitment to low-memory, high-throughput data movement across the ingestion and analytical layers. Rather than reading uploaded files into memory multiple times to validate signatures, check byte limits, compute cryptographic hashes, and write to disk, `StorageProvider.save_upload_streaming` (`Backend/app/services/storage.py`, lines 31–76) and `LocalStorageProvider.save_upload_streaming` (lines 118–164) process incoming `UploadFile` streams in 64 KB chunks in a single pass. During the very first chunk, `verify_header_bytes` (`Backend/app/core/file_validation.py`) inspects magic byte signatures for `.xlsx` (`PK\x03\x04`), `.xls` (OLE2 `\xD0\xCF\x11\xE0`), `.parquet` (`PAR1`), `.feather`/`.arrow` (`ARROW1`), and `.gz` (`\x1f\x8b`), while rejecting binary null-byte streams masquerading as `.csv` files. Simultaneously, the stream updates an incremental `hashlib.sha256()` digest and enforces `MAX_UPLOAD_SIZE_MB` before flushing directly to disk, immediately unlinking partial files if the size ceiling is breached.

Once on disk, `load_dataframe` (`Backend/app/services/data_processing.py`, lines 437–548) applies pre-parse archive inspection (`_validate_zip_bomb` and `_validate_gzip_bomb`, lines 119–200) to protect the process against decompression bombs (`CWE-409`). The zip validator inspects central directory headers to cap archive member counts at 5,000, reject path traversal entries (`../` or leading `/`), cap uncompressed payload size at `MAX_EXCEL_DECOMPRESSED_SIZE_MB` (200 MB), and reject compression ratios exceeding 100:1 on archives over 10 MB. CSV ingestion further benefits from `_detect_csv_parameters` (lines 251–301), which sniffs character encodings (`utf-8-sig`, `utf-16`, `cp1252`, `latin-1`), delimiters, and leading metadata rows before invoking Polars lazy scanning (`pl.scan_csv` followed by `collect(streaming=True)`).

For interactive analytical SQL queries (`POST /api/jobs/{task_id}/query-sql`), `DuckDBAnalyticalSession` (`Backend/app/services/duckdb_engine.py`) bridges Polars and DuckDB via Apache Arrow memory tables (`df.to_arrow()`), avoiding intermediate CSV serialization or memory copying. Furthermore, the session configures `SET enable_external_access = false;` at the native DuckDB C++ engine level alongside explicit memory and thread caps (`SET memory_limit = '2GB'; SET threads = 4;`), ensuring that even if a query attempts to invoke DuckDB table functions to read local files or remote HTTP resources, the underlying C++ engine refuses external I/O.

### Four-Tier Row Intelligence and Statistical Isolation

Most automated exploratory data analysis tools suffer from a foundational flaw: they compute column means, standard deviations, and quartiles on raw ingested tables that still contain embedded subtotals, repeated mid-file headers, trailing footer disclaimers, or masked null placeholders such as `-999` and `"N/A"`. GetReport solves this problem cleanly through its `row_intelligence.py` and `text_intelligence.py` modules.

| Row Verdict Tier | Severity Rank | Classification Criteria and Detection Mechanism | Downstream Pipeline Handling |
| :--- | :--- | :--- | :--- |
| `PROTECT` | 0 (Lowest) | Passes all structural, format, statistical, and cross-column ML checks with zero anomalies | Guaranteed bit-identical preservation throughout downstream cleaning |
| `REVIEW` | 1 | Extreme univariate outliers ($>3.0 \times \text{IQR}$), out-of-fold regression residual anomalies ($Z \ge 6.0$), or categorical conflicts | Preserved untouched in data; surfaced with contrastive natural-language explanations for human audit |
| `FIX` | 2 | Contains recognized masked null tokens (`"N/A"`, `"-999"`, `"?"`) or imputable nulls in columns with $\le 30\%$ missingness and non-sparse rows ($>10\%$ populated) | Deterministically repaired during cleaning without fabricating synthetic rows |
| `EXCLUDE` | 3 (Highest) | Empty rows, repeated header rows matching column names, summary/total rows (`"Grand Total"`, `"Subtotal"`), trailing report footers, or exact duplicates | Stripped in Step 1b prior to statistical profiling; recorded in `TransformationDAG` with row IDs |

The design of `triage_rows` (`Backend/app/services/row_intelligence.py`) enforces five strict mathematical invariants:
1. Structural artifact detection executes before any statistical estimation, preventing `"Grand Total"` rows from doubling sum metrics or corrupting numeric column types.
2. Gated imputation blocks median or mode replacement on sparse rows ($\le 10\%$ populated cells across $\ge 5$ columns) and heavily missing columns ($>30\%$ null rate), preventing the pipeline from hallucinating synthetic records.
3. Identifier-like columns (`id`, `uuid`, `sku`, `zipcode`, `phone`) are automatically detected and exempted from numerical imputation and outlier clipping.
4. A self-supervised machine learning layer fits 3-fold out-of-fold `HistGradientBoostingRegressor` models across numeric predictors (for datasets with $\ge 300$ rows where $R^2 \ge 0.30$), scaling residuals by the Median Absolute Deviation ($	ext{MAD} \times 1.4826$) within each missingness pattern group and flagging only extreme tail anomalies ($Z \ge 6.0$, exceeding $2.5 \times$ the 99th percentile, capped at $\le 1\%$ of rows) with human-readable contrastive drivers.
5. Companion text triage (`Backend/app/services/text_intelligence.py`) learns regex character shapes (`\p{Nd} -> 9`, `\p{Lu} -> A`, `\p{Ll} -> a`), flags rare typographical variants via `SequenceMatcher`, and detects cross-column categorical contradictions using out-of-fold `HistGradientBoostingClassifier` probability thresholds ($P(\text{actual}) \le 0.05$ while $P(\text{predicted}) \ge 0.90$ and accuracy $\ge 0.97$).

### Comprehensive Statistical, Forensic, and Time-Series Depth

Beyond row-level hygiene, the analytical subsystem (`Backend/app/services/analysis/*`, `confidence_scoring.py`, `smart_schema.py`, `timeseries_kats.py`) demonstrates strong statistical rigor that avoids common pitfalls of naive Gaussian assumptions:

- **Parametric and Non-Parametric Moments (`Backend/app/services/analysis/statistics.py`)**: Executes a single batched Polars lazy aggregation to compute mean, standard deviation, median, interquartile range (IQR), Median Absolute Deviation (MAD), 5% trimmed mean, coefficient of variation (CV), Pearson moment skewness, excess kurtosis, and Bowley's resistant quartile skewness:
  $$S_{\text{Bowley}} = \frac{Q_3 + Q_1 - 2Q_2}{Q_3 - Q_1}$$
- **Non-Linear Mixed-Type Association (`Backend/app/services/analysis/correlations.py`)**: Implements a $\Phi_K$ (Phik) contingency chi-square correlation estimator capable of capturing non-linear, quadratic, cyclical, and mixed numeric-categorical associations that standard Pearson and Spearman correlations miss.
- **Skewness-Adjusted Outlier Fences (`Backend/app/services/analysis/outliers.py`)**: Combines standard Tukey IQR fences ($1.5\times$ and $3.0\times$), Hubert-Vandervieren medcouple/skewness-adjusted fences for asymmetric distributions, log-transformed IQR fences for heavy right-skewed monetary metrics, MAD modified Z-scores, and multivariate ensemble intersection (`IsolationForest` $\cap$ `LocalOutlierFactor`).
- **Forensic Benford's Law Auditing (`Backend/app/services/confidence_scoring.py`)**: Evaluates leading significant digits $d \in \{1, \dots, 9\}$ against the theoretical logarithmic expectation $P(d) = \log_{10}(1 + 1/d)$ using Pearson's chi-square goodness-of-fit test ($\chi^2$ across 8 degrees of freedom) to detect artificially fabricated or rounded financial figures.
- **Symbolic Equation & Functional Dependency Discovery (`Backend/app/services/smart_schema.py`)**: Automatically mines exact and near-exact functional dependencies ($X \to Y$) and tests arithmetic relationships ($A + B = C$, $A - B = C$, $A \times B = C$, $A / B = C$) across numeric columns to identify derived fields.
- **Vectorized Time-Series Changepoints & Decomposition (`Backend/app/services/timeseries_kats.py` & `analysis/time_series.py`)**: Implements Cumulative Sum (CUSUM) log-likelihood ratio hypothesis testing against chi-square critical values, Holt-Winters exponential smoothing with 95% prediction intervals, Mann-Kendall non-parametric monotonic trend tests, and Seasonal-Trend decomposition (STL).

### Registry-Driven Extensibility and Multi-Tier Resilience

The codebase employs a clean, generic `Registry[T]` pattern (`Backend/app/core/factories/registry.py`) across four core extension seams:
1. **Report Compilation Chain (`Backend/app/core/factories/reports.py`)**: `ReportEngineFactory` pre-flights available engines (`TypstReportEngine`, `WeasyPrintReportEngine`, `ReportLabReportEngine`) via `is_available()` and wraps them in a `FallbackReportEngine`. In production, `TypstReportEngine` compiles templates in under 100 ms using under 25 MB of RAM; if Typst fails at render time, the composite engine automatically logs the exception and falls back to ReportLab without failing the user's request.
2. **Data Exporters (`Backend/app/core/factories/exporters.py`)**: `BaseDataExporter` implements the Template Method pattern, enforcing CWE-1236 spreadsheet formula neutralization (`_neutralize_formulas`) across all string columns in one place before delegating format-specific serialization (`CsvDataExporter`, `ExcelDataExporter`, `ParquetDataExporter`, `JsonDataExporter`).
3. **Storage Providers (`Backend/app/core/factories/storage.py`)**: Cleanly abstracts `LocalStorageProvider`, `DatabaseStorageProvider` (PostgreSQL `BYTEA` / SQLite `BLOB` for ephemeral PaaS filesystems like Render), and `S3StorageProvider`.
4. **Dual-Mode Background Task Dispatcher (`Backend/app/core/task_dispatcher.py`)**: Automatically probes Redis availability with a fast timeout and routes background jobs to an asynchronous ARQ worker pool (`Backend/app/tasks_arq.py`) in production while seamlessly falling back to native `asyncio.create_task` coroutines in local or Redis-free environments.

### Pure ASGI Middleware and Custom Frontend Type-Safety Tooling

On the HTTP boundary, `SecurityHeadersMiddleware` (`Backend/app/core/security_headers.py`) and `RequestIDMiddleware` (`Backend/app/core/request_id.py`) are implemented as pure ASGI middleware classes operating directly on `(boundary, receive, send)` rather than inheriting from Starlette's `BaseHTTPMiddleware`. This avoids a well-known Starlette pitfall where `BaseHTTPMiddleware` buffers streaming responses (`StreamingResponse` used by SSE endpoints `/api/status/{task_id}/stream` and `/api/jobs/{task_id}/chat/stream`) or deadlocks background tasks.

On the frontend, the repository includes a custom [Oxlint](https://www.google.com/search?q=https%3A%2F%2Foxc.rs%2F) AST plugin suite (`Frontend/tools/oxlint/anti-slop/`) containing 15 custom static analysis rules (`no-chained-type-assertions`, `no-widen-then-assert`, `require-safety-comment-for-type-assertion`, `no-unsafe-dictionary-type`, `no-unknown-returns`, etc.). This tooling enforces explicit `// SAFETY:` justification comments above type assertions in application code (`Frontend/src/App.tsx`, `Frontend/src/services/api.ts`, `Frontend/src/components/IssueLedger.tsx`), reflecting strong TypeScript hygiene in the newer modules.

## Architectural Cons, Structural Debt, and Functional Bugs

Despite the strengths of its analytical core, a thorough inspection of the codebase uncovers substantial architectural drift, dead and duplicated modules, and several high-impact functional bugs that silently degrade system behavior at runtime.

### Dual-Stack Infrastructure Drift and Code Duplication

The repository exhibits clear signs of an incomplete migration across three major subsystems: task queues, PDF rendering engines, and database access layers. Rather than removing superseded implementations when introducing newer ones, both generations of code remain active or partially wired in the repository.

| Duplicated or Legacy Area | Primary / Active Implementation | Redundant / Dead Counterpart | Maintenance and Operational Impact |
| :--- | :--- | :--- | :--- |
| Background Task Workers | `Backend/app/tasks_arq.py` (465 lines, ARQ + `task_dispatcher.py`) | `Backend/app/tasks.py` (475 lines, Celery) & `Backend/app/core/celery_app.py` | ~500 lines of near-identical pipeline orchestration (`inspect_file`, `resume_analysis`, `generate_pdf`, `rag_ingest`, `build_rag_narrative`). `celery[redis]` still bloats `requirements.txt` and pings Redis on import |
| PDF Report Generators | `Backend/app/services/report_typst.py` & `Backend/app/services/report_generator.py` | `Backend/app/services/report_renderer.py` (325 lines) & `Backend/app/services/report_weasyprint.py` | Both `report_generator.py` and `report_renderer.py` define `generate_pdf_report()` using ReportLab Platypus with divergent layouts (`letter` vs `A4`). `Dockerfile` comments reference WeasyPrint while installing Typst fonts |
| Report Templates | `Backend/app/templates/audit_report.typ` (52.1 KB) & `Backend/app/templates/report.html` (46.3 KB) | `Backend/app/services/templates/audit_report.typ` (53.1 KB) & `Backend/app/services/templates/report.html` (54.6 KB) | Two diverging copies of both the 1,200-line Typst template and the HTML template exist in different directories (`app/templates/` vs `app/services/templates/`). Edits to one copy silently fail to propagate to the other |
| Database ORM vs Raw SQL | `Backend/app/db.py` (518 lines, raw `sqlite3` / `aiosqlite` / `psycopg2` / `asyncpg`) | `Backend/app/db/base.py` & `Backend/app/db/models.py` (SQLAlchemy ORM) | `app/db/base.py` imports `sqlalchemy` (which is not in `requirements.txt`) and points to `sqlite:///./app.db`, while `app/db.py` points to `Backend/data/tasks.db`. Having both `app/db.py` and `app/db/` creates a module/directory naming collision |
| Committed Scratch Artifacts | `Backend/app/services/row_intelligence.py` & `data_processing.py` | `Backend/patches/row_intelligence.patch` (80.4 KB), `Design_MD_Summary.json` (47.1 KB), `skills-lock.json` | 132 KB of temporary git diffs, scraped third-party design system notes, and local agent lockfiles are committed to the repository root and backend tree |

Particularly problematic is the template path resolution in `Backend/app/services/report_typst.py` (lines 28–31):
```python
_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_PRIMARY_TEMPLATE = _TEMPLATE_DIR / "audit_report.typ"
_FALLBACK_TEMPLATE = Path(__file__).parent / "templates" / "audit_report.typ"
```
Because `_PRIMARY_TEMPLATE` (`Backend/app/templates/audit_report.typ`, 52,171 bytes) always exists on disk, `_FALLBACK_TEMPLATE` (`Backend/app/services/templates/audit_report.typ`, 53,116 bytes—which actually contains newer Snowflake truncation notices on line 302!) is **never loaded** by `report_typst.py`! Conversely, `report_weasyprint.py` (line 25) loads `_TEMPLATE_DIR = Path(__file__).parent / "templates"` (`Backend/app/services/templates/report.html`), completely ignoring `Backend/app/templates/report.html`!

### High-Impact Functional Bugs Discovered in Backend Services

A line-by-line trace of the backend service layer revealed seven concrete functional defects that impair accuracy, reliability, and resource management:

1. **Silent 2,000-Character Truncation of All Ingested RAG Reports (`Backend/app/services/rag_service.py`, lines 47, 569, 605)**:
   In `rag_service.py`, `SecurityGuard.sanitize_input(text)` is designed as a user chat query sanitizer and explicitly slices input to 2,000 characters (`text = text[:2000]`). However, both `EnhancedRAGService.ingest_report` (line 569) and `ingest_report_blocking` (line 605) pass the **entire multi-section dataset report narrative** through `SecurityGuard.sanitize_input(text_content)` before passing it to `self.text_splitter.split_text(text_content)`. As a result, every dataset narrative generated by `build_rag_narrative` (which typically spans 8,000 to 25,000 characters covering all column statistics, correlations, outliers, quality scores, and AI takeaways) is silently truncated at character 2,000 before embedding. Roughly 80% to 90% of the dataset's analytical findings are permanently discarded from the vector store before a single chunk is indexed.

2. **Duplicate Function Definition Overwriting Fuzzy Duplicate Detection (`Backend/app/services/issue_ledger.py`, lines 378 and 582)**:
   The function `_detect_fuzzy_duplicate_issues(df: pl.DataFrame) -> list[Issue]` is defined twice in the same module (`issue_ledger.py`). The first implementation (lines 378–423) uses `difflib.SequenceMatcher` with a 0.88 similarity ratio on entity columns (`name`, `company`, `vendor`, `customer`, `city`, `brand`). The second implementation (lines 582–617) uses `disambiguate_categorical_entities` and `jaro_winkler_similarity` with a 0.80 threshold across all `pl.Utf8` columns. In Python, the second definition silently overwrites the first in the module namespace, rendering lines 378–423 dead code and causing the 0.80 Jaro-Winkler threshold to run on every string column (which frequently flags distinct short codes or dates as fuzzy duplicates).

3. **Closed Event Loop Crash in `generate_findings_sync` (`Backend/app/services/llm_finding.py`, lines 72–118, 489)**:
   `llm_finding.py` instantiates module-level `AsyncOpenAI` clients (`_gemini_client`, `_or_client`, `_openai_client`) at import time. Each `AsyncOpenAI` client maintains an internal `httpx.AsyncClient` connection pool that binds to the active `asyncio` event loop on its first request. However, `tasks_arq.py` (line 277) invokes `await asyncio.to_thread(generate_findings_sync, analysis_result)`, and `generate_findings_sync` calls `asyncio.run(generate_findings(analysis_data))`. `asyncio.run()` spins up a temporary event loop on the worker thread, executes one HTTP request (binding the global `AsyncOpenAI` client's `httpx` transport to that temporary loop), and then **closes the loop** upon returning. On the second dataset analyzed by the process, `asyncio.run()` creates a new event loop, and the global `AsyncOpenAI` client raises `RuntimeError: Event loop is closed` when attempting to reuse its pooled connections, forcing the platform into its deterministic offline fallback for all subsequent jobs.

4. **Non-Existent LLM Model Identifiers Causing Cascading HTTP 404 Latency (`Backend/app/services/llm_finding.py`, lines 45–64; `rag_service.py`, lines 766, 996)**:
   The primary model priority lists contain fabricated or non-existent model identifiers:
   - `GEMINI_MODELS` lists `"gemini-3.7-flash"`, `"gemini-3.6-flash"`, and `"gemini-3.5-flash-lite"` ahead of `"gemini-2.5-flash"`.
   - `OPENROUTER_PAID_MODELS` lists `"deepseek/deepseek-v4-flash"`, `"qwen/qwen3.6-flash"`, and `"z-ai/glm-5.5-air"`.
   - `AntigravityAgent` loops through `["gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash"]`.
   Every LLM invocation sequentially sends requests to three non-existent Gemini model endpoints, receiving three consecutive `404 NotFoundError` responses before finally falling back to `gemini-2.5-flash`. This adds 1.5 to 4.0 seconds of wasted network round-trips and noisy warning logs to every chat turn and report generation job.

5. **Unimplemented Issue Modification Endpoint (`Backend/app/api/routes/issues.py`, line 21)**:
   `issues.py` defines `class IssueModifyRequest(BaseModel): fix_code: str; note: str = ""` on line 21, and `IssueLedger.modify()` is implemented and unit-tested in `Backend/app/services/issue_ledger.py` (lines 118–127), yet `issues.py` only exposes `/approve`, `/reject`, `/approve-all`, `/reject-all`, and `/lock`—completely omitting the `@router.post("/jobs/{task_id}/issues/{issue_id}/modify")` route handler.

6. **Snowflake Connection Leaks and 1-Second Authentication Churn (`Backend/app/services/snowflake/driver.py`, lines 175–203; `executor.py`, lines 58–80)**:
   In `RealSnowflakeDriver`, `submit_query_async` (line 175) and `fetch_arrow_batches` (line 195) call `conn = self._create_connection(config)` without a `with` context manager and never call `conn.close()`, leaking open Snowflake sessions and sockets. Meanwhile, during the asynchronous query polling loop in `execute_and_stream_snowflake` (`executor.py`, line 58), `while driver.is_query_running(session_config, sfqid):` runs every 1.0 second (`time.sleep(1.0)`), and `is_query_running` opens a brand-new Snowflake connection (`with self._create_connection(config) as conn:`) on every single iteration! A 60-second Snowflake query performs 60 full TLS handshakes and 60 Snowflake JWT/password authentications. Ironically, `EncryptedSessionVault` (`Backend/app/services/snowflake/vault.py`, lines 183–204) implements `get_pooled_connection` and `set_pooled_connection`, which are never called anywhere in the codebase.

7. **Fragile Custom Database Adapter (`Backend/app/db.py`, lines 26–128)**:
   Instead of using SQLAlchemy 2.0 Async or SQLModel, `db.py` wraps `psycopg2`, `asyncpg`, `sqlite3`, and `aiosqlite` in custom adapter classes (`PostgresCursor`, `AsyncPostgresCursor`) that rewrite SQLite `?` placeholders into `%s` or `$1, $2, ...` via a regular expression (`r"(\'(?:[^'\\]|\\.|'')*\'|"(?:[^"\\]|\\.|"")*")|\?"`). Additionally, `AsyncPostgresConnection.commit()` is implemented as `pass` (relying on `asyncpg` autocommit mode), which means multi-statement operations cannot be wrapped in an atomic transaction, and `AsyncPostgresCursor.execute()` always calls `await self.conn.fetch(pg_sql, *sanitized_params)` even for `INSERT`, `UPDATE`, and `DELETE` statements.

### Frontend Architecture, State Management, and CI/CD Deficiencies

On the frontend, the user interface is visually polished and responsive, but exhibits severe component bloat and testing gaps:

| Frontend / CI Area | Observed Metric or Configuration | Architectural Consequence |
| :--- | :--- | :--- |
| `ReportGeneration.tsx` Component Size | 2,190 lines of code, 28 `useState` hooks in a single component | Mixes PDF polling, chart gallery tabs, column confidence cards, DuckDB SQL IDE, schema browser, and CSV export in one file, causing wide re-render blast radii |
| `SnowflakeConnectModal.tsx` Component Size | 1,269 lines of code, 29 `useState` hooks in a single modal | Combines RSA PEM drag-and-drop parsing, credential form state, 3-level warehouse tree caching, SQL builder, and data preview table in one monolithic component |
| `doctor.config.json` Rule Suppression | 16 of 16 `react-doctor` rules set to `"off"` | Disables `no-giant-component`, `unused-file`, `unused-dependency`, `no-prop-callback-in-effect`, and state-hygiene checks, neutralizing the `react-doctor.yml` CI workflow |
| Frontend Automated Test Suite | 1 test file (`Frontend/src/test/example.test.ts`, 43 lines, 3 unit tests) | Zero component rendering tests, zero hook tests (`useTaskStatus.ts`), and zero interaction tests across 12,345 lines of TypeScript/React code |
| Unused UI Primitives & Dependencies | 30+ unused Shadcn/Radix UI files in `Frontend/src/components/ui/` (`menubar`, `carousel`, `calendar`, `drawer`, `input-otp`, `command`, `context-menu`, `resizable`, `sidebar`) | Inflates `node_modules`, `package.json`, and repository surface area with thousands of lines of dead UI wrapper code |
| CI Pipeline (`ci.yml`) Linter & Typecheck Omission | Runs only `pytest`, `npm run test`, and `npm run build` | Never executes `mypy`, `black --check`, `isort --check`, `npm run lint`, `npm run lint:oxlint`, or `npm run typecheck` (`tsc --noEmit`), allowing type errors and lint violations to merge undetected |

## Security Vulnerabilities and Threat Analysis

A comprehensive security audit was conducted across all trust boundaries: file upload and archive parsing, AST-sandboxed code execution, Polars concept expression evaluation, Issue Ledger remediation execution, DuckDB and Snowflake SQL interfaces, authentication and tenant isolation, RAG prompt/output handling, and deployment configurations. While the README claims a "0 Vulnerabilities | Audit Hardened" posture and includes tests for previously patched issues (`VULN-01` through `VULN-10`), deep source code inspection identified **15 distinct security vulnerabilities** (3 Critical, 5 High, 5 Medium, and 2 Low) that bypass the existing guards.

| ID | Vulnerability Title | Severity | CWE | Affected File(s) and Line Numbers |
| :--- | :--- | :--- | :--- | :--- |
| **SEC-CRIT-01** | AST Sandbox Escape and Arbitrary Code Execution via Module Attribute Traversal and Function Aliasing | **Critical** | [CWE-94](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F94.html) / [CWE-265](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F265.html) | `Backend/app/services/sandboxed_analyst.py` (L70–305), `Backend/app/api/routes/report.py` (L490), `Backend/app/services/rag_service.py` (L513) |
| **SEC-CRIT-02** | Arbitrary Python Code Injection in Issue Ledger via Unescaped Column Headers and Weak Remediation AST Check | **Critical** | [CWE-94](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F94.html) | `Backend/app/services/issue_ledger.py` (L180–900, L1020–1045) |
| **SEC-CRIT-03** | Polars Root Whitelist Bypass and Arbitrary Expression Evaluation in Virtual Concept Synthesizer | **Critical** | [CWE-94](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F94.html) | `Backend/app/services/concept_synthesizer.py` (L46–136, L215–225) |
| **SEC-HIGH-01** | SQL Injection in Snowflake Object Discovery via Unescaped `from_last` Cursor Parameter | **High** | [CWE-89](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F89.html) | `Backend/app/services/snowflake/driver.py` (L142–145), `Backend/app/api/routes/snowflake.py` (L183) |
| **SEC-HIGH-02** | Broken Multi-Tenant Isolation, Spoofable `X-User-Id` Headers, and Shared `"default_user"` Vault Fallback | **High** | [CWE-287](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F287.html) / [CWE-639](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F639.html) | `Backend/app/core/auth.py` (L136–150), `Backend/app/services/snowflake/vault.py` (L152–172), `Backend/app/api/routes/snowflake.py` (L111, L326) |
| **SEC-HIGH-03** | Stored HTML/XSS Injection in RAG Offline Fallback, Golden Query Output, and WeasyPrint SVG Filter Bypass | **High** | [CWE-79](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F79.html) | `Backend/app/services/rag_service.py` (L166–327, L684–696, L906–918), `Backend/app/services/templates/report.html` (L877–879) |
| **SEC-HIGH-04** | Hardcoded Default Vault `SECRET_KEY`, Unauthenticated Default Posture, and Client-Side `VITE_API_KEY` Exposure | **High** | [CWE-798](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F798.html) / [CWE-306](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F306.html) | `Backend/app/core/config.py` (L22–26), `render.yaml` (L12–48), `Frontend/src/services/api.ts` (L48, L398) |
| **SEC-HIGH-05** | Denial of Service via Synchronous Event-Loop Join Explosion, Pre-Truncation DuckDB OOM, and Missing Rate-Limit Middleware | **High** | [CWE-400](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F400.html) | `Backend/app/api/routes/upload.py` (L166–240), `Backend/app/services/duckdb_engine.py` (L104–112), `Backend/app/main.py` (L41–43) |
| **SEC-MED-01** | Rate-Limit Bypass via Left-Most `X-Forwarded-For` Header Spoofing Behind Reverse Proxies | **Medium** | [CWE-348](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F348.html) | `Backend/app/core/limiter.py` (L15–46) |
| **SEC-MED-02** | Sensitive Snowflake Session Bearer Token Exposed in GET URL Query Parameters | **Medium** | [CWE-598](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F598.html) | `Backend/app/api/routes/snowflake.py` (L142, L160, L179), `Frontend/src/services/api.ts` (L525–535) |
| **SEC-MED-03** | Incomplete Spreadsheet Formula Injection (CWE-1236) Protection in SQL CSV Download and DAG Audit Export | **Medium** | [CWE-1236](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F1236.html) | `Frontend/src/components/ReportGeneration.tsx` (L380–392), `Backend/app/services/transformation_dag.py` (L384–410) |
| **SEC-MED-04** | Cross-Request Matplotlib Global State Race Condition and Cross-Tenant Chart Leakage | **Medium** | [CWE-362](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F362.html) | `Backend/app/services/sandboxed_analyst.py` (L270, L315, L353), `Backend/app/services/visualization.py` (L48–68) |
| **SEC-MED-05** | Unsanitized `Content-Disposition` Header Parameter Injection in Direct Report Endpoint | **Medium** | [CWE-113](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F113.html) | `Backend/app/api/routes/report.py` (L290–305) |
| **SEC-LOW-01** | Unverified Pickle Deserialization via `joblib.load` in ML Artifact Serializer | **Low** | [CWE-502](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F502.html) | `Backend/app/services/ml_production/serializer.py` (L38–55) |
| **SEC-LOW-02** | World-Writable (`chmod 777`) Runtime Directories in Production Docker Image | **Low** | [CWE-732](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F732.html) | `Backend/Dockerfile` (L35–36) |


### Critical Vulnerabilities

#### SEC-CRIT-01: AST Sandbox Escape and Arbitrary Code Execution in SandboxedAnalystAgent

- **Location**: `Backend/app/services/sandboxed_analyst.py` (lines 70–305), exposed directly via `POST /api/jobs/{task_id}/sandbox-exec` (`Backend/app/api/routes/report.py`, line 490) and indirectly via the conversational agent tool `run_sandboxed_python_analysis` (`Backend/app/services/rag_service.py`, line 513).
- **Mechanism and Root Cause**:
  `SandboxedAnalystAgent` attempts to sandbox arbitrary user-submitted or LLM-generated Python code in-process using Python's `ast.parse` and `exec(code_str, {"__builtins__": self.SAFE_BUILTINS}, execution_namespace)`. This implementation contains two fatal flaws:
  1. **Call-Site-Only Method Filtering (Function Aliasing Bypass)**: In `validate_code_ast` (lines 194–212), `FORBIDDEN_METHOD_CALLS` (`read_csv`, `write_csv`, `system`, `popen`, `run`, `load`, `save`, etc.) is checked *only* when an `ast.Call` node's `func` is directly an `ast.Attribute` (`obj.method()`). It is **not** checked when an attribute is referenced without being immediately called (`fn = pl.read_csv`). When `fn('/etc/passwd')` is called on the next line, `node.func` is an `ast.Name(id='fn')`, which is only checked against `FORBIDDEN_BUILTIN_CALLS`. Thus, assigning any forbidden method to a local variable completely bypasses `FORBIDDEN_METHOD_CALLS`.
  2. **Live Module Graph Exposure in `execution_namespace` (Complete Sandbox Escape)**: Line 277 injects live Python module objects (`"pl": pl, "np": np, "plt": plt`) directly into `execution_namespace`. In Python, top-level modules hold non-dunder references to standard library modules that they imported at startup. Specifically, `matplotlib.pyplot` (`plt`) exposes `plt.sys` (the live `sys` module) and `plt.matplotlib.shutil`, while `numpy` (`np`) exposes `np.ctypeslib`. Accessing `plt.sys.modules` requires zero `import` statements and zero `__dunder__` attributes! Through `plt.sys.modules`, sandboxed code can retrieve `"os"`, `"subprocess"`, `"builtins"`, or `"app.core.config"`, assign functions like `builtins.open`, `builtins.eval`, or `os.listdir` to a local variable, and execute arbitrary operating system commands or read environment secrets (`GEMINI_API_KEY`, `DATABASE_URL`, `SECRET_KEY`).
  3. **Unsigned Thread ID Bug in Watchdog Termination**: When a sandboxed script times out (line 299), the watchdog calls `ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), ctypes.py_object(SystemExit))`. In CPython 3.7+, the first parameter of `PyThreadState_SetAsyncExc` is `unsigned long` (`ctypes.c_ulong`), and if the return value exceeds 1, `PyThreadState_SetAsyncExc(tid, None)` must be called to reset interpreter state.
- **Remediation**:
  Never execute untrusted user Python code inside the main FastAPI/worker process space using `exec()`, as Python AST sandboxes with access to complex C-extension modules (`numpy`, `polars`, `matplotlib`) are fundamentally porous.
  1. Move `SandboxedAnalystAgent` execution into an isolated, unprivileged subprocess or ephemeral container (such as `nsjail`, `bubblewrap`, `gVisor`, or a restricted subprocess with `seccomp` filters, `RLIMIT_AS` memory limits, `RLIMIT_CPU`, and an empty temporary working directory with no environment variables).
  2. As an immediate defense-in-depth patch at the AST layer, check `node.attr in cls.FORBIDDEN_METHOD_CALLS` inside the `isinstance(node, ast.Attribute)` branch (not just inside `ast.Call`), block attribute access to module-level internals (`sys`, `os`, `modules`, `builtins`, `ctypeslib`, `matplotlib`, `shutil`, `subprocess`, `io`), and pass a restricted proxy wrapper around `df` rather than exposing the full `plt` and `np` module objects.

#### SEC-CRIT-02: Arbitrary Python Code Injection in IssueLedger Remediation Execution

- **Location**: `Backend/app/services/issue_ledger.py` (lines 180–900 and `apply_remediation` at lines 982–1047).
- **Mechanism and Root Cause**:
  Throughout `issue_ledger.py`, every detector function (`_detect_missing_value_issues`, `_detect_business_rule_violations`, `_detect_summary_rows_and_mid_headers`, `_detect_duplicate_issues`, `_detect_empty_column_issues`, `_detect_constant_column_issues`, `_detect_type_mismatch_issues`, `_detect_outlier_issues`, `_detect_masked_null_issues`, `_detect_whitespace_and_case_issues`, `_detect_extreme_skewness_issues`, `_detect_rare_category_issues`) constructs executable Python code strings using unescaped f-string interpolation of column names and sample values:
  ```python
  fix_code = f"df = df.with_columns(pl.col('{col}').fill_null(pl.col('{col}').median()))"
  ```
  Although `data_processing.py` normalizes column names to snake_case during standard ingestion, `detect_issues(df)` and `apply_remediation(df, ledger)` can be called on DataFrames with arbitrary column names or via modified ledger issues (`IssueLedger.modify`). Furthermore, in `_detect_fuzzy_duplicate_issues` (line 420), cell values from the dataset (`fuzzy_pairs[0][1]`) are interpolated directly into `fix_code` without escaping.
  When `apply_remediation` executes (lines 1030–1041), its AST check is far weaker than `SandboxedAnalystAgent`:
  ```python
  tree = ast.parse(issue.fix_code, mode="exec")
  for node in ast.walk(tree):
      if isinstance(node, (ast.Import, ast.ImportFrom)):
          raise ValueError("Imports are prohibited in remediation fix code.")
      elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
          raise ValueError(f"Accessing private attribute '{node.attr}' is prohibited in remediation code.")
      elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
          if node.func.id in ("eval", "exec", "open", "compile", "__import__", "globals", "locals"):
              raise ValueError(f"Function '{node.func.id}' is prohibited in remediation code.")
  exec(issue.fix_code, safe_globals, loc_env)
  ```
  Notice that `safe_globals` contains `"pl": pl, "datetime": datetime, "re": re`, and the AST loop does **not** block `pl.read_csv`, `pl.write_csv`, `pl.scan_parquet`, `pl.DataFrame.map_elements`, `pl.DataFrame.pipe`, or `re.sub` with callable callbacks, nor does it block subscript access (`obj["__builtins__"]` or `re.sys`!).
- **Remediation**:
  Eliminate `exec(issue.fix_code)` entirely from `issue_ledger.py`. Notably, the codebase already has a clean, declarative, functional remediation registry in `Backend/app/core/factories/remediation.py` (`remediation_registry` with pure Polars functions `drop_null_rows`, `fill_null_mean`, `fill_null_median`, `fill_null_mode`, `fill_null_value`, `replace_outliers_iqr`)! `Issue` objects should store a declarative `action: str` and `params: dict[str, Any]` (displaying `fix_code` purely as a read-only UI preview string), and `apply_remediation` should dispatch exclusively through `app.core.factories.remediation.apply_remediation(issue.action, df, issue.column, issue.params)`.

#### SEC-CRIT-03: Polars Root Whitelist Bypass in VirtualConceptSynthesizer

- **Location**: `Backend/app/services/concept_synthesizer.py` (lines 46–136, 215–225), exposed via `POST /api/jobs/{task_id}/concepts/derive` (`Backend/app/api/routes/report.py`, line 513).
- **Mechanism and Root Cause**:
  `VirtualConceptSynthesizer.validate_expression_ast` parses `formula_or_intent` in `mode="eval"` and attempts to restrict calls on `pl` to `ALLOWED_POLARS_ROOTS` (`col`, `lit`, `when`, `coalesce`, etc., lines 117–122):
  ```python
  elif isinstance(node.func, ast.Attribute):
      ...
      if isinstance(node.func.value, ast.Name) and node.func.value.id == "pl":
          if node.func.attr not in cls.ALLOWED_POLARS_ROOTS:
              raise ConceptSynthesizerSecurityViolation(...)
  ```
  However, this check only triggers when `node` is an `ast.Call` and `node.func` is directly `pl.<attr>`. If an attacker accesses an attribute on `pl` *without* it being the direct `node.func` of an `ast.Call`—for example, wrapping `pl.read_excel` or `pl.read_database_uri` in a tuple or list comprehension `[f('/etc/passwd') for f in [pl.read_excel]][0]` or `(pl, pl.read_excel)[1]('file.xlsx')`—then:
  1. During the `ast.Attribute` visit for `pl.read_excel`, `validate_expression_ast` only checks `node.attr.startswith("__")` and `node.attr in cls.FORBIDDEN_EXPR_METHODS`. Because `FORBIDDEN_EXPR_METHODS` omits `read_excel`, `read_avro`, `read_delta`, `read_database`, `read_database_uri`, `scan_delta`, `map_batches`, and module-level attributes on `pl`, the `ast.Attribute` check passes!
  2. During the `ast.Call` visit for `f(...)` or `(...)[1](...)`, `node.func` is an `ast.Name` or `ast.Subscript`, not an `ast.Attribute` on `pl`, so the `ALLOWED_POLARS_ROOTS` check is completely skipped!
  3. The expression is then passed directly to Python's `eval(expr_str, {"__builtins__": self.SAFE_EVAL_BUILTINS}, {"pl": pl})` (line 169 and line 219).
- **Remediation**:
  Enforce the `ALLOWED_POLARS_ROOTS` whitelist inside the `isinstance(node, ast.Attribute)` handler whenever `isinstance(node.value, ast.Name) and node.value.id == "pl"`, and explicitly disallow `ast.ListComp`, `ast.SetComp`, `ast.DictComp`, `ast.GeneratorExp`, and `ast.Subscript` calls in `validate_expression_ast`. Better yet, parse arithmetic/column formulas into Polars `Expr` objects using `sqlglot.parse_one(expr, read="duckdb")` or `pl.sql_expr(expr_str)` rather than invoking Python's `eval()`.


### High-Severity Vulnerabilities

#### SEC-HIGH-01: SQL Injection in Snowflake Object Discovery (from_last Parameter)

- **Location**: `Backend/app/services/snowflake/driver.py` (lines 142–145), exposed via `GET /api/snowflake/objects` (`Backend/app/api/routes/snowflake.py`, line 183).
- **Mechanism and Root Cause**:
  While `discover_objects` properly quotes `database_name` and `schema_name` using `quote_ident()`, it concatenates the user-supplied query parameter `from_last: Optional[str] = Query(None)` directly into a single-quoted SQL string literal without escaping or validation:
  ```python
  tbl_sql = f"SHOW TABLES IN SCHEMA {schema_ident} LIMIT {limit}"
  if from_last:
      tbl_sql += f" FROM '{from_last}'"
  cur.execute(tbl_sql)
  ```
  Passing a `from_last` value containing a single quote `'` breaks out of the SQL string literal and enables SQL injection against the authenticated Snowflake warehouse session (`CWE-89`).
- **Remediation**:
  Validate `from_last` against a strict identifier pattern (`^[a-zA-Z0-9_$ -]{1,255}$`) or escape single quotes (`from_last.replace("'", "''")`) and disallow semicolons and SQL comment tokens (`--`, `/*`).

#### SEC-HIGH-02: Broken Multi-Tenant Isolation and Spoofable Identity Headers

- **Location**: `Backend/app/core/auth.py` (lines 136–150), `Backend/app/api/routes/*`, and `Backend/app/services/snowflake/vault.py` (lines 152–172).
- **Mechanism and Root Cause**:
  GetReport implements `check_task_ownership(job.owner_id, caller_owner_id)` to prevent Broken Object Level Authorization (BOLA / IDOR). However, this mechanism fails in practice for four compounding reasons:
  1. **Unverified Client Headers**: Every route extracts `caller_owner_id = request.headers.get("x-user-id") or request.headers.get("x-consumer-id")`. Because there is no JWT signature verification, HMAC, or server-side session lookup validating `X-User-Id`, any client can spoof any tenant's `X-User-Id` header.
  2. **Frontend Never Sends `X-User-Id`**: Inspection of `Frontend/src/services/api.ts` confirms that the frontend never sends `X-User-Id` or `X-Consumer-Id`. Consequently, every job created via the web UI has `owner_id = None`. Because `check_task_ownership` only checks ownership `if task_owner_id:`, all UI-created jobs are globally accessible to any caller who knows or guesses the `task_id`.
  3. **Universal `"default_user"` Fallback in Snowflake Vault**: In `snowflake.py` (line 111), `caller_owner_id` defaults to `"default_user"` when `X-User-Id` is absent. Even worse, in `EncryptedSessionVault.retrieve_session` (`vault.py`, lines 153–155):
     ```python
     candidate_users = [user_id]
     if user_id != "default_user":
         candidate_users.append("default_user")
     ```
     If a session was stored under `"default_user"`, any user ID can decrypt it.
  4. **Unprotected Session Disconnect**: `POST /api/snowflake/disconnect/{session_id}` (`snowflake.py`, line 326) invalidates any session ID without checking `user_id` or ownership.
- **Remediation**:
  Bind job ownership and Snowflake vault sessions to server-issued, cryptographically signed session tokens (such as an `HttpOnly`, `Secure`, `SameSite=Lax` cookie containing an HMAC-signed or JWT session ID generated on first visit) or per-tenant API keys rather than trusting raw `X-User-Id` headers. Remove the `"default_user"` fallback from `retrieve_session`.

#### SEC-HIGH-03: Stored/Reflected HTML and XSS Injection in RAG Fallbacks and WeasyPrint SVG Filter

- **Location**: `Backend/app/services/rag_service.py` (lines 166–327, 684–696, 906–918) and `Backend/app/services/templates/report.html` (lines 877–879).
- **Mechanism and Root Cause**:
  1. **Unescaped HTML Construction in `rag_service.py`**: While LLM-generated answers pass through `_format_answer_for_ui()` (which calls `html.escape()`), two other code paths in `rag_service.py` construct raw HTML strings directly from dataset contents without calling `html.escape()`:
     - `_generate_smart_dataset_answer()` (lines 166–327) interpolates `filename`, column names (`c`), outlier min/max values, issue descriptions (`desc`), and recommendation titles directly into HTML markup (`<h3>...{filename}</h3>`, `<b>{col}</b>: {desc}`).
     - The Golden Query execution path in `chat_with_report` (lines 684–696) and `chat_stream_with_report` (lines 906–918) interpolates `golden.question`, `golden.sql_query`, column names (`k`), and raw dataset cell values (`v`) directly into HTML (`f"<b>{k}</b>: {v}"`).
     Although `ChatInterface.tsx` uses `DOMPurify.sanitize` with an `ALLOWED_TAGS` whitelist (`b`, `strong`, `i`, `em`, `code`, `pre`, `h3`, `h4`, `br`, `ul`, `ol`, `li`, `span`, `p`) and `ALLOWED_ATTR: ["class"]`, an attacker can still inject arbitrary allowed HTML formatting and Tailwind classes (`<span class="fixed inset-0 bg-black z-50">...</span>`) to deface or overlay the entire application UI (HTML/CSS UI Redressing), and any API client consuming `/chat` without DOMPurify is exposed to full Stored XSS.
  2. **Trivial SVG Filter Bypass in `report.html` (line 878)**:
     ```jinja2
     {% if '<svg' in img_data and '<script' not in img_data | lower and '<foreignobject' not in img_data | lower %}
         <div class="chart-card__svg-wrapper">{{ img_data | safe }}</div>
     ```
     Because this condition only checks whether the 4-character substring `'<svg'` appears *anywhere* in `img_data` and only blacklists `<script` and `<foreignobject`, any payload passed to `POST /api/generate-report` (when WeasyPrint is used) such as `<svg onload="..."><image href="..."/></svg>` or `<!-- <svg --> <iframe src="...">` bypasses Jinja2 autoescaping via `| safe`.
- **Remediation**:
  Wrap every interpolated variable (`filename`, `col`, `desc`, `golden.question`, `golden.sql_query`, `k`, `v`) with `html.escape(str(...))` inside `_generate_smart_dataset_answer`, `chat_with_report`, and `chat_stream_with_report`. Remove `"class"` from `ALLOWED_ATTR` in `ChatInterface.tsx`'s `DOMPurify.sanitize` configuration. In `report.html`, parse and sanitize SVG strings using an XML parser (`defusedxml`) that strips all event-handler attributes (`on*`), external `<use>`/`<image>` references, and non-SVG elements before rendering.

#### SEC-HIGH-04: Hardcoded Default SECRET_KEY, Unauthenticated Default Posture, and Client-Side VITE_API_KEY Exposure

- **Location**: `Backend/app/core/config.py` (lines 22–26), `render.yaml` (lines 12–48), and `Frontend/src/services/api.ts` (lines 48, 398).
- **Mechanism and Root Cause**:
  1. In `config.py`, `SECRET_KEY` defaults to the static string `"getreport-vault-secret-key-32b"`. In `render.yaml` (the production Render deployment blueprint), `SECRET_KEY` is completely omitted from `envVars`. Consequently, production deployments provisioned via `render.yaml` use the publicly known hardcoded repository secret to derive AES-256-GCM keys for the Snowflake credential vault!
  2. `REQUIRE_AUTH` defaults to `False` and `API_KEY` defaults to `""`. If an operator deploys the backend without setting `API_KEY`, `verify_api_key` returns `None` (authentication disabled), exposing file uploads, database queries, and `POST /api/jobs/{task_id}/sandbox-exec` to anonymous internet traffic.
  3. Even when `API_KEY` is configured, `Frontend/src/services/api.ts` reads `import.meta.env.VITE_API_KEY`. In Vite, all `VITE_*` variables are statically embedded in plaintext into the compiled JavaScript bundle served to every visitor's browser.
- **Remediation**:
  Add `SECRET_KEY` with `generateValue: true` to `render.yaml`, and in `config.py`, generate a cryptographically random ephemeral key via `secrets.token_urlsafe(32)` if `SECRET_KEY` is not set (or fail startup when `DATABASE_URL` is configured and `SECRET_KEY` is still the default). Replace the shared client-side `VITE_API_KEY` pattern with server-managed authentication sessions or OAuth/OIDC tokens.

#### SEC-HIGH-05: Denial of Service via Synchronous Join Explosion, Pre-Truncation DuckDB OOM, and Unregistered SlowAPIMiddleware

- **Location**: `Backend/app/api/routes/upload.py` (lines 166–240), `Backend/app/services/duckdb_engine.py` (lines 104–112), and `Backend/app/main.py` (lines 41–43).
- **Mechanism and Root Cause**:
  1. **Synchronous Event-Loop Blocking & Quadratic Join Explosion in `POST /api/upload/join`**:
     In `upload.py`, `upload_and_join_files` is declared as `async def`, meaning it executes directly on the main FastAPI event loop. Inside the loop (lines 208–218), it calls `load_dataframe(...)`, `join_datasets(...)`, and `joined_df.write_csv(...)` **synchronously** without `run_in_threadpool` or `asyncio.to_thread`. Worse, while `join_type == "cross"` is blocked, an `inner`, `left`, or `full` join on two 50,000-row CSV files where the join key column contains identical values (e.g., `id = 1` on every row) produces $50,000 \times 50,000 = 2.5 \times 10^9$ rows! This immediately freezes the main event loop and triggers an Out-Of-Memory (OOM) crash (`CWE-400`). The same synchronous event-loop blocking occurs in `GET /api/jobs/{task_id}/export/{export_format}` and `GET /api/jobs/{task_id}/export-gx` (`Backend/app/api/routes/report.py`, lines 331–340 and 434).
  2. **Pre-Truncation Full Materialization in DuckDB (`duckdb_engine.py`, lines 104–110)**:
     ```python
     result_pl = self.conn.execute(sql).pl()
     if max_rows is not None and max_rows > 0:
         result_pl = result_pl.head(min(max_rows, 1000))
     ```
     Calling `.pl()` on the DuckDB relation materializes the **entire** query result into a Polars DataFrame in memory *before* `.head()` is applied! If a user submits a multi-way self-join (`SELECT a.* FROM dataset a, dataset b, dataset c`) without a `LIMIT` clause, DuckDB and Polars allocate gigabytes of RAM before `.head(1000)` ever executes. (By contrast, `execute_parameterized_query` on line 206 appends `LIMIT {max_rows}` to the SQL query before execution, or `fetchmany(max_rows)` can be used on the DuckDB cursor.)
  3. **Missing `SlowAPIMiddleware` Registration (`main.py`, lines 41–43)**:
     `main.py` sets `app.state.limiter = limiter` and registers the exception handler, but never calls `app.add_middleware(SlowAPIMiddleware)`. In `slowapi`, `default_limits=["60/minute"]` is only enforced automatically if `SlowAPIMiddleware` is added to the FastAPI app. Consequently, the 15 endpoints across `status.py`, `report.py`, `snowflake.py`, and `endpoints.py` that lack an explicit `@limiter.limit(...)` decorator have **zero rate limiting**.
- **Remediation**:
  Register `SlowAPIMiddleware` in `main.py`; offload all synchronous Polars/DuckDB operations in `async def` routes via `await asyncio.to_thread(...)`; enforce a pre-join duplicate-key cardinality check or output row cap in `join_datasets`; and append `SELECT * FROM ({sql}) LIMIT {max_rows}` or use `.fetch_arrow_table(rows_per_batch=max_rows)` in `DuckDBAnalyticalSession.execute_read_query`.


### Medium and Low-Severity Vulnerabilities

#### SEC-MED-01: Rate-Limit Bypass via X-Forwarded-For Spoofing (Backend/app/core/limiter.py, lines 15–46)
When the backend runs inside a container behind a cloud reverse proxy (such as Render or Docker bridge networking), `request.client.host` is a private IP address (`10.x.x.x` or `172.x.x.x`), so `is_trusted_proxy` evaluates to `True`. `_get_real_client_ip` then extracts `candidate = xff.split(",")[0].strip()`—the **left-most** address in `X-Forwarded-For`. Standard reverse proxies *append* the connecting client's IP to the *end* of any client-supplied `X-Forwarded-For` header (`X-Forwarded-For: <spoofed_ip>, <real_client_ip>`). By reading index `[0]`, an attacker can send a random `X-Forwarded-For` IP on every request and completely bypass all `slowapi` rate limits (`CWE-348`). When behind a single trusted reverse proxy that appends to `X-Forwarded-For`, the limiter must read the **right-most** untrusted IP (`xff.split(",")[-1].strip()`).

#### SEC-MED-02: Snowflake Session Token Exposed in GET Query Strings (Backend/app/api/routes/snowflake.py, lines 142, 160, 179)
Whereas the team specifically fixed `VULN-02` (`CWE-598`) for SSE streams by introducing stream tickets to avoid passing `api_key` in GET URLs, `GET /api/snowflake/databases`, `GET /api/snowflake/schemas`, and `GET /api/snowflake/objects` require the active `session_id` as a URL query parameter (`?session_id=...`). This exposes the live Snowflake session bearer token in plaintext across web server access logs, proxy logs, browser history, and APM telemetry. `session_id` should be transmitted via a custom request header (`X-Snowflake-Session-Id`).

#### SEC-MED-03: Incomplete Formula Injection (CWE-1236) Sanitization in SQL & DAG CSV Exports
While `BaseDataExporter._neutralize_formulas` (`Backend/app/core/factories/exporters.py`) properly prefixes `=`, `+`, `-`, `@`, `	`, and `
` with a single quote `'` for full dataset exports, two other CSV export paths bypass this protection:
1. `downloadSqlCsv` in `Frontend/src/components/ReportGeneration.tsx` (lines 380–392) serializes DuckDB query results to CSV in the browser without neutralizing formula trigger characters and without quoting column headers (`cols.join(",")`).
2. `TransformationDAG.to_audit_csv()` (`Backend/app/services/transformation_dag.py`, lines 384–410) calls `df.write_csv()` directly without neutralizing `Target Column` names (which can be user-supplied via `/concepts/derive`).

#### SEC-MED-04: Cross-Request Matplotlib Global State Race Condition (Backend/app/services/sandboxed_analyst.py & visualization.py)
Both `SandboxedAnalystAgent.execute_polars` (`sandboxed_analyst.py`, lines 270, 315, 353) and `visualization.py` (lines 52, 63) use Matplotlib's global state machine (`plt.gcf()`, `plt.close("all")`). Because FastAPI runs threadpool tasks concurrently across requests, Thread A calling `plt.close("all")` at the end of `_fig_to_base64` will destroy Thread B's active figure mid-render, or Thread A calling `plt.gcf()` in `sandboxed_analyst.py` can capture and return a chart generated by Thread B (`CWE-362`). Charts should be created using the object-oriented `matplotlib.figure.Figure()` API rather than the global `pyplot` state machine.

#### SEC-MED-05: Unsanitized Content-Disposition Header in POST /api/generate-report (Backend/app/api/routes/report.py, line 301)
In `generate_report_direct`, `headers={"Content-Disposition": f"attachment; filename={body.filename}"}` interpolates `body.filename` without calling `_sanitize_download_filename(body.filename)`, allowing header parameter injection (`filename="malicious.exe"; foo=bar`) if untrusted input reaches this endpoint.

#### SEC-LOW-01 & SEC-LOW-02: Unverified joblib.load Deserialization & chmod 777 in Dockerfile
- `ModelArtifactSerializer.load_artifact` (`Backend/app/services/ml_production/serializer.py`, line 48) invokes `joblib.load(artifact_path)`, which uses Python's `pickle` module (`CWE-502`). If wired to untrusted files, it allows arbitrary code execution; artifacts should be signed with HMAC-SHA256 using `SECRET_KEY` before loading.
- `Backend/Dockerfile` (line 36) executes `chmod -R 777 /app/temp_uploads /app/outputs /app/temp_cache /app/data`. Because line 42 already assigns ownership to `appuser:appuser` (`chown -R appuser:appuser /app`), `chmod 750` should be used instead of world-writable `777` permissions (`CWE-732`).

## Detailed Codebase Walkthrough by Subsystem

### Backend Core, Configuration, and Database Layer

An examination of `Backend/app/core/` and `Backend/app/db.py` highlights a tension between cloud-native aspiration and local-first prototyping. `Settings` in `Backend/app/core/config.py` uses `pydantic-settings` to centralize configuration, and `main.py` disables Swagger/OpenAPI documentation endpoints (`docs_url=None, redoc_url=None, openapi_url=None`) to reduce reconnaissance surface in production. However, several operational edge cases require attention:

| File Path | Key Components | Positive Design Traits | Identified Flaws and Edge Cases |
| :--- | :--- | :--- | :--- |
| `Backend/app/main.py` | `lifespan`, `CORSMiddleware`, `/health` | Closes sync DB pool after schema init; runs 24h file cleanup on startup; boundarys Vercel preview regex to project owner | Omits `SlowAPIMiddleware`; logs a warning when `SECRET_KEY` is default in production instead of refusing startup or generating a random secret |
| `Backend/app/core/auth.py` | `verify_api_key`, `create_stream_ticket`, `check_task_ownership` | Uses `secrets.compare_digest` for constant-time key comparison; provides single-use stream tickets for SSE | `_stream_tickets` is stored in process-local memory (`dict`), failing when `WEB_CONCURRENCY > 1`; `check_task_ownership` trusts unsigned `X-User-Id` headers |
| `Backend/app/core/limiter.py` | `_get_real_client_ip`, `limiter` | Validates IP format via `ipaddress.ip_address`; falls back to `memory://` if Redis is unreachable | Reads `xff.split(",")[0]` instead of `[-1]`; pings Redis synchronously at module import time |
| `Backend/app/db.py` | `PostgresConnection`, `AsyncPostgresConnection`, `init_db` | Pre-pings connections (`SELECT 1`) with 3 retries to survive Neon Serverless SSL idle disconnects; enables SQLite `WAL` mode | Custom regex `?` placeholder rewriting can corrupt SQL strings if complex escaping is used; `AsyncPostgresConnection.commit()` is a no-op; coexists with dead `Backend/app/db/` package |
| `Backend/app/core/task_dispatcher.py` | `dispatch_task`, `dispatch_task_sync` | Gracefully degrades from ARQ Redis pool to `asyncio.create_task` or daemon threads | `_redis_available` is cached globally forever as a boolean; if Redis has a transient 5-second blip at startup, the process permanently disables ARQ until restart |

In `Backend/app/core/task_dispatcher.py` (lines 32–61), caching `_redis_available = False` permanently on the first failed ping means a transient network hiccup between Render and Redis during container boot permanently degrades the web service to running heavy CPU/RAM analysis jobs inside the web process for the entire lifetime of the container. Storing a failure timestamp and retrying Redis connectivity every 60 seconds would make the dispatcher self-healing.

### Data Ingestion, Cleaning, and Row Intelligence Pipeline

The data processing pipeline in `Backend/app/services/data_processing.py` (1,622 lines) is the workhorse of the backend. It coordinates file loading, header sanitization, currency/percentage string coercion, phone number standardization, row triage, missing value imputation, outlier handling, and `TransformationDAG` lineage recording.

While the algorithmic flow is well-conceived, `data_processing.py` has grown into a 1,622-line monolith that mixes low-level archive validation (`_validate_zip_bomb`, `_validate_gzip_bomb`), CSV dialect sniffing (`_detect_csv_parameters`), phone number regex heuristics (`_is_phone_column`), MICE iterative imputation (`impute_multivariate_mice`), dataset joining (`join_datasets`), and CSV export sanitization (`sanitize_df_for_csv_export`—which duplicates `BaseDataExporter._neutralize_formulas` in `Backend/app/core/factories/exporters.py`).

Additionally, in `_sanitize_and_coerce_df` (`Backend/app/services/data_processing.py`, lines 304–418), every column header is automatically converted to `snake_case` immediately upon loading (`load_dataframe`). Because `load_dataframe` is called both during Phase 1 (`inspect_file_task`), Phase 2 (`clean_data_task`, `analyze_data_task`), and during exports (`export_cleaned_data_or_report`), the column renaming happens silently inside `load_dataframe` before `clean_data` is ever called! As a result, when `clean_data` later checks whether columns need to be renamed to `snake_case` to populate `cleaning_report.columns_renamed`, the columns are *already* in `snake_case` from `load_dataframe`, causing `cleaning_report.columns_renamed` to under-report the actual header transformations performed during ingestion.

### Conversational RAG, Knowledge Graph, and LLM Orchestration

The conversational subsystem spans `Backend/app/services/rag_service.py` (1,134 lines), `Backend/app/core/rag_utils.py` (574 lines), `Backend/app/core/graph_store.py` (223 lines), `Backend/app/services/dataset_graph_builder.py` (288 lines), `Backend/app/services/query_router.py` (198 lines), and `Backend/app/services/llm_finding.py` (492 lines).

The retrieval architecture is impressively sophisticated:
- `TableSemanticChunker` (`rag_utils.py`, lines 129–303) binds column headers directly to cell values (`- Header: Value`) so table chunks never lose their schema context during vector embedding.
- `DatasetGraphStore` (`graph_store.py`) and `build_dataset_graph` (`dataset_graph_builder.py`) construct a `networkx.MultiDiGraph` linking `Dataset`, `FeatureColumn`, and `DataQualityIssue` nodes with `CONTAINS_COLUMN`, `HAS_QUALITY_ISSUE`, and `CORRELATED_WITH` edges without spending a single LLM token.
- `QueryRouter` (`query_router.py`) classifies incoming questions into `analytical_sql`, `graph_local`, `graph_global`, or `hybrid` modes, optionally accelerating classification via TypeSafe AI's Jev model (`typesafe_service.py`).
- `PostgresVectorStore.hybrid_search_async` (`rag_utils.py`, lines 409–482) combines `pgvector` HNSW cosine distance (`<=>`) with PostgreSQL full-text search (`to_tsvector` / `websearch_to_tsquery`) using Reciprocal Rank Fusion ($\text{RRF} = \frac{1}{60 + r_v} + \frac{1}{60 + r_k}$), followed by second-stage reranking in `CrossEncoderReranker`.

However, in addition to the 2,000-character truncation bug in `ingest_report` (`SecurityGuard.sanitize_input`), two further issues affect `rag_service.py`:
1. **`TableSemanticChunker.chunk_dataset_profile` Is Never Called in Production**: Although `TableSemanticChunker.chunk_dataset_profile` (`rag_utils.py`, lines 195–303) was specifically written to transform structured profiling dicts and `IssueLedger` items into rich metadata-tagged entity chunks, `tasks.py` and `tasks_arq.py` instead call `build_rag_narrative()` to generate a flat string and pass that string to `ingest_report_blocking()`, which only calls `self.text_splitter.split_text(text_content)`! Thus, `chunk_dataset_profile` is only ever called in unit tests, never in the actual application pipeline.
2. **Hardcoded `vector(1536)` Dimension Mismatch in PostgreSQL (`Backend/db_schema.sql` line 29 & `Backend/app/db.py` line 336)**:
   The PostgreSQL `document_chunks` table defines `embedding vector(1536)`. While OpenAI's `text-embedding-3-small` outputs 1,536 dimensions, Google's `gemini-embedding-001` / `text-embedding-004` (which is the **primary default embedding model** when `GEMINI_API_KEY` is configured in `rag_service.py`, line 545!) outputs **768 dimensions** (or 3,072 dimensions depending on configuration)! When `PostgresVectorStore.add_texts_async` attempts to insert a 768-dimensional Gemini embedding into a `vector(1536)` column in PostgreSQL, `pgvector` raises a dimension mismatch error (`expected 1536 dimensions, not 768`), causing every Gemini-backed PostgreSQL vector insert to fail and fall back to in-memory TF-IDF!

### Document Compilation Engines (Typst, WeasyPrint, ReportLab)

The PDF generation layer (`Backend/app/core/factories/reports.py`, `report_typst.py`, `report_generator.py`, `report_sections.py`, `report_weasyprint.py`, `report_renderer.py`) provides three distinct rendering engines. Using Typst (`typst>=0.15.0`) as the primary engine is an excellent architectural choice for memory-constrained cloud containers (512 MB Render instances), as Typst's Rust compiler renders complex multi-page tables and vector charts in milliseconds with a fraction of the RAM required by Chromium or WeasyPrint/Cairo.

However, the data contract passed into the three engines is inconsistent:
- In `Backend/app/api/routes/report.py`, `POST /api/jobs/{task_id}/report` dispatches `app.tasks.generate_pdf`, which passes `analysis_data` with `metadata = result["analysis"]["metadata"]`.
- By contrast, `GET /api/jobs/{task_id}/report/full` (`report.py`, lines 167–172) constructs `params` by overwriting `params["metadata"] = analysis.get("info")`! Because `job.result["info"]` (`get_dataset_info(df)`) has keys like `"rows"` and `"columns"` (a list of column names), whereas `job.result["analysis"]["metadata"]` has `"total_rows"`, `"total_columns"` (an integer), `"numeric_columns"`, and `"missing_value_pct"`, calling `/report/full` passes a incompatible `metadata` dictionary structure into `generate_pdf_report`, causing missing metrics in the rendered PDF header!

### Frontend Application and Real-Time Synchronization

The frontend (`Frontend/src/`) is organized around React Router v6 lazy-loaded routes (`App.tsx`), a centralized fetch/SSE/WebSocket client (`services/api.ts`), a multi-transport task synchronization hook (`hooks/useTaskStatus.ts`), and domain components (`components/`).

A subtle synchronization flaw exists in `Frontend/src/pages/Workspace.tsx` (lines 31–44) and `Frontend/src/components/FileUpload.tsx` (lines 58–108):
- During `step === "upload"`, `FileUpload` mounts and calls `useTaskStatus(taskId)`. When Stage 1 inspection finishes (`WAITING_FOR_USER`), the user reviews the `IssueLedger` and clicks "Proceed", which triggers Stage 2 (`POST /api/jobs/{task_id}/analyze`).
- `FileUpload` then waits until `normalizedStatus === "COMPLETED"` before calling `onFileUploaded(taskResult, taskId)`, which transitions `Workspace` from `step = "upload"` to `step = "preview"`.
- That means by the time the user reaches `step === "preview"` and clicks "Generate Report" (transitioning `Workspace` to `step === "generating"`), the backend job is **already `COMPLETED`**!
- Yet when `step === "generating"` begins, `ReportGeneration.tsx` (lines 184–235) calls `await api.generatePersistentReport(taskId)` (`POST /api/jobs/{task_id}/report`), which triggers `generate_pdf_task` a **second time** even though `resume_analysis_pipeline_task` (`tasks_arq.py`, line 305) *already compiled and saved the exact same PDF report* at the end of Stage 2! Avoiding this redundant re-compilation when `report_status === "ready"` or `report_download_url` is already present would make report viewing instantaneous.

## Prioritized Improvement Roadmap and Remediation Plan

To elevate GetReport from a feature-rich prototype into a hardened, production-grade enterprise platform, engineering efforts should be sequenced across four structured phases addressing critical security exposures, functional bugs, architectural consolidation, and frontend/CI maturity.

| Priority Tier | Target Subsystem | Actionable Engineering Remediation | Expected Impact and Verification Metric |
| :--- | :--- | :--- | :--- |
| **P0: Immediate (24–48h)** | Execution Sandboxes (`sandboxed_analyst.py`, `concept_synthesizer.py`, `issue_ledger.py`) | Replace `exec()` in `issue_ledger.py` with `remediation_registry`; enforce attribute-level AST whitelisting in `concept_synthesizer.py`; isolate `SandboxedAnalystAgent` in a restricted subprocess and block module graph traversal (`plt.sys`, function aliasing) | Eliminates all 3 Critical RCE / sandbox escape vectors (`SEC-CRIT-01`, `SEC-CRIT-02`, `SEC-CRIT-03`) |
| **P0: Immediate (24–48h)** | SQL & Auth Boundaries (`snowflake/driver.py`, `auth.py`, `config.py`, `render.yaml`) | Escape/validate `from_last` in `discover_objects`; generate cryptographic `SECRET_KEY` in `render.yaml` and `config.py`; bind task/session ownership to server-signed tokens rather than raw `X-User-Id` headers | Eliminates SQLi (`SEC-HIGH-01`), BOLA/tenant spoofing (`SEC-HIGH-02`), and static vault key derivation (`SEC-HIGH-04`) |
| **P1: Short-Term (Week 1)** | RAG & LLM Pipeline (`rag_service.py`, `llm_finding.py`, `db_schema.sql`) | Remove `SecurityGuard.sanitize_input` 2,000-char truncation from `ingest_report`; wire `TableSemanticChunker.chunk_dataset_profile`; remove non-existent model IDs (`gemini-3.7-flash`); call `await generate_findings()` natively instead of `asyncio.run()`; HTML-escape all fallback and Golden Query outputs | Restores 100% RAG narrative indexing, cuts LLM latency by 1.5–4.0s, fixes event-loop crash on 2nd run, and neutralizes Stored XSS (`SEC-HIGH-03`) |
| **P1: Short-Term (Week 1)** | Event-Loop & DoS Hardening (`main.py`, `upload.py`, `duckdb_engine.py`, `limiter.py`) | Register `SlowAPIMiddleware`; wrap sync Polars/DuckDB calls in `asyncio.to_thread`; cap join output rows in `join_datasets`; push `LIMIT` into DuckDB SQL before `.pl()`; read `xff.split(",")[-1]` in rate limiter | Prevents OOM container crashes (`SEC-HIGH-05`) and rate-limit evasion (`SEC-MED-01`) |
| **P2: Medium-Term (Weeks 2–3)** | Backend Consolidation (`tasks.py`, `report_renderer.py`, `db/`, `templates/`) | Remove legacy Celery `tasks.py`/`celery_app.py`, dead `app/db/` SQLAlchemy stubs, duplicate `report_renderer.py`, duplicate `app/services/templates/`, and root scratch files (`row_intelligence.patch`, `Design_MD_Summary.json`) | Reduces backend maintenance surface by ~2,500 lines of dead/duplicated code and eliminates template divergence |
| **P2: Medium-Term (Weeks 2–3)** | Thread-Safe Rendering & Snowflake Pooling (`visualization.py`, `snowflake/driver.py`) | Migrate Matplotlib chart generation from global `plt.gcf()` to isolated `Figure()` instances; use `EncryptedSessionVault` connection pool in `RealSnowflakeDriver` and close cursors/connections via context managers | Eliminates cross-tenant chart race conditions (`SEC-MED-04`) and stops 1-second Snowflake re-authentication churn |
| **P3: Long-Term (Month 1)** | Frontend Decomposition & CI Gate Enforcement (`ReportGeneration.tsx`, `ci.yml`, `doctor.config.json`) | Split `ReportGeneration.tsx` (2,190 lines) and `SnowflakeConnectModal.tsx` (1,269 lines) into focused sub-components; add Vitest + Testing Library component tests; enable `tsc --noEmit`, `eslint`, `oxlint`, and `mypy` in `.github/workflows/ci.yml` | Enforces continuous type safety, lint hygiene, and regression coverage across both frontend and backend |

### Concrete Code-Level Remediation Blueprints

#### 1. Hardening the Concept Synthesizer and Eliminating exec() in Issue Ledger
In `Backend/app/services/concept_synthesizer.py`, `validate_expression_ast` must validate all attribute accesses on `pl` regardless of whether they occur directly inside an `ast.Call` node, and block comprehensions and subscript invocations:
```python
elif isinstance(node, ast.Attribute):
    if node.attr.startswith("__") or node.attr in cls.FORBIDDEN_ATTRIBUTES:
        raise ConceptSynthesizerSecurityViolation(f"Accessing private attribute '{node.attr}' is prohibited.")
    if node.attr in cls.FORBIDDEN_EXPR_METHODS:
        raise ConceptSynthesizerSecurityViolation(f"Accessing method '{node.attr}' is prohibited.")
    if isinstance(node.value, ast.Name) and node.value.id == "pl":
        if node.attr not in cls.ALLOWED_POLARS_ROOTS:
            raise ConceptSynthesizerSecurityViolation(
                f"Polars attribute 'pl.{node.attr}' is not permitted in concept expressions."
            )
```
In `Backend/app/services/issue_ledger.py`, replace `exec(issue.fix_code, safe_globals, loc_env)` with declarative dispatch through `Backend/app/core/factories/remediation.py`, and ensure any displayed `fix_code` string uses `repr(col)` rather than unescaped `'{col}'`.

#### 2. Fixing the RAG Truncation Bug and Async Event-Loop Lifecycle
In `Backend/app/services/rag_service.py` (lines 569 and 605), separate query sanitization (which caps user questions at 2,000 characters) from document ingestion sanitization (which should strip null/control characters without truncating document length):
```python
@staticmethod
def sanitize_document(text: str, max_length: int = 250_000) -> str:
    if not text:
        return ""
    text = text[:max_length]
    return "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch >= " ").strip()
```
In `Backend/app/tasks_arq.py` (line 277), replace `await asyncio.to_thread(generate_findings_sync, analysis_result)` with a direct native coroutine call `await generate_findings(analysis_result)` (importing `generate_findings` from `app.services.llm_finding`), keeping all `AsyncOpenAI` (`httpx.AsyncClient`) network I/O on the main worker event loop and preventing `RuntimeError: Event loop is closed`.

#### 3. Replacing Global Matplotlib pyplot State with Isolated Figure Instances
In both `Backend/app/services/visualization.py` and `Backend/app/services/sandboxed_analyst.py`, replace `plt.figure()` / `plt.gcf()` / `plt.close("all")` with thread-local or explicitly instantiated `matplotlib.figure.Figure` objects rendered via `FigureCanvasAgg`:
```python
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

fig = Figure(figsize=(8, 4), dpi=CHART_DPI)
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)
```
This guarantees complete thread isolation when multiple chart generation tasks or analyst sandboxes execute concurrently in FastAPI's threadpool.


## References and Standards Cited

- [FastAPI Official Documentation — Lifespan Events, Security Dependencies, and Middleware](https://www.google.com/search?q=https%3A%2F%2Ffastapi.tiangolo.com%2Fadvanced%2Fmiddleware%2F)
- [Polars User Guide — Lazy API, Streaming Execution, and Expressions](https://www.google.com/search?q=https%3A%2F%2Fdocs.pola.rs%2Fuser-guide%2Fconcepts%2Flazy-api%2F)
- [DuckDB Documentation — Securing DuckDB and enable\_external\_access Configuration](https://www.google.com/search?q=https%3A%2F%2Fduckdb.org%2Fdocs%2Foperations_manual%2Fsecuring_duckdb%2Foverview.html)
- [Typst Documentation — Rust Compilation Engine and Scripting](https://www.google.com/search?q=https%3A%2F%2Ftypst.app%2Fdocs%2F)
- [MITRE CWE-79: Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F79.html)
- [MITRE CWE-89: Improper Neutralization of Special Elements used in an SQL Command (SQL Injection)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F89.html)
- [MITRE CWE-94: Improper Control of Generation of Code (Code Injection)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F94.html)
- [MITRE CWE-113: Improper Neutralization of CRLF Sequences in HTTP Headers (HTTP Response Splitting)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F113.html)
- [MITRE CWE-287: Improper Authentication](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F287.html)
- [MITRE CWE-348: Use of Less Trusted Source (X-Forwarded-For IP Spoofing)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F348.html)
- [MITRE CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization (Race Condition)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F362.html)
- [MITRE CWE-400: Uncontrolled Resource Consumption (Denial of Service)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F400.html)
- [MITRE CWE-502: Deserialization of Untrusted Data](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F502.html)
- [MITRE CWE-598: Use of GET Request Method With Sensitive Query Strings](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F598.html)
- [MITRE CWE-639: Authorization Bypass Through User-Controlled Key (BOLA / IDOR)](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F639.html)
- [MITRE CWE-732: Incorrect Permission Assignment for Critical Resource](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F732.html)
- [MITRE CWE-798: Use of Hard-coded Credentials](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F798.html)
- [MITRE CWE-1236: Improper Neutralization of Formula Elements in a CSV File](https://www.google.com/search?q=https%3A%2F%2Fcwe.mitre.org%2Fdata%2Fdefinitions%2F1236.html)
