# GetReport: Complete Architectural & Technical System Report

**Repository:** `GetReport` (`d:\Git_Repo\GetReport`)  
**Product Classification:** Enterprise Data Profiling, Forensic Quality Auditing, Automated Remediation & Executive Intelligence Platform  
**Target Environments:** Self-hosted On-Premise, Private Cloud (AWS/GCP/Azure), Kubernetes, Snowflake Native  
**Document Classification:** Grounded Technical Documentation (Zero-Hallucination Blueprint)  

---

## 1. Executive Summary: What GetReport Actually Does

**GetReport** is an enterprise data audit, forensic quality assurance, and automated intelligence platform. It converts messy, untrusted raw tabular data (CSV, Parquet, Excel, TSV, JSON, or Snowflake cloud data warehouse queries) into audit-ready, statistically verified datasets accompanied by publication-grade vector PDF executive reports, interactive knowledge graphs, and conversational RAG assistants.

### Core Problems Solved
1. **The "Silent Data Corruption" Crisis:** Modern analytical and ML pipelines often fail because of subtle data quality issues—such as Benford's Law distribution anomalies (indicating synthetic or manipulated transactions), disguised null placeholders (`"N/A"`, `"-999"`, `"None"`), semantic type mismatches (dates stored as integers, timestamps as strings), and structural changepoints. GetReport catches these before data reaches models or dashboards.
2. **The "Black-Box" Data Cleaning Problem:** Traditional automated data cleaners silently drop or mutate rows, destroying historical audit trails. GetReport enforces **Row Intelligence** and a **Human-in-the-Loop Issue Ledger**: every single row receives an immutable verdict (`PROTECT`, `REVIEW`, `FIX`, or `EXCLUDE`), and cleaning transformations are logged as a directed acyclic graph (Transformation DAG).
3. **The Memory & Scaling Wall:** Profiling massive multi-gigabyte datasets with Pandas frequently causes Out-of-Memory (OOM) fatal crashes. GetReport replaces Pandas with **Polars** (Apache Arrow zero-copy streaming columnar engine) and **DuckDB** (embedded in-process vectorized OLAP engine), processing millions of records in sub-second streaming batches with bounded RAM.
4. **Fragile PDF Report Generation:** Traditional HTML-to-PDF converters (WeasyPrint, Chromium/Puppeteer) consume over 500 MB of RAM per job and crash inside resource-constrained Docker containers. GetReport compiles reports using **Typst**—a native Rust declarative typesetting engine that executes in less than 25 MB RAM and compiles in sub-100ms.

---

## 2. End-to-End Operational Pipeline: Step-by-Step Workflow

GetReport operates on a two-phase architecture designed around human oversight and asynchronous task dispatching.

```
                              GETREPORT COMPLETE DATA LIFECYCLE
                              
  [User Ingestion]                 [Phase 1: Inspection & Triage]           [Human-in-the-Loop Gate]
┌──────────────────┐             ┌────────────────────────────────┐       ┌────────────────────────┐
│ • File Upload    │────────────▶│ Polars Zero-Copy Load          │──────▶│ Issue Ledger Modal     │
│   (CSV, Parquet, │             │ • Structural Junk Detection    │       │ • Approve / Reject /   │
│   Excel, TSV)    │             │ • Disguised Null Identification│       │   Modify Fixes         │
│ • Snowflake Live │             │ • Smart Schema Inference       │       │ • Lock DAG & Rules     │
│   Warehouse DDL  │             │ • Row Triage (Verdict Matrix)  │       └───────────┬────────────┘
└──────────────────┘             └────────────────────────────────┘                   │
                                                                                      ▼
  [Report & Delivery]              [Parallel Synthesis Engine]             [Phase 2: Execution]
┌──────────────────┐             ┌────────────────────────────────┐       ┌────────────────────────┐
│ • Typst Vector   │◀────────────│ • Concurrent Matplotlib Charts │◀──────│ • Execute Approved DAG │
│   PDF Export     │             │ • LLM Multi-Turn Insights      │       │ • Polars Deep Profiling│
│ • KnowledgeGraph │             │   (Gemini 2.5 Flash / OpenAI)  │       │ • CUSUM & Benford      │
│   (Graphify/UA)  │             │ • Knowledge Graph Assembly     │       │ • Outliers & Skewness  │
│ • LightRAG Chat  │             │ • LightRAG Prose Vectorization │       │ • ML Readiness Score   │
└──────────────────┘             └────────────────────────────────┘       └────────────────────────┘
```

### Phase 1: Ingestion, Structural Scrubbing & Issue Detection (`inspect_file_task`)
* **Step 1: Ingestion & Format Normalization**
  - The client uploads a dataset via `POST /api/upload` or submits a Snowflake SQL query via `POST /api/snowflake/execute-query`.
  - The file is saved to content-addressable storage (`LocalStorageProvider` or S3) with an MD5/SHA-256 hash.
  - The file format is validated using MIME sniffing and magic bytes (rejecting corrupted archives or nested compression bombs).
* **Step 2: Polars Streaming Ingestion**
  - [data_processing.py](file:///d:/Git_Repo/GetReport/Backend/app/services/data_processing.py) loads the file using Polars streaming (`pl.scan_csv`, `pl.read_parquet`, or `fastexcel`).
  - Zero-copy columnar memory mapping ensures minimal memory overhead.
* **Step 3: Row Intelligence & Non-Data Row Stripping**
  - [row_intelligence.py](file:///d:/Git_Repo/GetReport/Backend/app/services/row_intelligence.py) runs regex-based structural scanning on leading, middle, and trailing rows.
  - It identifies and flags non-data junk: repeated table headers, decorative separator lines, footer notes, and subtotal/grand-total summary lines (e.g. `Total: $450,000`). Stripping these prevents distribution metrics from becoming artificially skewed.
* **Step 4: Smart Schema Inference & Disguised Null Detection**
  - [smart_schema.py](file:///d:/Git_Repo/GetReport/Backend/app/services/smart_schema.py) scans string columns for disguised null placeholders (`"N/A"`, `"-999"`, `"null"`, `"None"`, `"???"`, `"#"`, `"TBD"`).
  - It tests candidate type casts against regex patterns for ISO datetimes, international currencies, phone numbers, and decimal integers.
* **Step 5: Issue Ledger Creation & Pause**
  - [issue_ledger.py](file:///d:/Git_Repo/GetReport/Backend/app/services/issue_ledger.py) constructs an `IssueLedger` containing all detected anomalies categorized by severity (`critical`, `high`, `medium`, `low`).
  - The job status in SQLite transitions to `TaskStatus.WAITING_FOR_USER`.
  - The frontend displays the interactive **Issue Ledger** where engineers approve, reject, or customize proposed cleaning rules.

### Human-in-the-Loop Approval Gate
* The user approves or rejects specific transformations (e.g., approve dropping empty columns, reject trimming outliers, modify mode imputation to median imputation).
* The user clicks **Lock Issues & Start Deep Analysis**, which triggers `POST /api/jobs/{task_id}/analyze` with the approved rule dictionary.

### Phase 2: Transformation, Profiling, Synthesis & Delivery (`resume_analysis_pipeline_task`)
* **Step 6: Transformation DAG Execution**
  - [transformation_dag.py](file:///d:/Git_Repo/GetReport/Backend/app/services/transformation_dag.py) executes the approved cleaning rules deterministically.
  - Every transformation records its input row count, output row count, modified cell count, execution timestamp, and reversibility signature.
  - The cleaned DataFrame is persisted to disk as an intermediate Apache Parquet artifact.
  - [comparison.py](file:///d:/Git_Repo/GetReport/Backend/app/services/comparison.py) computes a delta report (`raw_df` vs. `cleaned_df`).
* **Step 7: Deep Statistical Profiling (Polars Vectorized Expressions)**
  - [analysis/statistics.py](file:///d:/Git_Repo/GetReport/Backend/app/services/analysis/statistics.py) computes mean, standard deviation, variance, min, max, median, quartiles (Q1, Q3), IQR, skewness, and kurtosis in a single vectorized Polars pass.
  - [confidence_scoring.py](file:///d:/Git_Repo/GetReport/Backend/app/services/confidence_scoring.py) calculates pillar scores (Completeness, Consistency, Validity, Stability) and assigns overall letter grades ($A$ through $F$).
  - [analysis/correlations.py](file:///d:/Git_Repo/GetReport/Backend/app/services/analysis/correlations.py) runs Pearson, Spearman, and $\phi_K$ correlation matrices.
  - [analysis/outliers.py](file:///d:/Git_Repo/GetReport/Backend/app/services/analysis/outliers.py) runs Hubert-Vandervieren skewness-adjusted boxplot bounds, modified Z-scores (MAD), and multivariate Isolation Forest.
  - [analysis/ml_readiness.py](file:///d:/Git_Repo/GetReport/Backend/app/services/analysis/ml_readiness.py) evaluates feature variance, target leakage, class imbalance, and encoding complexity to yield an ML Readiness Score (0–100%).
* **Step 8: Concurrent Visualization & LLM Insight Generation**
  - Using Python's `asyncio.gather`:
    - **Worker A ([visualization.py](file:///d:/Git_Repo/GetReport/Backend/app/services/visualization.py)):** Generates headless, high-DPI distribution histograms, correlation heatmaps, boxplots, and time-series trendlines.
    - **Worker B ([llm_insight.py](file:///d:/Git_Repo/GetReport/Backend/app/services/llm_insight.py)):** Packages the statistical profile into a structured prompt and queries Google Gemini 2.5 Flash (with OpenAI and OpenRouter fallback) to generate executive business insights, key risks, and strategic next steps.
* **Step 9: Knowledge Graph Assembly & Clustering**
  - [dataset_graph_builder.py](file:///d:/Git_Repo/GetReport/Backend/app/services/dataset_graph_builder.py) and [core/graph_store.py](file:///d:/Git_Repo/GetReport/Backend/app/core/graph_store.py) assemble an in-memory `MultiDiGraph`:
    - Nodes for the Dataset, Columns, Missingness Anomalies, Benford Outliers, PII Vulnerabilities, and SQL Lineage entities.
    - Graphify modularity clustering (Leiden / Louvain) computes community partitions, hub labels, cohesion scores, and "God Nodes" (high-degree structural anchors).
    - Understand-Anything layer detection assigns nodes to 6 architectural data layers.
    - Semantic SHA-256 fingerprinting captures structural state to measure future schema drift.
* **Step 10: Vector PDF Report Compilation**
  - [report_typst.py](file:///d:/Git_Repo/GetReport/Backend/app/services/report_typst.py) injects metrics, tables, and generated SVG/PNG charts into the Rust Typst template ([audit_report.typ](file:///d:/Git_Repo/GetReport/Backend/app/templates/audit_report.typ)).
  - Compiles the final report into a publication-grade PDF in sub-100ms.
* **Step 11: Background LightRAG Ingestion**
  - The analysis metrics, cleaning transformations, and statistical summaries are translated into structured English prose.
  - Ingested asynchronously into a hybrid TF-IDF + Cosine embedding vector store and connected to the Knowledge Graph for interactive natural-language chat.

---

## 3. Comprehensive Catalog of Algorithms

GetReport relies on a diverse set of statistical, machine learning, graph theoretical, and information retrieval algorithms. Every algorithm is executed in-process with zero external API fees (except optional LLM summary generation).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        GETREPORT COMPLETE ALGORITHM TAXONOMY                          │
├──────────────────────────┬─────────────────────────────────────────────────────────────┤
│ Domain                   │ Algorithm & Mathematical Specification                      │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Statistical Moments      │ Streaming Welford's Algorithm (Mean, Variance, Skew, Kurt) │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Outlier & Anomaly Bounds │ • Hubert-Vandervieren Skewness-Adjusted Octile Fences       │
│                          │ • Log-Transformed Tukey Fences (Right-Skewed Pareto)        │
│                          │ • Modified Z-Score using Median Absolute Deviation (MAD)    │
│                          │ • Multivariate Isolation Forest (iForest Tree Partitioning) │
│                          │ • Local Outlier Factor (LOF k-nearest density ratio)        │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Forensic Quality         │ • Benford's Law (Log10 first-digit Chi-Square Goodness-of-Fit)│
│                          │ • Population Stability Index (PSI relative entropy)         │
│                          │ • Shannon Information Entropy & Simpson Diversity Index     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Time Series Intelligence │ • CUSUM (Cumulative Sum) Log-Likelihood Ratio Tests         │
│                          │ • Holt-Winters Exponential Smoothing with Confidence Bounds │
│                          │ • Mann-Kendall Non-Parametric Rank Trend Test               │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Non-Linear Association   │ • Phi-K (ϕ_K) Bivariate Normal Dispersion Correlation       │
│                          │ • Spearman Rank Order Correlation                           │
│                          │ • Pearson Product-Moment Correlation                        │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Graph & Topology         │ • Leiden / Louvain Modularity Maximization Clustering       │
│                          │ • Hub-Based Centrality Community Labeling                   │
│                          │ • Degree Centrality & Cut-Vertex God Node Detection         │
│                          │ • Cross-Community Surprising Edge Discovery                 │
│                          │ • Bidirectional Breadth-First-Search (BFS) Causal Paths     │
├──────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Conversational RAG       │ • Reciprocal Rank Fusion (RRF: k=60)                        │
│                          │ • Sparse BM25 / Sublinear TF-IDF Vectorization              │
│                          │ • Dense Cosine Similarity Semantic Search                   │
│                          │ • LightRAG Dual-Level (Local Entity & Global Thematic) Hop  │
└──────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### 3.1. Statistical Profiling & Distribution Modeling
1. **Streaming Moments Calculation (Welford's Algorithm):**
   - Computes running mean $\mu$, sample variance $s^2$, skewness $\gamma_1$, and kurtosis $\gamma_2$ in a single streaming pass through Polars without materializing full arrays in RAM.
2. **Freedman-Diaconis Binning Rule:**
   $$\text{Bin Width} = 2 \cdot \frac{\text{IQR}(x)}{\sqrt[3]{n}}$$
   Selects the optimal histogram bin width to prevent over-smoothing of multi-modal features.
3. **Median Absolute Deviation (MAD):**
   $$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
   Serves as a robust non-parametric measure of dispersion immune to catastrophic outlier distortion.

### 3.2. Outlier & Anomaly Detection Algorithms
1. **Hubert-Vandervieren Skewness-Adjusted Boxplot Fences:**
   Standard Tukey boxplots assume symmetric Gaussian noise, causing massive false-positive flags on naturally skewed data (e.g. income, transaction amounts). GetReport incorporates the medcouple $MC = \text{medcouple}(x)$:
   - When $MC \ge 0$:
     $$[\ Q_1 - 1.5 e^{-4 MC} \cdot \text{IQR},\quad Q_3 + 1.5 e^{3 MC} \cdot \text{IQR}\ ]$$
   - When $MC < 0$:
     $$[\ Q_1 - 1.5 e^{-3 MC} \cdot \text{IQR},\quad Q_3 + 1.5 e^{4 MC} \cdot \text{IQR}\ ]$$
2. **Modified Z-Score (Iglewicz & Hoaglin):**
   $$M_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}$$
   Scores are flagged as severe outliers when $|M_i| > 3.5$.
3. **Multivariate Isolation Forest & Local Outlier Factor (LOF):**
   Random recursive axis-aligned hyperplanes isolate anomalous multi-column combinations that appear normal when viewed in 1D projections.

### 3.3. Forensic Data Auditing & Compliance Algorithms
1. **Benford's Law Forensic First-Digit Test:**
   Tests whether leading significant digits $d \in \{1, \dots, 9\}$ follow the logarithmic distribution:
   $$P(d) = \log_{10}\left(1 + \frac{1}{d}\right)$$
   Computes Pearson's $\chi^2$ statistic and Chebyshev MAD:
   $$\chi^2 = \sum_{d=1}^9 \frac{(O_d - E_d)^2}{E_d}$$
   Deviations exceeding critical thresholds flag artificial transaction manipulation, synthetic test numbers, or arbitrary pricing limits.
2. **Population Stability Index (PSI):**
   Measures distributional drift between historical runs and current ingestions across $B$ quantile bins:
   $$\text{PSI} = \sum_{b=1}^B \left( Actual_b - Expected_b \right) \cdot \ln\left( \frac{Actual_b}{Expected_b} \right)$$
   - $\text{PSI} < 0.10$: Stable / No Drift.
   - $0.10 \le \text{PSI} \le 0.25$: Moderate Drift / Investigation Recommended.
   - $\text{PSI} > 0.25$: Significant Schema/Data Drift / Retraining Critical.
3. **Shannon Information Entropy ($H$) & Diversity Index:**
   $$H(X) = - \sum_{i=1}^k p(x_i) \log_2 p(x_i)$$
   Identifies near-constant low-variance columns ($H \to 0$) and high-cardinality ID collisions ($H \to \log_2 n$).

### 3.4. Time Series Intelligence (Meta Kats-Inspired)
1. **CUSUM (Cumulative Sum) Changepoint Detection:**
   Evaluates sequential Log-Likelihood Ratio (LLR) hypothesis tests against Gaussian distributions:
   $$\Lambda(\tau) = \frac{n}{2} \ln \sigma_0^2 - \frac{\tau}{2} \ln \sigma_1^2 - \frac{n - \tau}{2} \ln \sigma_2^2$$
   Finds exact timestamps where business processes fundamentally shifted.
2. **Holt-Winters Triple Exponential Smoothing:**
   Vectorized alpha/beta/gamma smoothing generating recursive point forecasts with 95% confidence bands.
3. **Mann-Kendall Rank Trend Test:**
   Non-parametric monotonic trend detection evaluating sign concordances of all pairs $(x_j - x_k)$.

### 3.5. Non-Linear Correlation & Association
1. **$\phi_K$ (Phi-K) Correlation Coefficient:**
   Transforms any pair of variables (numeric vs. numeric, numeric vs. categorical, or categorical vs. categorical) into a 2D contingency grid, calculates the Pearson $\chi^2$ contingency score, and inverts it into a normalized correlation $r \in [0, 1]$ matching the standard bivariate normal distribution. Captures inverted U-shapes, periodic sinusoids, and categorical segmentations that Pearson completely misses ($r \approx 0$).

### 3.6. Knowledge Graph & Network Topology
1. **Leiden & Louvain Modularity Maximization:**
   Partitions the data graph into non-overlapping semantic communities by maximizing graph modularity $Q$:
   $$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
2. **Degree Centrality & Hub Identification:**
   Identifies "God Nodes" (columns or issues with outsized topological degrees) that act as single points of failure across schemas and joins.
3. **Surprising Connections:**
   Evaluates observed edge weights between community clusters against random expectation, identifying hidden dependencies (e.g., an unindexed PII column unexpectedly correlated with a public revenue metric).

### 3.7. Conversational Information Retrieval (RAG)
1. **Reciprocal Rank Fusion (RRF):**
   Merges sparse lexical search (BM25/TF-IDF) with dense vector search (Cosine Similarity on embeddings):
   $$\text{RRF Score}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)},\quad k = 60$$
2. **LightRAG Dual-Level Graph Traversal:**
   - **Local 1-Hop Retrieval:** Traverses explicit entity edges (e.g. `col:TransactionAmount` $\to$ `issue:MissingValues`).
   - **Global Thematic Retrieval:** Scans community-level centroids and Understand-Anything architectural layers to answer high-level synthesis questions (*"What are the main risks across this dataset?"*).

---

## 4. Complete Technology Stack & Dependencies

```
                                TECHNOLOGY STACK BY LAYER
                                
  [Presentation Layer]              [API & Orchestration Layer]         [Engine & Analytics Layer]
┌───────────────────────────┐     ┌────────────────────────────┐      ┌────────────────────────────┐
│ • React 18.3 + TypeScript │     │ • FastAPI 0.110 (ASGI)     │      │ • Polars (Apache Arrow)    │
│ • Vite 5.4 (SWC Compiler) │────▶│ • Uvicorn + Granian Workers│─────▶│ • DuckDB (Vectorized SQL)  │
│ • TailwindCSS v3.4        │     │ • Pydantic v2 Settings     │      │ • NumPy + SciPy + Sklearn  │
│ • Radix UI / Shadcn / Vaul│     │ • ARQ (Redis Task Queue)   │      │ • NetworkX (Graph Store)   │
│ • Recharts + Lucide Icons │     │ • SQLite WAL State Store   │      │ • Snowflake Connector      │
└───────────────────────────┘     └────────────────────────────┘      └────────────────────────────┘
                                                                                    │
                                  [Document & AI Layer]                             ▼
                                ┌────────────────────────────┐        ┌────────────────────────────┐
                                │ • Typst 0.15 (Rust Engine) │◀───────│ In-Memory Arrow Buffers   │
                                │ • Matplotlib 3.8 (Agg)     │        │ Zero-Copy IPC Columnar Data│
                                │ • Google Gemini 2.5 Flash  │        │ Disk-Mapped Parquet Caches │
                                │ • OpenAI / OpenRouter      │        └────────────────────────────┘
                                └────────────────────────────┘
```

### 4.1. Backend Framework & Runtimes
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **Python** | `>=3.12.0` | Core backend runtime utilizing modern union syntax (`X \| Y`), async tasks, and enhanced type hints. |
| **FastAPI** | `>=0.110.0` | High-performance ASGI REST API framework providing automatic OpenAPI docs and dependency injection. |
| **Uvicorn / Granian** | `>=0.29.0` | Multi-worker asynchronous HTTP server supporting WebSockets and SSE streaming. |
| **Pydantic** | `>=2.2.1` | High-speed data parsing, schema enforcement, and type validation written in Rust. |
| **ARQ / Redis** | `>=0.26.1` | Non-blocking asynchronous task queue for distributed background job execution, with an in-memory task dispatcher fallback. |
| **SQLite (WAL Mode)** | Native | Embedded persistent metadata database running in Write-Ahead-Logging mode for zero-latency concurrent job state updates. |

### 4.2. Columnar Processing, SQL & Storage
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **Polars** | `>=1.2.1` | Multi-threaded in-memory DataFrame engine built on Apache Arrow. Handles all statistical aggregations and data cleaning. |
| **DuckDB** | `>=1.0.0` | Embedded vectorized analytical OLAP database. Powers interactive zero-copy SQL queries across Parquet caches. |
| **PyArrow** | `>=14.0.0` | Apache Arrow memory format for zero-copy memory transfers between Polars and DuckDB. |
| **FastExcel / OpenPyXL** | `>=0.10.0` | Native Rust-powered Excel file parser capable of reading multi-sheet `.xlsx` files without memory bloat. |
| **Snowflake Connector** | `>=3.7.0` | Direct enterprise Snowflake data warehouse extraction supporting RSA private-key and OAuth authentication. |
| **SQLGlot** | `>=23.0.0` | Multi-dialect SQL parser used for AST table, column, CTE, and view lineage extraction. |

### 4.3. Statistical Analysis, Machine Learning & Graph Theory
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **NumPy & SciPy** | `>=1.26.4` | Matrix computations, chi-square cumulative distribution functions, and CUSUM likelihood calculations. |
| **Scikit-Learn** | `>=1.4.0` | Isolation Forest, Local Outlier Factor, PCA dimensionality reduction, and categorical label encodings. |
| **NetworkX** | `>=3.0` | Directed multigraph data structure powering LightRAG, Graphify modularity clustering, and topological paths. |

### 4.4. Document Compilation & Visual Output
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **Typst** | `>=0.15.0` | Native Rust document typesetting engine. Generates vector PDF reports with zero memory leaks in sub-100ms. |
| **Matplotlib** | `>=3.8.3` | Headless vector chart generator (`Agg` backend) generating publication-grade SVG and high-DPI PNG visualizations. |
| **ReportLab** | `>=4.1.0` | Secondary vector PDF engine acting as a zero-dependency fallback. |

### 4.5. AI & Large Language Model Integration
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **Google GenAI SDK** | `>=1.0.0` | Primary LLM integration connecting to Gemini 2.5 Flash for rapid, cost-effective executive narrative synthesis. |
| **OpenAI SDK** | `>=1.14.0` | Fallback adapter supporting GPT-4o, Claude 3.5 Sonnet, and OpenRouter endpoint routing. |

### 4.6. Frontend Single-Page Application (SPA)
| Technology | Version | Purpose in GetReport |
| :--- | :--- | :--- |
| **React** | `18.3.1` | Declarative user interface library with hooks, context state management, and concurrent rendering. |
| **Vite** | `5.4.19` | Ultra-fast build tool and development server utilizing Rollup and SWC native compilation. |
| **TypeScript** | `5.8.3` | Strict static typing across all UI state, API payloads, and component properties. |
| **TailwindCSS** | `3.4.17` | Utility-first CSS framework styled with the Slate/Dark-Navy enterprise palette. |
| **Radix UI / Shadcn** | Modern | Accessible, unstyled UI primitives (Dialogs, Accordions, Tabs, Tooltips, Sliders, Context Menus). |
| **Recharts** | `2.15.4` | Composable SVG charting library for interactive distributions and correlation matrices. |
| **Lucide React** | `0.462.0` | Comprehensive enterprise icon set. |
| **PapaParse** | `5.5.3` | In-browser CSV streaming parser for zero-latency client-side file previews. |

---

## 5. Security & Enterprise Hardening

GetReport incorporates defense-in-depth security principles across the entire application:
1. **Magic-Byte Sniffing:** Verifies file signatures before reading (blocking extension-spoofed binaries).
2. **Zip Slip & Tar Bomb Defense:** Recursively validates relative extraction paths and caps uncompressed byte counts.
3. **AST-Sandboxed Python Analyst ([sandboxed_analyst.py](file:///d:/Git_Repo/GetReport/Backend/app/services/sandboxed_analyst.py)):** When users or LLMs run exploratory Python code, an AST visitor inspects every node. It strictly forbids `import`, `__import__`, `eval`, `exec`, `open`, `os`, `sys`, `subprocess`, file writing, and network sockets, terminating execution if thread limits (default: 5.0 seconds) are exceeded.
4. **Strict Loopback Binding:** Local development servers bind exclusively to `127.0.0.1` to prevent unauthorized LAN exposure.
5. **PII Masking Guard ([pii_guard.py](file:///d:/Git_Repo/GetReport/Backend/app/services/pii_guard.py)):** Identifies Social Security Numbers, Credit Cards, IBANs, and private credentials, automatically applying hashing or masking before feeding text to external LLMs.

---

## 6. Summary: Key Architecture Differentiators

| Capability | Traditional Profilers (Pandas-Profiling, YData) | GetReport Platform |
| :--- | :--- | :--- |
| **Engine Core** | Single-threaded Pandas (High memory, crashes on $>1\text{GB}$) | **Polars Streaming + DuckDB OLAP** (Zero-copy, processes $10\text{M}+$ rows) |
| **Data Cleaning** | Uncontrolled black-box drops | **Deterministic Row Intelligence** (`PROTECT`/`REVIEW`/`FIX`/`EXCLUDE`) + Human-in-the-Loop Issue Ledger |
| **Audit Lineage** | None | **Immutable Transformation DAG** + Delta Comparison Report |
| **PDF Compilation** | WeasyPrint / Headless Chrome ($500\text{MB}+$ RAM, slow) | **Rust Typst Engine** ($<25\text{MB}$ RAM, sub-100ms compilation) |
| **Graph Intelligence** | None | **NetworkX + Graphify + Understand-Anything** (Modularity clustering & 6 architectural layers) |
| **Conversational AI** | Static one-shot summaries | **LightRAG Dual-Level Hybrid Chat** (Local entity graph traversal + global thematic search) |
