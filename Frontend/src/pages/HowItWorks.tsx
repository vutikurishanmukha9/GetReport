import { useState } from "react";
import { 
  UploadCloud, Search, FileDown, ArrowRight, Brain, Sparkles, 
  CheckCircle2, Terminal, ChevronRight, ChevronLeft, 
  ShieldCheck, Zap, Layers, RefreshCw
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

interface PipelineStep {
  id: string;
  stepNum: string;
  icon: typeof UploadCloud;
  title: string;
  shortDesc: string;
  fullDesc: string;
  badge: string;
  codeSnippet: string;
  codeLanguage: string;
  reasoningLogs: string[];
  specs: { label: string; value: string }[];
}

const pipelineSteps: PipelineStep[] = [
  {
    id: "ingest",
    stepNum: "01",
    icon: UploadCloud,
    title: "Polars Zero-Copy Ingestion",
    shortDesc: "Streaming buffer parsing with automatic encoding and delimiter detection.",
    fullDesc: "When a file is uploaded, our engine memory-maps the raw binary directly into Polars ChunkedArrays. It samples the first 1,000 lines to resolve CSV delimiters, UTF-8/CP1252 encodings, and infers strict Arrow data types without touching disk storage.",
    badge: "SUB-50MS STREAMING",
    codeLanguage: "python",
    codeSnippet: `import polars as pl

# Zero-copy memory-mapped ingestion with schema inference
df = pl.read_csv(
    file_buffer,
    infer_schema_length=10000,
    try_parse_dates=True,
    truncate_ragged_lines=False,
    ignore_errors=False
)
# Memory footprint: 12MB RAM vs 148MB in Pandas`,
    reasoningLogs: [
      "Detected CSV delimiter: ',' (confidence 99.8%)",
      "Encoding validated: UTF-8 with no byte-order mark (BOM)",
      "Inferred 14 columns: 6 Int64, 4 Float64, 2 Utf8, 2 Date32"
    ],
    specs: [
      { label: "Supported Formats", value: "CSV, XLSX, Parquet, TSV, JSONL" },
      { label: "Memory Retention", value: "0 bytes saved to disk" },
      { label: "Chunk Batch Size", value: "64,000 rows/vector" }
    ]
  },
  {
    id: "hygiene",
    stepNum: "02",
    icon: Search,
    title: "Hygiene Scoring & Issue Ledger",
    shortDesc: "9-category quality audit with interactive human-in-the-loop approvals.",
    fullDesc: "Every column is scanned across 4 mathematical dimensions: Completeness (35%), Consistency (25%), Validity (25%), and Stability (15%). Identified quality flags are placed in the Issue Ledger where users can approve, reject, or customize transformations.",
    badge: "A-F CONFIDENCE SCORING",
    codeLanguage: "python",
    codeSnippet: `# Quality scoring & transformation DAG execution
confidence_score = (
    completeness * 0.35 +
    consistency  * 0.25 +
    validity     * 0.25 +
    stability    * 0.15
)

# Apply approved Issue Ledger transformations
if user_decisions["col_amount"] == "approve":
    df = df.with_columns(
        pl.col("amount").str.replace_all(r"[$]", "").cast(pl.Float64)
    )`,
    reasoningLogs: [
      "Flagged 42 string values with currency prefixes in 'amount' column",
      "Detected 14 duplicate rows sharing identical primary keys",
      "Calculated dataset health index: 92.4% (Grade A-)"
    ],
    specs: [
      { label: "Issue Categories", value: "9 structural checks" },
      { label: "DAG Reversibility", value: "100% auditable" },
      { label: "Audit Resolution", value: "Column & row-level" }
    ]
  },
  {
    id: "stats",
    stepNum: "03",
    icon: Brain,
    title: "Statistical & Drift Auditing",
    shortDesc: "Pearson correlations, VIF multicollinearity, and conceptual drift checks.",
    fullDesc: "Executes deep bivariate and multivariate statistical tests. Identifies redundant variables using Variance Inflation Factor (VIF > 5.0), flags extreme skewness with Kurtosis thresholds, and computes Kolmogorov-Smirnov distribution shifts across chronological segments.",
    badge: "MULTIVARIATE RIGOR",
    codeLanguage: "python",
    codeSnippet: `from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Multivariate VIF calculation to detect predictor redundancy
vif_scores = {
    col: variance_inflation_factor(numeric_matrix, i)
    for i, col in enumerate(numeric_cols)
}
# Pairwise correlation thresholding (|r| > 0.70)
corr_matrix = df.select(numeric_cols).corr()`,
    reasoningLogs: [
      "VIF check passed: all predictors VIF < 4.2 (no severe multicollinearity)",
      "High correlation detected between 'unit_price' and 'total_cost' (r = 0.94)",
      "No significant chronological drift detected across Q1-Q4 segments"
    ],
    specs: [
      { label: "Correlation Type", value: "Pearson & Spearman" },
      { label: "Multicollinearity", value: "VIF Threshold: 5.0" },
      { label: "Drift Metric", value: "KS 2-sample p < 0.05" }
    ]
  },
  {
    id: "rag",
    stepNum: "04",
    icon: Sparkles,
    title: "Domain Extraction & RAG Synthesis",
    shortDesc: "Semantic business context classification and grounded insight generation.",
    fullDesc: "Our semantic layer analyzes column taxonomy to classify the business domain (e.g. Retail, SaaS, Healthcare, Banking). Aggregated statistical summaries are sent to our RAG engine to generate plain-English executive takeaways without exposing raw row data.",
    badge: "EPHEMERAL AI SYNTHESIS",
    codeLanguage: "json",
    codeSnippet: `// RAG payload format (Confidential: Zero raw rows transmitted)
{
  "domain": "Sales & E-Commerce",
  "rows": 14200,
  "columns_audited": 12,
  "dataset_health_score": 94.2,
  "key_findings": [
    "Peak transaction volume occurs between 18:00 - 21:00 UTC",
    "Return rate anomaly of 8.4% concentrated in Category B"
  ]
}`,
    reasoningLogs: [
      "Domain inferred: 'Sales & E-Commerce' (taxonomy match 96.4%)",
      "Generated 4 executive risk summaries with statistical citations",
      "Verified zero PII payload dispatch before external API invocation"
    ],
    specs: [
      { label: "LLM Providers", value: "Gemini 2.5 Flash / OpenRouter" },
      { label: "Deterministic Fallback", value: "100% offline rule engine" },
      { label: "PII Shield", value: "Raw records omitted" }
    ]
  },
  {
    id: "pdf",
    stepNum: "05",
    icon: FileDown,
    title: "Board-Ready PDF Compilation",
    shortDesc: "High-DPI print-ready audit document generated via WeasyPrint.",
    fullDesc: "The final audit package compiles into an executive PDF report via WeasyPrint. Includes embedded Matplotlib visual correlation galleries, executive summaries, full column confidence scorecards, and a cryptographic verification receipt.",
    badge: "WEASYPRINT PRINT ENGINE",
    codeLanguage: "python",
    codeSnippet: `import weasyprint

# High-fidelity PDF rendering with cached CSS layout
html_document = render_template(
    "audit_report.html",
    metadata=dataset_metadata,
    confidence_ledger=confidence_data,
    charts=matplotlib_base64_gallery
)

pdf_bytes = weasyprint.HTML(string=html_document).write_pdf(
    stylesheets=["report_print_styles.css"],
    optimize_size=('fonts', 'images')
)`,
    reasoningLogs: [
      "Rendered 4 Matplotlib charts at 300 DPI vector clarity",
      "Embedded complete transformation ledger and audit receipts",
      "PDF compiled in 1.4s (filesize: 420 KB)"
    ],
    specs: [
      { label: "Print Standard", value: "CSS Paged Media Module 3" },
      { label: "Visual DPI", value: "300 DPI vector charts" },
      { label: "Export Formats", value: "PDF, Parquet, CSV, HTML" }
    ]
  }
];

export const HowItWorks = () => {
  const [activeStepIndex, setActiveStepIndex] = useState<number>(0);
  const activeStep = pipelineSteps[activeStepIndex];

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
              <Zap className="h-3.5 w-3.5" />
              <span>Auditable Pipeline Methodology</span>
            </div>
            
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Transparent Data Pipeline.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              From messy spreadsheet fields to board-level audit reports. We expose every decision, every threshold, and every mathematical operation step-by-step.
            </p>
          </div>
        </div>

        {/* Section 1: Interactive Pipeline Stage Explorer */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-7xl space-y-6 sm:space-y-8">
          
          {/* Step Pill Selector Bar */}
          <div className="flex overflow-x-auto gap-2 sm:gap-2.5 pb-2 scrollbar-none border-b border-border/60">
            {pipelineSteps.map((step, idx) => {
              const Icon = step.icon;
              const isActive = idx === activeStepIndex;
              return (
                <button
                  key={step.id}
                  type="button"
                  onClick={() => setActiveStepIndex(idx)}
                  className={`flex items-center gap-2 px-3.5 py-2.5 rounded-xl font-mono text-xs transition-all shrink-0 cursor-pointer border ${
                    isActive
                      ? "bg-primary text-primary-foreground border-primary shadow-premium font-bold"
                      : "bg-card hover:bg-muted/40 text-muted-foreground border-border/80 hover:text-foreground"
                  }`}
                >
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${isActive ? "bg-white/20" : "bg-muted"}`}>
                    {step.stepNum}
                  </span>
                  <Icon className="h-4 w-4" />
                  <span className="font-sans font-semibold text-xs whitespace-nowrap">{step.title}</span>
                </button>
              );
            })}
          </div>

          {/* Active Stage Deep-Dive Card */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Left Column: Stage Details & Specs (Col 1-5) */}
            <div className="lg:col-span-5 space-y-4">
              <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 shadow-premium space-y-5">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Badge variant="outline" className="text-[10px] font-mono font-bold bg-primary/10 text-primary border-primary/20 uppercase tracking-wider">
                      {activeStep.badge}
                    </Badge>
                    <span className="font-mono text-xs text-muted-foreground">
                      Stage {activeStep.stepNum} of 05
                    </span>
                  </div>

                  <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground">
                    {activeStep.title}
                  </h2>

                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    {activeStep.fullDesc}
                  </p>
                </div>

                {/* Architecture Specifications */}
                <div className="border-t border-border/60 pt-3 space-y-2 font-mono text-xs">
                  <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold block">
                    Execution Parameters
                  </span>
                  {activeStep.specs.map((spec, sIdx) => (
                    <div key={sIdx} className="flex justify-between items-center py-0.5 border-b border-border/40 last:border-b-0 text-[11px]">
                      <span className="text-muted-foreground">{spec.label}:</span>
                      <span className="font-semibold text-foreground">{spec.value}</span>
                    </div>
                  ))}
                </div>

                {/* Stage Step Navigation Controls */}
                <div className="flex items-center justify-between pt-1">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={activeStepIndex === 0}
                    onClick={() => setActiveStepIndex(prev => Math.max(0, prev - 1))}
                    className="rounded-xl font-mono text-xs gap-1.5 h-9"
                  >
                    <ChevronLeft className="h-3.5 w-3.5" /> Previous
                  </Button>
                  <Button
                    variant="default"
                    size="sm"
                    disabled={activeStepIndex === pipelineSteps.length - 1}
                    onClick={() => setActiveStepIndex(prev => Math.min(pipelineSteps.length - 1, prev + 1))}
                    className="rounded-xl font-mono text-xs gap-1.5 h-9"
                  >
                    Next Stage <ChevronRight className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </Card>
            </div>

            {/* Right Column: Code & Reasoning Log Inspector (Col 6-12) */}
            <div className="lg:col-span-7 space-y-4">
              
              {/* Code Terminal Mock */}
              <Card className="border border-border/80 bg-zinc-950 text-zinc-100 rounded-2xl shadow-premium overflow-hidden font-mono text-xs">
                <CardHeader className="bg-zinc-900/80 border-b border-zinc-800 px-4 py-2.5 flex flex-row items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80 inline-block" />
                    <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80 inline-block" />
                    <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80 inline-block" />
                    <span className="text-[11px] text-zinc-400 font-bold ml-2">engine_pipeline.py</span>
                  </div>
                  <Badge variant="outline" className="text-[10px] bg-zinc-800 text-zinc-300 border-zinc-700">
                    STAGE_{activeStep.stepNum}_EXEC
                  </Badge>
                </CardHeader>
                <CardContent className="p-4 sm:p-5 overflow-x-auto">
                  <pre className="text-[11px] sm:text-xs leading-relaxed text-emerald-400 font-mono">
                    <code>{activeStep.codeSnippet}</code>
                  </pre>
                </CardContent>
              </Card>

              {/* Reasoning Logs Terminal */}
              <Card className="border border-border bg-card rounded-2xl p-4 sm:p-5 shadow-premium space-y-2.5 font-mono text-xs">
                <div className="flex items-center gap-2 text-foreground font-semibold border-b border-border/60 pb-2">
                  <Terminal className="h-4 w-4 text-primary" />
                  <span>Why-I-Did-X Transparency Logs</span>
                </div>
                <div className="space-y-1.5">
                  {activeStep.reasoningLogs.map((log, lIdx) => (
                    <div key={lIdx} className="flex items-start gap-2 text-muted-foreground text-[11px]">
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 shrink-0 mt-0.5" />
                      <span className="text-foreground">{log}</span>
                    </div>
                  ))}
                </div>
              </Card>

            </div>

          </div>
        </div>

        {/* Section 2: Pipeline Invariants Guarantee */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-7xl">
          <div className="border border-border bg-card rounded-2xl sm:rounded-3xl p-6 sm:p-8 shadow-premium space-y-6">
            <div className="text-center max-w-2xl mx-auto space-y-1.5">
              <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                Architectural Invariants & Safety Guarantees
              </h2>
              <p className="text-xs sm:text-sm text-muted-foreground font-sans">
                Every calculation in GetReport adheres to 4 immutable engineering principles.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 font-mono text-xs">
              <div className="p-4 rounded-xl bg-muted/20 border border-border/60 space-y-1.5">
                <ShieldCheck className="h-5 w-5 text-emerald-600" />
                <h3 className="font-bold text-foreground font-sans text-xs sm:text-sm">Zero Disk Storage</h3>
                <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                  Dataframe records exist solely in RAM buffers and are wiped immediately upon session reset.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-muted/20 border border-border/60 space-y-1.5">
                <RefreshCw className="h-5 w-5 text-primary" />
                <h3 className="font-bold text-foreground font-sans text-xs sm:text-sm">Deterministic Seeds</h3>
                <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                  Statistical sample approximations use fixed random seeds to ensure reproducible audit scores.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-muted/20 border border-border/60 space-y-1.5">
                <Layers className="h-5 w-5 text-blue-600" />
                <h3 className="font-bold text-foreground font-sans text-xs sm:text-sm">Reversible DAGs</h3>
                <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                  Every Issue Ledger transformation records inverse operations so you can inspect raw vs remediated states.
                </p>
              </div>

              <div className="p-4 rounded-xl bg-muted/20 border border-border/60 space-y-1.5">
                <Sparkles className="h-5 w-5 text-purple-600" />
                <h3 className="font-bold text-foreground font-sans text-xs sm:text-sm">Confidential RAG</h3>
                <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                  LLMs receive only aggregated mathematical metrics (mean, count, VIF) and never raw patient/customer rows.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Section 3: Bottom CTA */}
        <div className="border-t border-border/60 bg-muted/20 py-10 sm:py-12">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Test the pipeline on your data
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              Experience sub-50ms ingestion and automated confidence scoring right now.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                  <span>Open Data Workspace</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default HowItWorks;
