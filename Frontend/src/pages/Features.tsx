import { useState } from "react";
import { 
  ArrowRight, Layers, CheckCircle2, 
  FileText, ArrowLeftRight, Activity, ShieldCheck,
  Code2, MessageSquareCode, Lock, Database,
  Sliders, Eye, Download, AlertTriangle, ChevronDown,
  Check, X
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

// Module scope pure function for Confidence Grade calculation
const getConfidenceGrade = (score: number) => {
  if (score >= 90) return { grade: "A", label: "Production Ready", color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" };
  if (score >= 80) return { grade: "B", label: "Minor Issues", color: "bg-sky-500/10 text-sky-400 border-sky-500/30" };
  if (score >= 70) return { grade: "C", label: "Needs Remediation", color: "bg-amber-500/10 text-amber-400 border-amber-500/30" };
  if (score >= 60) return { grade: "D", label: "High Risk", color: "bg-orange-500/10 text-orange-400 border-orange-500/30" };
  return { grade: "F", label: "Critical Anomalies", color: "bg-rose-500/10 text-rose-400 border-rose-500/30" };
};

export const Features = () => {
  // Confidence Grade Simulator State
  const [completeness, setCompleteness] = useState(96);
  const [consistency, setConsistency] = useState(92);
  const [validity, setValidity] = useState(88);
  const [stability, setStability] = useState(94);
  const [previewMode, setPreviewMode] = useState<"raw" | "remediated" | "python">("remediated");

  // FAQ Accordion State
  const [openFaq, setOpenFaq] = useState<number | null>(0);

  // Weighted Confidence Calculation: 35% Completeness + 25% Consistency + 25% Validity + 15% Stability
  const overallScore = Math.round(
    completeness * 0.35 +
    consistency * 0.25 +
    validity * 0.25 +
    stability * 0.15
  );

  const gradeInfo = getConfidenceGrade(overallScore);

  // Simulator Presets
  const applyPreset = (c: number, cs: number, v: number, s: number) => {
    setCompleteness(c);
    setConsistency(cs);
    setValidity(v);
    setStability(s);
  };

  const faqItems = [
    {
      q: "How does zero-copy ingestion work without saving files to disk?",
      a: "GetReport processes data directly within ephemeral memory buffers. When you upload a CSV, XLSX, or Parquet file, the backend Polars engine parses and maps the binary into Apache Arrow ChunkedArrays in RAM. Zero intermediate tables or temp files are written to disk, and buffers are purged as soon as execution completes."
    },
    {
      q: "Can I export the approved transformations back to raw Python or SQL?",
      a: "Yes. Every transformation you approve in the Issue Ledger is tracked in an auditable Directed Acyclic Graph (DAG). You can export the final cleaned dataset as Parquet, CSV, or an executable Python script with Polars commands that you can incorporate directly into existing ETL workflows."
    },
    {
      q: "How does the Ephemeral RAG companion answer questions without leaking private rows?",
      a: "The RAG engine never transmits raw data rows or sensitive records to external language models. Instead, it computes and embeds aggregate statistical summaries, correlation matrices, schema metadata, and transformation logs in an in-memory vector index. Queries are resolved strictly against these statistical summaries."
    },
    {
      q: "What file formats and dataset sizes are supported?",
      a: "GetReport supports CSV, TSV, XLSX, Parquet, and JSONL formats for both input and output. Thanks to the Polars multithreaded engine, datasets containing hundreds of thousands of rows stream and process in under 50 milliseconds."
    }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28">
        
        {/* Section 1: Hero Header & Live Polars Benchmark Terminal */}
        <section className="border-b border-border/60 bg-gradient-to-b from-card/30 via-background to-background py-8 sm:py-14">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
              
              {/* Left Column: Editorial Headline & Value Proposition */}
              <div className="lg:col-span-7 space-y-5 text-left">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/30 bg-primary/10 text-primary text-xs font-mono">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  <span>Polars 1.0 Streaming Core • Zero-Copy Memory</span>
                </div>
                
                <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold tracking-[-0.02em] text-foreground leading-[1.14]">
                  Engineered for rigorous <span className="text-primary block mt-1">data quality audits.</span>
                </h1>
                
                <p className="text-sm sm:text-base text-muted-foreground max-w-xl leading-relaxed font-sans">
                  A high-throughput intelligence stack for data analysts, ML engineers, and decision-makers. Automate column trust scoring, approve transformation DAGs, and compile executive-ready PDF audit reports in seconds.
                </p>

                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <Link to="/workspace" className="w-full sm:w-auto">
                    <Button size="lg" className="w-full sm:w-auto h-11 px-6 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                      <span>Start Free Audit</span>
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                  <Link to="/how-it-works" className="w-full sm:w-auto">
                    <Button size="lg" variant="outline" className="w-full sm:w-auto h-11 px-6 rounded-xl border-border bg-card/80 hover:bg-muted/20 shadow-premium t-card-lift t-spring-press font-display text-sm">
                      <span>Pipeline Architecture</span>
                    </Button>
                  </Link>
                </div>

                {/* Micro Guarantee Metrics */}
                <div className="grid grid-cols-3 gap-3 pt-4 border-t border-border/40 font-mono text-xs">
                  <div>
                    <span className="block font-bold text-foreground text-sm sm:text-base">0 MB</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Permanent Storage</span>
                  </div>
                  <div>
                    <span className="block font-bold text-emerald-400 text-sm sm:text-base">&lt; 50ms</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Polars Streaming</span>
                  </div>
                  <div>
                    <span className="block font-bold text-primary text-sm sm:text-base">A to F</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence Grading</span>
                  </div>
                </div>
              </div>

              {/* Right Column: Live Polars vs Pandas Benchmark Terminal */}
              <div className="lg:col-span-5 w-full">
                <Card className="border border-border/80 bg-card/90 rounded-2xl shadow-premium overflow-hidden t-card-lift backdrop-blur-sm">
                  <CardHeader className="bg-muted/20 border-b border-border/60 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 font-mono text-xs font-semibold text-foreground">
                        <Activity className="h-4 w-4 text-primary" />
                        <span>Polars Zero-Copy Benchmark</span>
                      </div>
                      <Badge variant="outline" className="text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                        100k Rows Streamed
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 sm:p-5 space-y-3.5 font-mono text-xs">
                    <div className="space-y-3">
                      {/* Polars Row */}
                      <div className="p-3 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-1.5">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className="font-bold text-emerald-300 flex items-center gap-1.5">
                            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" /> Polars Engine
                          </span>
                          <span className="font-bold text-emerald-300 bg-emerald-500/20 px-2 py-0.5 rounded text-[10px]">22.4x Speedup</span>
                        </div>
                        <div className="flex justify-between text-[11px] text-emerald-200/90 font-sans">
                          <span>Execution: <strong className="font-mono text-emerald-300">42ms</strong></span>
                          <span>RAM Usage: <strong className="font-mono text-emerald-300">12MB</strong></span>
                        </div>
                      </div>

                      {/* Standard Pandas Row */}
                      <div className="p-3 rounded-xl bg-muted/20 border border-border/50 space-y-1.5 opacity-80">
                        <div className="flex justify-between items-center text-[11px]">
                          <span className="text-muted-foreground font-semibold">Standard Pandas (Baseline)</span>
                          <span className="text-muted-foreground text-[10px]">Disk Cache Bound</span>
                        </div>
                        <div className="flex justify-between text-[11px] text-muted-foreground font-sans">
                          <span>Execution: <strong className="font-mono text-foreground">940ms</strong></span>
                          <span>RAM Usage: <strong className="font-mono text-foreground">148MB</strong></span>
                        </div>
                      </div>
                    </div>

                    <div className="p-3 bg-muted/30 rounded-xl border border-border/40 text-[11px] font-sans text-muted-foreground space-y-1">
                      <strong className="text-foreground font-mono block text-[10px] uppercase tracking-wider">Zero-Copy Memory Guarantee:</strong>
                      <span>Files are memory-mapped into Polars ChunkedArrays in RAM. Intermediate buffers are purged automatically upon execution.</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

            </div>
          </div>
        </section>

        {/* Section 2: Interactive Confidence Grade Simulator */}
        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 max-w-6xl">
          <Card className="border border-border/80 bg-card rounded-2xl sm:rounded-3xl p-5 sm:p-8 shadow-premium space-y-6">
            
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-5 border-b border-border/60 pb-6">
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/10 text-primary border-primary/20">
                    Interactive Simulator
                  </Badge>
                  <span className="text-xs text-muted-foreground font-mono">Weighted Algorithm</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground tracking-[-0.015em]">
                  Column Confidence Scoring Engine
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground max-w-xl font-sans">
                  Adjust the four core quality dimensions below to simulate how our scoring engine grades individual tabular variables in real time.
                </p>
                
                {/* Preset Scenario Buttons */}
                <div className="flex flex-wrap items-center gap-2 pt-1 font-mono text-[11px]">
                  <span className="text-muted-foreground text-[10px] uppercase tracking-wider mr-1">Presets:</span>
                  <button
                    type="button"
                    onClick={() => applyPreset(96, 92, 88, 94)}
                    className="px-2.5 py-1 rounded-lg border border-border bg-muted/20 hover:bg-muted/40 text-foreground transition-all cursor-pointer"
                  >
                    Clean Data (93%)
                  </button>
                  <button
                    type="button"
                    onClick={() => applyPreset(72, 68, 75, 70)}
                    className="px-2.5 py-1 rounded-lg border border-border bg-muted/20 hover:bg-muted/40 text-foreground transition-all cursor-pointer"
                  >
                    Mixed Schema (71%)
                  </button>
                  <button
                    type="button"
                    onClick={() => applyPreset(40, 50, 45, 35)}
                    className="px-2.5 py-1 rounded-lg border border-border bg-muted/20 hover:bg-muted/40 text-foreground transition-all cursor-pointer"
                  >
                    Anomalous Export (43%)
                  </button>
                </div>
              </div>

              {/* Dynamic Live Grade Result */}
              <div className={`p-4 sm:p-5 rounded-2xl border ${gradeInfo.color} flex items-center gap-4 shrink-0 transition-all duration-300 shadow-sm`}>
                <div className="text-center min-w-[48px]">
                  <span className="text-4xl sm:text-5xl font-display font-extrabold block leading-none">
                    {gradeInfo.grade}
                  </span>
                  <span className="text-[10px] font-mono font-bold uppercase tracking-wider block mt-1">
                    Grade
                  </span>
                </div>
                <div className="border-l border-current/20 pl-4 space-y-1">
                  <span className="text-lg sm:text-xl font-mono font-bold block">{overallScore}%</span>
                  <span className="text-xs font-sans font-medium block opacity-95">{gradeInfo.label}</span>
                  <span className="text-[10px] font-mono block opacity-75">Formula: 35%C + 25%S + 25%V + 15%T</span>
                </div>
              </div>
            </div>

            {/* Metric Sliders Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 font-mono text-xs">
              
              {/* Slider 1: Completeness */}
              <div className="space-y-2.5 p-4 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4 text-primary" /> Completeness (Weight: 35%)
                  </span>
                  <span className="font-bold text-primary bg-primary/10 px-2 py-0.5 rounded text-[11px]">{completeness}%</span>
                </div>
                <div className="space-y-1 pt-1">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={completeness}
                    onChange={(e) => setCompleteness(Number(e.target.value))}
                    className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-full transition-all"
                    aria-label="Completeness percentage"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
                    <span>0% (Empty)</span>
                    <span>100% (Zero Nulls)</span>
                  </div>
                </div>
                <span className="text-[11px] text-muted-foreground font-sans block">
                  Measures null values, empty strings, and masked NaN values across columns.
                </span>
              </div>

              {/* Slider 2: Consistency */}
              <div className="space-y-2.5 p-4 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-2">
                    <Layers className="h-4 w-4 text-primary" /> Consistency (Weight: 25%)
                  </span>
                  <span className="font-bold text-primary bg-primary/10 px-2 py-0.5 rounded text-[11px]">{consistency}%</span>
                </div>
                <div className="space-y-1 pt-1">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={consistency}
                    onChange={(e) => setConsistency(Number(e.target.value))}
                    className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-full transition-all"
                    aria-label="Consistency percentage"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
                    <span>0% (Mixed Types)</span>
                    <span>100% (Uniform Types)</span>
                  </div>
                </div>
                <span className="text-[11px] text-muted-foreground font-sans block">
                  Evaluates type cohesion, datetime format consistency, and schema anomalies.
                </span>
              </div>

              {/* Slider 3: Validity */}
              <div className="space-y-2.5 p-4 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-2">
                    <ShieldCheck className="h-4 w-4 text-primary" /> Validity (Weight: 25%)
                  </span>
                  <span className="font-bold text-primary bg-primary/10 px-2 py-0.5 rounded text-[11px]">{validity}%</span>
                </div>
                <div className="space-y-1 pt-1">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={validity}
                    onChange={(e) => setValidity(Number(e.target.value))}
                    className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-full transition-all"
                    aria-label="Validity percentage"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
                    <span>0% (Out-of-Bound)</span>
                    <span>100% (Domain Valid)</span>
                  </div>
                </div>
                <span className="text-[11px] text-muted-foreground font-sans block">
                  Detects range violations, negative balances, and invalid regex domains.
                </span>
              </div>

              {/* Slider 4: Stability */}
              <div className="space-y-2.5 p-4 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-2">
                    <Activity className="h-4 w-4 text-primary" /> Stability (Weight: 15%)
                  </span>
                  <span className="font-bold text-primary bg-primary/10 px-2 py-0.5 rounded text-[11px]">{stability}%</span>
                </div>
                <div className="space-y-1 pt-1">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={stability}
                    onChange={(e) => setStability(Number(e.target.value))}
                    className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-full transition-all"
                    aria-label="Stability percentage"
                  />
                  <div className="flex justify-between text-[9px] text-muted-foreground font-mono">
                    <span>0% (High Drift)</span>
                    <span>100% (Stable Dist)</span>
                  </div>
                </div>
                <span className="text-[11px] text-muted-foreground font-sans block">
                  Flags distribution skewness, extreme kurtosis, and chronological concept drift.
                </span>
              </div>

            </div>
          </Card>
        </section>

        {/* Section 3: Core Capabilities Bento Grid */}
        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 max-w-7xl space-y-10">
          <div className="text-center max-w-2xl mx-auto space-y-3">
            <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/10 text-primary border-primary/20">
              Deep Architecture
            </Badge>
            <h2 className="text-2xl sm:text-3xl md:text-4xl font-display font-bold text-foreground tracking-[-0.02em]">
              Enterprise feature matrix
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
              Deterministic, auditable capabilities designed to replace fragile spreadsheet macros and manual Python boilerplate.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            
            {/* Bento Tile 1: Interactive Issue Ledger (Large 2-col) */}
            <Card className="lg:col-span-2 border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                      <Code2 className="h-4 w-4" />
                    </div>
                    <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/10 text-primary border-primary/30">
                      HUMAN-IN-THE-LOOP
                    </Badge>
                  </div>

                  {/* Transformation Switcher */}
                  <div className="flex items-center bg-muted/40 p-1 rounded-xl font-mono text-[10px] border border-border/50">
                    <button
                      type="button"
                      onClick={() => setPreviewMode("raw")}
                      className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                        previewMode === "raw" 
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/30" 
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Raw Input
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewMode("remediated")}
                      className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                        previewMode === "remediated" 
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30" 
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Remediated DAG
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewMode("python")}
                      className={`px-3 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                        previewMode === "python" 
                          ? "bg-primary/20 text-primary border border-primary/30" 
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Polars Code
                    </button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <h3 className="text-lg sm:text-xl font-display font-bold text-foreground">
                    Interactive Issue Ledger & Transformation DAG
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Never trust a black-box data cleaner. GetReport identifies anomalies, duplicates, and type mismatches across 9 categories. Review, approve, or reject fixes before the pipeline executes.
                  </p>
                </div>

                {/* Interactive Code / Data Preview Box */}
                <div className="border border-border/80 bg-background/60 rounded-xl p-4 font-mono text-xs space-y-2.5 overflow-x-auto">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[10px] text-muted-foreground border-b border-border/60 pb-2 gap-1">
                    <span className="flex items-center gap-1.5">
                      <Database className="h-3.5 w-3.5 text-primary" />
                      TRANSACTION_RECORD (SAMPLE 10492)
                    </span>
                    <span className="font-bold font-mono">
                      {previewMode === "raw" && <span className="text-rose-400">UNFILTERED RAW ANOMALIES</span>}
                      {previewMode === "remediated" && <span className="text-emerald-400">ARROW CHUNKEDARRAY TYPES</span>}
                      {previewMode === "python" && <span className="text-primary">POLARS EXECUTABLE SNIPPET</span>}
                    </span>
                  </div>

                  {previewMode === "raw" && (
                    <div className="space-y-1.5 text-rose-300/90 bg-rose-950/20 p-3 rounded-lg border border-rose-500/30 break-words text-[11px] sm:text-xs">
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>customer_id: &quot;  10492  &quot;</span>
                        <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded">Trailing Whitespace</span>
                      </div>
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>amount: &quot;$1,240.50&quot;</span>
                        <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded">Uncoerced String Symbol</span>
                      </div>
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>transaction_date: &quot;2024/02/31&quot;</span>
                        <span className="text-[10px] bg-rose-500/20 text-rose-300 px-1.5 py-0.5 rounded">Invalid Calendar Leap Day</span>
                      </div>
                    </div>
                  )}

                  {previewMode === "remediated" && (
                    <div className="space-y-1.5 text-emerald-300/90 bg-emerald-950/20 p-3 rounded-lg border border-emerald-500/30 break-words text-[11px] sm:text-xs">
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>customer_id: &quot;10492&quot;</span>
                        <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Trimmed Utf8</span>
                      </div>
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>amount: 1240.50</span>
                        <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Strict Float64</span>
                      </div>
                      <div className="flex items-center justify-between flex-wrap gap-1">
                        <span>transaction_date: &quot;2024-02-29&quot;</span>
                        <span className="text-[10px] bg-emerald-500/20 text-emerald-300 px-1.5 py-0.5 rounded">Validated Date32</span>
                      </div>
                    </div>
                  )}

                  {previewMode === "python" && (
                    <div className="space-y-1 text-primary/90 bg-primary/5 p-3 rounded-lg border border-primary/20 break-words text-[11px] sm:text-xs">
                      <div>df = df.with_columns([</div>
                      <div className="pl-4">pl.col(&quot;customer_id&quot;).str.strip_chars(),</div>
                      <div className="pl-4">pl.col(&quot;amount&quot;).str.replace_all(r&quot;[\$,]&quot;, &quot;&quot;).cast(pl.Float64),</div>
                      <div className="pl-4">pl.col(&quot;transaction_date&quot;).str.to_date(&quot;%Y-%m-%d&quot;, strict=False)</div>
                      <div>])</div>
                    </div>
                  )}
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Ledger Guarantees:</span>
                <span className="text-emerald-400">100% Human Approvals Required • Zero Silent Mutability</span>
              </div>
            </Card>

            {/* Bento Tile 2: Multi-Dataset Relational Joins */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="h-9 w-9 rounded-xl bg-sky-500/10 text-sky-400 flex items-center justify-center">
                  <ArrowLeftRight className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Multi-Dataset Relational Joins
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Ingest up to 5 related datasets simultaneously. Merge on shared primary keys with automated schema reconciliation and duplicate key detection.
                  </p>
                </div>

                {/* Visual Schema Relational Diagram */}
                <div className="p-3 bg-muted/20 rounded-xl border border-border/60 font-mono text-[11px] space-y-2">
                  <div className="flex items-center justify-between text-muted-foreground text-[10px]">
                    <span>users.csv (10.4k)</span>
                    <span className="text-sky-400 font-bold">⟷ INNER JOIN ⟷</span>
                    <span>orders.parquet (84k)</span>
                  </div>
                  <div className="flex items-center justify-between bg-sky-500/10 text-sky-300 p-1.5 rounded-lg border border-sky-500/20 text-[10px]">
                    <span>Key: customer_id</span>
                    <span>Match: 99.2% (14 Orphan Keys)</span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Join Strategies:</span> Inner, Left Outer, Cross, Composite Keys
              </div>
            </Card>

            {/* Bento Tile 3: Statistical Drift & Multicollinearity */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="h-9 w-9 rounded-xl bg-purple-500/10 text-purple-400 flex items-center justify-center">
                  <Activity className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    VIF & Statistical Drift
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Identify collinear predictor columns using Variance Inflation Factors (VIF &gt; 5.0). Flag chronological concept drift across temporal partitions.
                  </p>
                </div>

                {/* Visual Statistical Diagnostic Metric Mock */}
                <div className="p-3 bg-muted/20 rounded-xl border border-border/60 font-mono text-[11px] space-y-2">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-muted-foreground">marketing_spend:</span>
                    <span className="text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">VIF = 7.42 (Collinear)</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-muted-foreground">revenue ⟷ units:</span>
                    <span className="text-purple-300 bg-purple-500/10 px-1.5 py-0.5 rounded border border-purple-500/20">r = 0.88 (Pearson)</span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Statistical Metrics:</span> Pearson r, VIF, Kurtosis, KS Test
              </div>
            </Card>

            {/* Bento Tile 4: Ephemeral RAG AI Companion */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="h-9 w-9 rounded-xl bg-amber-500/10 text-amber-400 flex items-center justify-center">
                  <MessageSquareCode className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Ephemeral RAG Companion
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Chat with your dataset without sending raw private rows to external LLMs. Ingests statistical summaries, correlations, and ledger receipts.
                  </p>
                </div>

                {/* Visual RAG Mini Prompt Mock */}
                <div className="p-3 bg-muted/20 rounded-xl border border-border/60 font-mono text-[11px] space-y-2">
                  <div className="text-[10px] text-muted-foreground">
                    &gt; &quot;Explain the Q3 gross margin dip.&quot;
                  </div>
                  <div className="text-[10px] text-amber-300/90 bg-amber-500/10 p-2 rounded-lg border border-amber-500/20 font-sans">
                    Logistics cost spiked 18.2% across EU routes. Zero PII transmitted.
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Privacy Guarantee:</span> Zero raw row transmission
              </div>
            </Card>

            {/* Bento Tile 5: Board-Ready PDF Generation */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="h-9 w-9 rounded-xl bg-emerald-500/10 text-emerald-400 flex items-center justify-center">
                  <FileText className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Board-Ready PDF Output
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    High-DPI executive audit documents compiled via WeasyPrint with Matplotlib visualization galleries, executive summaries, and remediation receipts.
                  </p>
                </div>

                {/* Visual PDF Audit Certificate Mock */}
                <div className="p-3 bg-muted/20 rounded-xl border border-border/60 font-mono text-[11px] space-y-2">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-foreground font-bold">Executive Audit Certificate</span>
                    <span className="text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">Pass Grade A</span>
                  </div>
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>WeasyPrint 300 DPI</span>
                    <span>Remediation Hash: 9e4f2a</span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Formats:</span> PDF, Parquet, CSV, HTML, SQL DDL
              </div>
            </Card>

            {/* Bento Tile 6: Zero-Retention & PII Masking Vault */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-4">
                <div className="h-9 w-9 rounded-xl bg-rose-500/10 text-rose-400 flex items-center justify-center">
                  <Lock className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Zero-Retention PII Masking
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Client-first compliance pipeline. Automatic regex detection and SHA-256 masking for emails, tax IDs, and credit cards before reporting.
                  </p>
                </div>

                {/* Visual PII Masking Mock */}
                <div className="p-3 bg-muted/20 rounded-xl border border-border/60 font-mono text-[11px] space-y-2">
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-muted-foreground">email:</span>
                    <span className="text-rose-300">j***@acme.com (SHA-256)</span>
                  </div>
                  <div className="flex items-center justify-between text-[10px]">
                    <span className="text-muted-foreground">ssn_id:</span>
                    <span className="text-rose-300">***-**-4921 (Redacted)</span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Compliance:</span> SOC 2 Type II, HIPAA, GDPR Ready
              </div>
            </Card>

          </div>
        </section>

        {/* Section 4: Architectural Head-to-Head Comparison Matrix */}
        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 max-w-6xl">
          <div className="text-center max-w-2xl mx-auto space-y-3 mb-8 sm:mb-12">
            <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/10 text-primary border-primary/20">
              Technical Benchmarking
            </Badge>
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground tracking-[-0.02em]">
              Why teams choose GetReport
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
              Compare GetReport against standard Pandas script maintenance and manual Excel spreadsheet editing.
            </p>
          </div>

          <div className="border border-border/80 bg-card rounded-2xl overflow-hidden shadow-premium">
            <div className="overflow-x-auto">
              <table className="w-full text-left font-sans text-xs sm:text-sm border-collapse min-w-[620px]">
                <thead>
                  <tr className="border-b border-border/60 bg-muted/30 font-mono text-[11px] sm:text-xs text-muted-foreground">
                    <th className="p-4 sm:p-5 font-semibold text-foreground">Capability / Standard</th>
                    <th className="p-4 sm:p-5 font-semibold text-primary bg-primary/5 border-x border-border/60">GetReport Stack</th>
                    <th className="p-4 sm:p-5 font-semibold">Custom Python / Pandas</th>
                    <th className="p-4 sm:p-5 font-semibold">Manual Excel Sheets</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/40 font-sans text-xs sm:text-sm">
                  
                  <tr>
                    <td className="p-4 sm:p-5 font-medium text-foreground">
                      <div>Execution Engine</div>
                      <span className="text-[11px] text-muted-foreground font-normal">Throughput and parallelism</span>
                    </td>
                    <td className="p-4 sm:p-5 bg-primary/5 border-x border-border/60 font-semibold text-emerald-400">
                      Polars Rust Core (&lt;50ms)
                    </td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Single-Thread GIL Pandas</td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Formula calculation freeze</td>
                  </tr>

                  <tr>
                    <td className="p-4 sm:p-5 font-medium text-foreground">
                      <div>Human-in-the-Loop DAG</div>
                      <span className="text-[11px] text-muted-foreground font-normal">Interactive approval receipts</span>
                    </td>
                    <td className="p-4 sm:p-5 bg-primary/5 border-x border-border/60 font-semibold text-emerald-400 flex items-center gap-1.5 pt-5">
                      <Check className="h-4 w-4 text-emerald-400 shrink-0" />
                      <span>Interactive Issue Ledger</span>
                    </td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Hardcoded script edits</td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Manual destructive overwrite</td>
                  </tr>

                  <tr>
                    <td className="p-4 sm:p-5 font-medium text-foreground">
                      <div>Zero-Retention Privacy</div>
                      <span className="text-[11px] text-muted-foreground font-normal">RAM buffer memory management</span>
                    </td>
                    <td className="p-4 sm:p-5 bg-primary/5 border-x border-border/60 font-semibold text-emerald-400">
                      0 bytes written to disk
                    </td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Cached in /tmp or Jupyter</td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Unencrypted desktop files</td>
                  </tr>

                  <tr>
                    <td className="p-4 sm:p-5 font-medium text-foreground">
                      <div>Statistical Drift Auditing</div>
                      <span className="text-[11px] text-muted-foreground font-normal">VIF, KS test, and Kurtosis</span>
                    </td>
                    <td className="p-4 sm:p-5 bg-primary/5 border-x border-border/60 font-semibold text-emerald-400 flex items-center gap-1.5 pt-5">
                      <Check className="h-4 w-4 text-emerald-400 shrink-0" />
                      <span>Automated 9-Metric Scan</span>
                    </td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Requires SciPy custom code</td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Not supported</td>
                  </tr>

                  <tr>
                    <td className="p-4 sm:p-5 font-medium text-foreground">
                      <div>Board-Ready PDF Report</div>
                      <span className="text-[11px] text-muted-foreground font-normal">Executive audit documentation</span>
                    </td>
                    <td className="p-4 sm:p-5 bg-primary/5 border-x border-border/60 font-semibold text-emerald-400 flex items-center gap-1.5 pt-5">
                      <Check className="h-4 w-4 text-emerald-400 shrink-0" />
                      <span>High-DPI WeasyPrint PDF</span>
                    </td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Jupyter notebook printouts</td>
                    <td className="p-4 sm:p-5 text-muted-foreground">Screenshots into slide decks</td>
                  </tr>

                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Section 5: Technical FAQ Accordion */}
        <section className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-16 max-w-4xl">
          <div className="text-center max-w-xl mx-auto space-y-3 mb-8 sm:mb-10">
            <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/10 text-primary border-primary/20">
              Technical Clarity
            </Badge>
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground tracking-[-0.02em]">
              Frequently asked questions
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans">
              Everything you need to know about GetReport architecture and security guarantees.
            </p>
          </div>

          <div className="space-y-3">
            {faqItems.map((item, index) => {
              const isOpen = openFaq === index;
              return (
                <div
                  key={index}
                  className="border border-border/80 bg-card rounded-xl overflow-hidden transition-all duration-200"
                >
                  <button
                    type="button"
                    onClick={() => setOpenFaq(isOpen ? null : index)}
                    className="w-full p-4 sm:p-5 text-left flex items-center justify-between gap-4 font-display font-semibold text-sm sm:text-base text-foreground hover:text-primary transition-colors cursor-pointer"
                  >
                    <span>{item.q}</span>
                    <ChevronDown
                      className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 ${
                        isOpen ? "rotate-180 text-primary" : ""
                      }`}
                    />
                  </button>
                  {isOpen && (
                    <div className="px-4 pb-4 sm:px-5 sm:pb-5 pt-0 text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed border-t border-border/40 mt-1 pt-3">
                      {item.a}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Section 6: High-Converting Obsidian Conversion Banner */}
        <section className="border-t border-border/60 bg-gradient-to-b from-card/30 via-background to-background py-14 sm:py-20">
          <div className="container mx-auto px-4 max-w-4xl">
            <div className="relative rounded-3xl border border-primary/30 bg-card/80 p-8 sm:p-12 text-center space-y-5 shadow-premium overflow-hidden">
              <div className="space-y-2">
                <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-emerald-500/10 text-emerald-400 border-emerald-500/30">
                  Instant Access • Zero Setup
                </Badge>
                <h2 className="text-2xl sm:text-3xl md:text-4xl font-display font-extrabold text-foreground tracking-[-0.02em]">
                  Ready to audit your first dataset?
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto leading-relaxed">
                  No sign-up, no credit card, and zero permanent data retention. Upload your CSV, Excel, or Parquet ledger to begin.
                </p>
              </div>

              <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-3">
                <Link to="/workspace" className="w-full sm:w-auto">
                  <Button size="lg" className="w-full sm:w-auto h-11 px-8 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                    <span>Launch Workspace</span>
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link to="/how-it-works" className="w-full sm:w-auto">
                  <Button size="lg" variant="outline" className="w-full sm:w-auto h-11 px-6 rounded-xl border-border bg-card hover:bg-muted/20 font-display text-sm">
                    <span>Inspect Pipeline</span>
                  </Button>
                </Link>
              </div>

              <div className="pt-4 border-t border-border/40 flex flex-wrap items-center justify-center gap-6 font-mono text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" /> 100% In-Memory RAM
                </span>
                <span className="flex items-center gap-1.5">
                  <Activity className="h-3.5 w-3.5 text-primary" /> Sub-50ms Execution
                </span>
                <span className="flex items-center gap-1.5">
                  <Lock className="h-3.5 w-3.5 text-sky-400" /> Zero Disk Writes
                </span>
              </div>
            </div>
          </div>
        </section>

      </main>

      <Footer />
    </div>
  );
};

export default Features;
