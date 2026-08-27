import { useState } from "react";
import { 
  Zap, ArrowRight, Layers, CheckCircle2, 
  Sparkles, FileText, ArrowLeftRight, Activity, Sliders, ShieldCheck,
  Code2
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

// Module scope pure function to satisfy React Doctor
const getConfidenceGrade = (score: number) => {
  if (score >= 90) return { grade: "A", label: "Production Ready", color: "bg-emerald-500/10 text-emerald-700 border-emerald-500/30" };
  if (score >= 80) return { grade: "B", label: "Minor Issues", color: "bg-blue-500/10 text-blue-700 border-blue-500/30" };
  if (score >= 70) return { grade: "C", label: "Needs Remediation", color: "bg-amber-500/10 text-amber-800 border-amber-500/30" };
  if (score >= 60) return { grade: "D", label: "High Risk", color: "bg-orange-500/10 text-orange-700 border-orange-500/30" };
  return { grade: "F", label: "Critical Anomalies", color: "bg-rose-500/10 text-rose-700 border-rose-500/30" };
};

export const Features = () => {
  // Confidence Grade Simulator State
  const [completeness, setCompleteness] = useState(96);
  const [consistency, setConsistency] = useState(92);
  const [validity, setValidity] = useState(88);
  const [stability, setStability] = useState(94);
  const [previewMode, setPreviewMode] = useState<"raw" | "remediated">("remediated");

  // Weighted Confidence Calculation: 35% Completeness + 25% Consistency + 25% Validity + 15% Stability
  const overallScore = Math.round(
    completeness * 0.35 +
    consistency * 0.25 +
    validity * 0.25 +
    stability * 0.15
  );

  const gradeInfo = getConfidenceGrade(overallScore);

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Editorial Hero Header with Asymmetric Benchmark Card */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-10 items-center">
              
              {/* Left Column: Editorial Headline & Value Proposition */}
              <div className="lg:col-span-7 space-y-4 sm:space-y-5 text-left">
                <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20 t-badge-shimmer">
                  <Zap className="h-3.5 w-3.5" />
                  <span>100% In-Memory Polars Engine</span>
                </div>
                
                <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold tracking-tight text-foreground leading-[1.08] uppercase">
                  Engineered for rigorous <span className="text-primary block mt-0.5">data quality audits.</span>
                </h1>
                
                <p className="text-sm sm:text-base text-muted-foreground max-w-xl leading-relaxed font-sans">
                  A high-throughput intelligence stack for data analysts, ML engineers, and decision-makers. Automate column trust scoring, approve transformation DAGs, and compile executive-ready PDF audit reports in seconds.
                </p>

                <div className="pt-1 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
                  <Link to="/workspace" className="w-full sm:w-auto">
                    <Button size="lg" className="w-full sm:w-auto h-11 px-6 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                      <span>Start Free Audit</span>
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </Link>
                  <Link to="/how-it-works" className="w-full sm:w-auto">
                    <Button size="lg" variant="outline" className="w-full sm:w-auto h-11 px-6 rounded-xl border-border bg-card hover:bg-muted/20 shadow-premium t-card-lift t-spring-press font-display text-sm">
                      <span>Pipeline Architecture</span>
                    </Button>
                  </Link>
                </div>

                {/* Micro Guarantee Metrics */}
                <div className="grid grid-cols-3 gap-3 pt-3 border-t border-border/40 font-mono text-xs">
                  <div>
                    <span className="block font-bold text-foreground text-sm sm:text-base">0 MB</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Permanent Storage</span>
                  </div>
                  <div>
                    <span className="block font-bold text-emerald-600 text-sm sm:text-base">&lt; 50ms</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Rust Streaming</span>
                  </div>
                  <div>
                    <span className="block font-bold text-primary text-sm sm:text-base">A to F</span>
                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider">Confidence Grading</span>
                  </div>
                </div>
              </div>

              {/* Right Column: Live Polars vs Pandas Benchmark Terminal */}
              <div className="lg:col-span-5 w-full">
                <Card className="border border-border/80 bg-card rounded-2xl shadow-premium overflow-hidden t-card-lift">
                  <CardHeader className="bg-muted/30 border-b border-border/60 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 font-mono text-xs font-semibold text-foreground">
                        <Activity className="h-4 w-4 text-primary" />
                        <span>Polars Zero-Copy Benchmark</span>
                      </div>
                      <Badge variant="outline" className="text-[10px] font-mono bg-emerald-500/10 text-emerald-700 border-emerald-500/20">
                        100k Rows Streamed
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-4 sm:p-5 space-y-3.5 font-mono text-xs">
                    <div className="space-y-2.5">
                      {/* Polars Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-[11px]">
                          <span className="font-bold text-foreground flex items-center gap-1.5">
                            <span className="h-2 w-2 rounded-full bg-emerald-500 t-pulse-dot" /> Polars (GetReport Engine)
                          </span>
                          <span className="font-bold text-emerald-600">42ms • 12MB RAM</span>
                        </div>
                        <div className="h-2.5 w-full bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-emerald-500 rounded-full w-[14%]" />
                        </div>
                      </div>

                      {/* Standard Pandas Bar */}
                      <div className="space-y-1 opacity-70">
                        <div className="flex justify-between text-[11px]">
                          <span className="text-muted-foreground">Standard Pandas Ingest</span>
                          <span className="text-muted-foreground">940ms • 148MB RAM</span>
                        </div>
                        <div className="h-2.5 w-full bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-muted-foreground/50 rounded-full w-[88%]" />
                        </div>
                      </div>
                    </div>

                    <div className="p-3 bg-muted/40 rounded-xl border border-border/40 text-[11px] font-sans text-muted-foreground space-y-1">
                      <strong className="text-foreground font-mono block">Zero-Copy Memory Guarantee:</strong>
                      <span>Files are memory-mapped into Polars ChunkedArrays. Intermediate buffers are purged automatically after execution.</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

            </div>
          </div>
        </div>

        {/* Section 2: Interactive Confidence Grade Simulator */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-6xl">
          <Card className="border-2 border-primary/20 bg-card rounded-2xl sm:rounded-3xl p-5 sm:p-7 shadow-premium space-y-6">
            
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border/60 pb-5">
              <div className="space-y-1.5">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-mono font-bold uppercase tracking-wider">
                  <Sliders className="h-3.5 w-3.5" />
                  <span>Interactive Algorithm Simulator</span>
                </div>
                <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
                  Column Confidence Scoring Engine
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground max-w-xl font-sans">
                  Adjust the four core quality dimensions below to simulate how our scoring engine grades individual tabular variables in real-time.
                </p>
              </div>

              {/* Dynamic Live Grade Result */}
              <div className={`p-3.5 sm:p-4 rounded-2xl border ${gradeInfo.color} flex items-center gap-3.5 shrink-0 transition-all duration-200`}>
                <div className="text-center">
                  <span className="text-3xl sm:text-4xl font-display font-extrabold block leading-none">
                    {gradeInfo.grade}
                  </span>
                  <span className="text-[10px] font-mono font-bold uppercase tracking-wider block mt-1">
                    Grade
                  </span>
                </div>
                <div className="border-l border-current/20 pl-3.5 space-y-0.5">
                  <span className="text-base sm:text-lg font-mono font-bold block">{overallScore}%</span>
                  <span className="text-xs font-sans font-medium block opacity-90">{gradeInfo.label}</span>
                </div>
              </div>
            </div>

            {/* Metric Sliders Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 font-mono text-xs">
              
              {/* Slider 1: Completeness */}
              <div className="space-y-2 p-3.5 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-1.5">
                    <CheckCircle2 className="h-3.5 w-3.5 text-primary" /> Completeness (Weight: 35%)
                  </span>
                  <span className="font-bold text-primary">{completeness}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={completeness}
                  onChange={(e) => setCompleteness(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-lg"
                  aria-label="Completeness percentage"
                />
                <span className="text-[10px] text-muted-foreground font-sans block">
                  Measures null values, empty strings, and masked NaN values.
                </span>
              </div>

              {/* Slider 2: Consistency */}
              <div className="space-y-2 p-3.5 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-1.5">
                    <Layers className="h-3.5 w-3.5 text-primary" /> Consistency (Weight: 25%)
                  </span>
                  <span className="font-bold text-primary">{consistency}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={consistency}
                  onChange={(e) => setConsistency(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-lg"
                  aria-label="Consistency percentage"
                />
                <span className="text-[10px] text-muted-foreground font-sans block">
                  Evaluates type cohesion, datetime format consistency, and schema anomalies.
                </span>
              </div>

              {/* Slider 3: Validity */}
              <div className="space-y-2 p-3.5 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-1.5">
                    <ShieldCheck className="h-3.5 w-3.5 text-primary" /> Validity (Weight: 25%)
                  </span>
                  <span className="font-bold text-primary">{validity}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={validity}
                  onChange={(e) => setValidity(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-lg"
                  aria-label="Validity percentage"
                />
                <span className="text-[10px] text-muted-foreground font-sans block">
                  Detects range violations, negative balances, and invalid regex domains.
                </span>
              </div>

              {/* Slider 4: Stability */}
              <div className="space-y-2 p-3.5 rounded-xl bg-muted/20 border border-border/60">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5 text-primary" /> Stability (Weight: 15%)
                  </span>
                  <span className="font-bold text-primary">{stability}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={stability}
                  onChange={(e) => setStability(Number(e.target.value))}
                  className="w-full accent-primary cursor-pointer h-2 bg-muted rounded-lg"
                  aria-label="Stability percentage"
                />
                <span className="text-[10px] text-muted-foreground font-sans block">
                  Flags distribution skewness, extreme kurtosis, and chronological drift.
                </span>
              </div>

            </div>
          </Card>
        </div>

        {/* Section 3: Core Capabilities Bento Grid */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-7xl space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-2">
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">
              Enterprise Feature Matrix
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
              Comprehensive tools designed to replace manual Python spreadsheet scripts with deterministic, auditable workflows.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 lg:gap-6">
            
            {/* Bento Tile 1: Interactive Issue Ledger (Large 2-col) */}
            <Card className="lg:col-span-2 border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-3.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
                      <Code2 className="h-4 w-4" />
                    </div>
                    <Badge variant="outline" className="text-[10px] font-mono uppercase tracking-wider bg-primary/5 text-primary border-primary/20">
                      HUMAN-IN-THE-LOOP
                    </Badge>
                  </div>

                  {/* Transformation Switcher */}
                  <div className="flex items-center bg-muted p-1 rounded-xl font-mono text-[10px]">
                    <button
                      type="button"
                      onClick={() => setPreviewMode("raw")}
                      className={`px-2.5 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                        previewMode === "raw" ? "bg-white text-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Raw Input
                    </button>
                    <button
                      type="button"
                      onClick={() => setPreviewMode("remediated")}
                      className={`px-2.5 py-1 rounded-lg font-semibold transition-all cursor-pointer ${
                        previewMode === "remediated" ? "bg-primary text-primary-foreground shadow-xs" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Remediated
                    </button>
                  </div>
                </div>

                <div className="space-y-1.5">
                  <h3 className="text-lg font-display font-bold text-foreground">
                    Interactive Issue Ledger & Transformation DAG
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Never trust a black-box data cleaner. GetReport identifies anomalies, duplicates, and type mismatches across 9 categories. Review, approve, or reject fixes before the pipeline executes.
                  </p>
                </div>

                {/* Interactive Code / Data Preview Mock */}
                <div className="border border-border/80 bg-muted/30 rounded-xl p-3.5 font-mono text-xs space-y-2 overflow-x-auto">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[10px] text-muted-foreground border-b border-border/60 pb-1.5 gap-1">
                    <span>TRANSACTION_RECORD (SAMPLE)</span>
                    <span className="text-primary font-bold">{previewMode === "raw" ? "UNFILTERED RECORD" : "REMEDIATED VIA POLARS"}</span>
                  </div>
                  {previewMode === "raw" ? (
                    <div className="space-y-1 text-rose-700 bg-rose-50/50 p-2.5 rounded-lg border border-rose-200 break-words text-[11px] sm:text-xs">
                      <div>customer_id: &quot;  10492  &quot; <span className="text-[10px] text-muted-foreground block sm:inline">(leading/trailing whitespace)</span></div>
                      <div>amount: &quot;$1,240.50&quot; <span className="text-[10px] text-muted-foreground block sm:inline">(string with currency symbol)</span></div>
                      <div>transaction_date: &quot;2024/02/31&quot; <span className="text-[10px] text-muted-foreground block sm:inline">(invalid leap day date)</span></div>
                    </div>
                  ) : (
                    <div className="space-y-1 text-emerald-800 bg-emerald-50/50 p-2.5 rounded-lg border border-emerald-200 break-words text-[11px] sm:text-xs">
                      <div>customer_id: &quot;10492&quot; <span className="text-[10px] text-emerald-600 block sm:inline">(trimmed utf-8)</span></div>
                      <div>amount: 1240.50 <span className="text-[10px] text-emerald-600 block sm:inline">(coerced Float64)</span></div>
                      <div>transaction_date: &quot;2024-02-29&quot; <span className="text-[10px] text-emerald-600 block sm:inline">(validated Date32)</span></div>
                    </div>
                  )}
                </div>
              </div>
            </Card>

            {/* Bento Tile 2: Multi-Dataset Relational Joins */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-3.5">
                <div className="h-9 w-9 rounded-xl bg-blue-500/10 text-blue-700 flex items-center justify-center">
                  <ArrowLeftRight className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Multi-Dataset Joins
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Ingest up to 5 related datasets simultaneously. Merge on shared primary keys with automated schema reconciliation and duplicate key detection.
                  </p>
                </div>
              </div>
              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Join Strategies:</span> Inner, Left Outer, Cross, Composite Keys
              </div>
            </Card>

            {/* Bento Tile 3: Statistical Drift & Multicollinearity */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-3.5">
                <div className="h-9 w-9 rounded-xl bg-purple-500/10 text-purple-700 flex items-center justify-center">
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
              </div>
              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Statistical Metrics:</span> Pearson r, VIF, Kurtosis, KS Test
              </div>
            </Card>

            {/* Bento Tile 4: Ephemeral RAG AI Companion */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-3.5">
                <div className="h-9 w-9 rounded-xl bg-amber-500/10 text-amber-700 flex items-center justify-center">
                  <Sparkles className="h-4 w-4" />
                </div>
                <div className="space-y-1.5">
                  <h3 className="text-base sm:text-lg font-display font-bold text-foreground">
                    Ephemeral RAG Companion
                  </h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed font-sans">
                    Chat with your dataset without sending raw private rows to external LLMs. Ingests statistical summaries, correlations, and ledger receipts.
                  </p>
                </div>
              </div>
              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Privacy Guarantee:</span> Zero raw row transmission
              </div>
            </Card>

            {/* Bento Tile 5: Board-Ready PDF Generation */}
            <Card className="border border-border bg-card rounded-2xl p-5 sm:p-7 flex flex-col justify-between shadow-premium t-card-lift">
              <div className="space-y-3.5">
                <div className="h-9 w-9 rounded-xl bg-primary/10 text-primary flex items-center justify-center">
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
              </div>
              <div className="pt-3 border-t border-border/40 font-mono text-[11px] text-muted-foreground">
                <span className="text-foreground font-semibold">Formats:</span> PDF, Parquet, CSV, HTML
              </div>
            </Card>

          </div>
        </div>

        {/* Section 4: Bottom Conversion Bar */}
        <div className="border-t border-border/60 bg-muted/20 py-10 sm:py-12">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Ready to audit your first dataset?
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              No sign-up, no credit card, and zero permanent data retention. Upload your CSV or Excel ledger to begin.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                  <span>Launch Workspace</span>
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

export default Features;
