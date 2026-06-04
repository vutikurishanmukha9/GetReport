import { BarChart3, Brain, FileText, Layout, Wand2, ShieldCheck, ShieldAlert, Gauge, HelpCircle, Zap, Target, ArrowRight, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export const Features = () => {
  return (
    <div className="min-h-screen bg-background animate-in fade-in duration-500">
      
      {/* Editorial Hero Banner */}
      <div className="border-b border-border/60 bg-background">
        <div className="container mx-auto px-4 py-20 text-center space-y-6">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono">
            <Zap className="h-3.5 w-3.5" />
            <span>Operational Excellence</span>
          </div>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold tracking-tight text-foreground max-w-4xl mx-auto leading-[1.05] uppercase">
            Designed for high performance data audit.
          </h1>
          
          <p className="text-sm sm:text-base md:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
            A unified suite built for data scientists and developers. Automate column grading, explain analytical decisions, and generate board-ready reports.
          </p>

          <div className="pt-2">
            <Link to="/">
              <Button size="lg" className="h-11 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all">
                <span>Start Free Audit</span>
                <ArrowRight className="ml-1.5 h-4 w-4" />
              </Button>
            </Link>
          </div>
        </div>
      </div>

      {/* Bento Grid Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="text-center max-w-xl mx-auto mb-16 space-y-2">
          <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground">Core Capabilities</h2>
          <p className="text-xs sm:text-sm text-muted-foreground">Every component is engineered for clarity, transparency, and sub-second operations.</p>
        </div>

        {/* Bento Mosaic Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          
          {/* Card 1: Column Confidence Scores (col-span-2) */}
          <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col md:flex-row gap-6 md:items-center justify-between group overflow-hidden relative">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-bl-full pointer-events-none" />
            <div className="space-y-4 flex-1">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Gauge className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
                  <span>Column Confidence Scores</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono rounded-full">NEW</Badge>
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Every column undergoes multi-dimensional testing—Completeness, Consistency, Validity, and Stability. Rated from A to F so you know exactly what is clean.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="w-full md:w-[320px] lg:w-[380px] border border-border bg-white rounded-xl p-4 font-mono text-[10px] space-y-2 shadow-xs shrink-0 self-stretch flex flex-col justify-center">
              <div className="flex justify-between items-center border-b border-border pb-2">
                <span className="font-bold text-foreground uppercase tracking-wider text-[8px] text-muted-foreground">Variable trust ledger</span>
                <span className="text-muted-foreground">confidence %</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-emerald-50 text-emerald-700 border border-emerald-250 h-5 px-2 rounded-full text-[8px] font-bold">A</Badge> customer_id</span>
                <span className="text-emerald-600 font-semibold">100.0%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-blue-50 text-blue-700 border border-blue-250 h-5 px-2 rounded-full text-[8px] font-bold">B</Badge> total_purchase</span>
                <span className="text-blue-600 font-semibold">89.4%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-red-50 text-red-750 border border-red-250 h-5 px-2 rounded-full text-[8px] font-bold">F</Badge> referral_code</span>
                <span className="text-red-600 font-semibold">18.5%</span>
              </div>
            </div>
          </Card>

          {/* Card 2: Why I Did X Explanations (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <HelpCircle className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                  <span>Explainable Auditing</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono rounded-full">NEW</Badge>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Total transparency for every step in the pipeline. Understand exactly why an analysis ran or was skipped.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border bg-white rounded-xl p-3.5 font-mono text-[9px] space-y-2 shadow-xs flex-1 flex flex-col justify-center">
              <div className="text-[8px] text-muted-foreground font-bold tracking-wider uppercase border-b border-border pb-1.5 mb-1">Pipeline Logs</div>
              <div className="space-y-1.5">
                <div className="text-emerald-700 font-semibold flex items-center gap-1"><span>▶</span> RUN: Outlier Detection</div>
                <div className="text-muted-foreground/80 pl-2">↳ Result: 12 anomalies flagged</div>
                <div className="text-amber-750 font-semibold flex items-center gap-1"><span>▶</span> SKIP: Time-Series Integrity</div>
                <div className="text-muted-foreground/80 pl-2">↳ Cause: Column "date" missing</div>
              </div>
            </div>
          </Card>

          {/* Card 3: Semantic Intelligence (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Target className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                  <span>Semantic Role AI</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono rounded-full">NEW</Badge>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Automatic schema detection. Classifies variables as targets, timestamps, metrics, or tags.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border bg-white rounded-xl p-3.5 font-mono text-[9px] space-y-2 shadow-xs flex-1 flex flex-col justify-center">
              <div className="text-[8px] text-muted-foreground font-bold tracking-wider uppercase border-b border-border pb-1.5 mb-1">Schema Detection</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-foreground font-semibold">email</span>
                  <span className="px-1.5 py-0.5 bg-muted text-muted-foreground rounded border border-border text-[8px] font-bold">ID / STRING</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-foreground font-semibold">gpa</span>
                  <span className="px-1.5 py-0.5 bg-primary/5 text-primary rounded border border-primary/20 text-[8px] font-bold">MEASURE / FLOAT</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-foreground font-semibold">timestamp</span>
                  <span className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded border border-blue-200 text-[8px] font-bold">DIMENSION / DATE</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Card 4: Intelligent Hygiene (col-span-2) */}
          <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col md:flex-row gap-6 md:items-center justify-between group overflow-hidden relative">
            <div className="space-y-4 flex-1">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Wand2 className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground">Intelligent Quality Healing</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Lightning-fast Polars engine cleans your dataset asynchronously. Resolves nulls, drops duplicates, and standardizes type representations.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="w-full md:w-[320px] lg:w-[380px] border border-border bg-white rounded-xl p-4 font-mono text-[10px] space-y-2 shadow-xs shrink-0 self-stretch flex flex-col justify-center">
              <div className="text-[8px] text-muted-foreground font-bold uppercase tracking-wider border-b border-border pb-1.5 mb-1">Operations report</div>
              <div className="text-foreground flex items-center gap-1.5">
                <span className="text-emerald-600 font-bold">✓</span>
                <span>Dropped <span className="text-emerald-700 font-semibold">142</span> duplicate rows</span>
              </div>
              <div className="text-foreground flex items-center gap-1.5">
                <span className="text-emerald-600 font-bold">✓</span>
                <span>Replaced <span className="text-emerald-700 font-semibold">12</span> numerical NaNs with Mean</span>
              </div>
              <div className="text-foreground flex items-center gap-1.5">
                <span className="text-emerald-600 font-bold">✓</span>
                <span>Coerced <span className="text-emerald-700 font-semibold">3</span> datetime column formats</span>
              </div>
            </div>
          </Card>

          {/* Card 5: Auto-Visualization (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <BarChart3 className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground">Auto-Visualizer</h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Frictionless Matplotlib rendering. Produces correlation heatmaps, histograms, and scatter plots.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border bg-white rounded-xl p-3.5 shadow-xs space-y-2 flex-1 flex flex-col justify-center">
              <div className="flex justify-between items-center border-b border-border pb-1.5 text-[8px] font-mono text-muted-foreground font-bold uppercase tracking-wider">
                <span>Correlation Matrix</span>
                <span className="text-primary font-semibold">r = 0.89</span>
              </div>
              <div className="flex items-end justify-center h-16 gap-2 pt-2">
                <div className="w-5 bg-primary/20 h-1/3 rounded-t-sm relative group flex items-center justify-center">
                  <span className="absolute -top-4 text-[7px] font-mono text-muted-foreground">0.3</span>
                </div>
                <div className="w-5 bg-primary/40 h-2/3 rounded-t-sm relative group flex items-center justify-center">
                  <span className="absolute -top-4 text-[7px] font-mono text-muted-foreground">0.6</span>
                </div>
                <div className="w-5 bg-primary h-full rounded-t-sm relative group flex items-center justify-center">
                  <span className="absolute -top-4 text-[7px] font-mono text-primary font-bold">1.0</span>
                </div>
                <div className="w-5 bg-primary/70 h-1/2 rounded-t-sm relative group flex items-center justify-center">
                  <span className="absolute -top-4 text-[7px] font-mono text-muted-foreground">0.5</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Card 6: Board-Ready Reports (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <FileText className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground">Board-Ready PDFs</h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Export complete summaries. Highly formatted PDFs featuring grades, distributions, and LLM advice.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border bg-white rounded-xl p-3.5 shadow-xs space-y-2 flex-1 flex flex-col justify-center">
              <div className="flex justify-between items-center border-b border-border pb-1.5 text-[8px] font-mono text-muted-foreground font-bold uppercase tracking-wider">
                <span>Generated Exports</span>
                <span className="text-emerald-600">PDF Ready</span>
              </div>
              <div className="space-y-1.5 font-mono text-[9px]">
                <div className="flex items-center justify-between p-1.5 bg-muted/30 border border-border/60 rounded">
                  <span className="flex items-center gap-1.5">
                    <FileText className="h-3 w-3 text-red-600 shrink-0" />
                    <span className="text-foreground">executive_summary.pdf</span>
                  </span>
                  <span className="text-[7px] text-muted-foreground">1.2 MB</span>
                </div>
                <div className="flex items-center justify-between p-1.5 bg-muted/30 border border-border/60 rounded">
                  <span className="flex items-center gap-1.5">
                    <FileText className="h-3 w-3 text-red-600 shrink-0" />
                    <span className="text-foreground">data_distribution.pdf</span>
                  </span>
                  <span className="text-[7px] text-muted-foreground">840 KB</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Card 8: Active Issue Ledger (col-span-1, row-span-2) */}
          <Card className="md:col-span-1 md:row-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col gap-6 group relative overflow-hidden">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <ShieldAlert className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                  <span>Active Issue Ledger</span>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Real-time tracking of data quality violations. Flag anomalies by severity, review auto-suggested cures, and audit actions.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup - Vertical List */}
            <div className="space-y-3 font-mono text-[9px]">
              <div className="text-[8px] text-muted-foreground font-bold tracking-wider uppercase border-b border-border pb-1.5 mb-1">
                Audit Trail & Issues
              </div>
              
              <div className="p-2 bg-white rounded-lg border border-border space-y-1.5 shadow-xs">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground">Completeness</span>
                  <Badge className="bg-red-50 text-red-750 border border-red-250 h-4 px-1.5 text-[7px] font-bold">HIGH</Badge>
                </div>
                <p className="text-muted-foreground leading-normal text-[8px]">"email" has 18.5% nulls.</p>
                <div className="text-primary font-medium text-[8px] pt-0.5 border-t border-dashed border-border flex justify-between">
                  <span>Action: Impute Mode</span>
                  <span className="text-muted-foreground/80">Pending</span>
                </div>
              </div>

              <div className="p-2 bg-white rounded-lg border border-border space-y-1.5 shadow-xs">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground">Consistency</span>
                  <Badge className="bg-amber-50 text-amber-700 border border-amber-250 h-4 px-1.5 text-[7px] font-bold">MED</Badge>
                </div>
                <p className="text-muted-foreground leading-normal text-[8px]">"age" has outliers (3.2x IQR).</p>
                <div className="text-emerald-700 font-medium text-[8px] pt-0.5 border-t border-dashed border-border flex justify-between">
                  <span>Action: Clip Outliers</span>
                  <span className="text-emerald-600 font-semibold">✓ Auto-Cured</span>
                </div>
              </div>

              <div className="p-2 bg-white rounded-lg border border-border space-y-1.5 shadow-xs">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground">Validity</span>
                  <Badge className="bg-blue-50 text-blue-750 border border-blue-250 h-4 px-1.5 text-[7px] font-bold">LOW</Badge>
                </div>
                <p className="text-muted-foreground leading-normal text-[8px]">"zip" contains invalid formats.</p>
                <div className="text-primary font-medium text-[8px] pt-0.5 border-t border-dashed border-border flex justify-between">
                  <span>Action: Standardize</span>
                  <span className="text-muted-foreground/80">Pending</span>
                </div>
              </div>

              <div className="p-2 bg-white rounded-lg border border-border space-y-1.5 shadow-xs">
                <div className="flex justify-between items-center">
                  <span className="font-semibold text-foreground">Uniqueness</span>
                  <Badge className="bg-amber-50 text-amber-700 border border-amber-250 h-4 px-1.5 text-[7px] font-bold">MED</Badge>
                </div>
                <p className="text-muted-foreground leading-normal text-[8px]">"id" contains duplicate keys.</p>
                <div className="text-emerald-700 font-medium text-[8px] pt-0.5 border-t border-dashed border-border flex justify-between">
                  <span>Action: Deduplicate</span>
                  <span className="text-emerald-600 font-semibold">✓ Auto-Cured</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Card 7: Context-Aware AI (col-span-2) */}
          <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col md:flex-row gap-6 md:items-center justify-between group overflow-hidden relative">
            <div className="space-y-4 flex-1">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Brain className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground">RAG Insights Partner</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Secure RAG system. Chat with your dataset ephemerally. Extract summaries, anomalies, and cleaning options without leaking raw records.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="w-full md:w-[320px] lg:w-[380px] border border-border bg-white rounded-xl p-4 space-y-2.5 text-[10px] font-sans shadow-xs shrink-0 self-stretch flex flex-col justify-center">
              <div className="flex gap-2">
                <Badge className="bg-primary text-primary-foreground h-5 px-2 rounded-full text-[8px] shrink-0 font-mono">User</Badge>
                <p className="text-foreground/80 leading-snug">What does the purchasing distribution imply?</p>
              </div>
              <div className="flex gap-2 border-t border-border pt-2.5">
                <Badge className="bg-muted text-muted-foreground border h-5 px-2 rounded-full text-[8px] shrink-0 font-mono">AI</Badge>
                <p className="text-muted-foreground/90 leading-snug">
                  The distribution is right-skewed [1], meaning a few high-value customers contribute 80% of revenue…
                </p>
              </div>
            </div>
          </Card>

        </div>
      </div>

      {/* Technical Specifications */}
      <div className="bg-muted/30 border-y border-border/60 py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16 space-y-2">
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground">Architecture Specs</h2>
            <p className="text-xs sm:text-sm text-muted-foreground">Built on multi-threaded compute engines and modern security protocols.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 max-w-4xl mx-auto">
            <div className="space-y-6">
              <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2 border-b border-border/40 pb-2">
                <ShieldCheck className="h-4.5 w-4.5 text-primary shrink-0" /> 
                <span>Security Foundations</span>
              </h3>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">PII Masking By Design</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Sensitive strings, email handles, and phone coordinates are automatically parsed and masked before model analysis.</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">Zero Retention Limits</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Datasets are processed directly inside ephemeral, volatile memory buffers and destroyed immediately upon pipeline exit.</p>
                  </div>
                </li>
              </ul>
            </div>

            <div className="space-y-6">
              <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2 border-b border-border/40 pb-2">
                <Gauge className="h-4.5 w-4.5 text-primary shrink-0" />
                <span>Performance Benchmarks</span>
              </h3>
              <ul className="space-y-4">
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">Polars Query Optimization</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Processes datasets with 200k+ rows in sub-second times using multi-threaded Rust query structures.</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">Asynchronous Task Workers</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Celery tasks execute heavy stats computations in isolated containers, maintaining UI responsiveness via WebSockets.</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Footer */}
      <div className="container mx-auto px-4 py-24 text-center space-y-6">
        <h2 className="text-3xl font-display font-bold text-foreground uppercase tracking-tight">Ready to audit?</h2>
        <div className="pt-2">
          <Link to="/workspace">
            <Button size="lg" className="h-11 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all">
              <span>Audit a Dataset</span>
              <ArrowRight className="ml-1.5 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>

    </div>
  );
};

export default Features;
