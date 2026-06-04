import { BarChart3, Brain, FileText, Layout, Wand2, ShieldCheck, Gauge, HelpCircle, Zap, Target, ArrowRight, Sparkles } from "lucide-react";
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
          <Card className="md:col-span-2 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group overflow-hidden relative">
            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/5 rounded-bl-full pointer-events-none" />
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Gauge className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
                  <span>Column Confidence Scores</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono">NEW</Badge>
                </h3>
                <p className="text-sm text-muted-foreground leading-relaxed max-w-md">
                  Every column undergoes multi-dimensional testing—Completeness, Consistency, Validity, and Stability. Rated from A to F so you know exactly what is clean.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border/50 bg-muted/30 rounded-xl p-3.5 font-mono text-[10px] space-y-2 max-w-md">
              <div className="flex justify-between items-center border-b border-border/30 pb-2">
                <span className="font-bold text-foreground">Variable trust ledger</span>
                <span className="text-muted-foreground">confidence %</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 h-5 px-1 rounded text-[8px] font-bold">A</Badge> customer_id</span>
                <span className="text-emerald-500 font-semibold">100.0%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-blue-500/15 text-blue-600 dark:text-blue-400 h-5 px-1 rounded text-[8px] font-bold">B</Badge> total_purchase</span>
                <span className="text-blue-500 font-semibold">89.4%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center gap-1.5"><Badge className="bg-red-500/15 text-red-650 dark:text-red-400 h-5 px-1 rounded text-[8px] font-bold">F</Badge> referral_code</span>
                <span className="text-red-500 font-semibold">18.5%</span>
              </div>
            </div>
          </Card>

          {/* Card 2: Why I Did X Explanations (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <HelpCircle className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                  <span>Explainable Auditing</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono">NEW</Badge>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Total transparency for every step in the pipeline. Understand exactly why an analysis ran or was skipped.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border/50 bg-muted/30 rounded-xl p-3 font-mono text-[9px] space-y-1">
              <div className="text-[8px] text-muted-foreground font-bold tracking-wider uppercase">Pipeline Logs</div>
              <div className="text-amber-500">▶ SKIP: Time-Series Integrity check</div>
              <div className="text-muted-foreground/80 pl-2">↳ Cause: Column "date" does not exist</div>
            </div>
          </Card>

          {/* Card 3: Semantic Intelligence (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Target className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground flex items-center gap-2">
                  <span>Semantic Role AI</span>
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[9px] font-mono">NEW</Badge>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Automatic schema detection. Classifies variables as categorical targets, timestamps, continuous metrics, or indicators.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 flex gap-2 font-mono text-[9px]">
              <span className="px-1.5 py-0.5 bg-muted rounded border border-border/60 text-muted-foreground">email ➔ ID</span>
              <span className="px-1.5 py-0.5 bg-primary/5 rounded border border-primary/20 text-primary">gpa ➔ NUMERIC</span>
            </div>
          </Card>

          {/* Card 4: Intelligent Hygiene (col-span-2) */}
          <Card className="md:col-span-2 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group overflow-hidden relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Wand2 className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground">Intelligent Quality Healing</h3>
                <p className="text-sm text-muted-foreground leading-relaxed max-w-md">
                  Lightning-fast Polars engine cleans your dataset asynchronously. Resolves nulls, drops duplicates, and standardizes type representations.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border/50 bg-muted/30 rounded-xl p-3.5 font-mono text-[10px] space-y-1.5 max-w-md">
              <div className="text-[8px] text-muted-foreground font-bold uppercase">Operations report</div>
              <div className="text-foreground">✓ Dropped <span className="text-emerald-500 font-semibold">142</span> duplicate rows</div>
              <div className="text-foreground">✓ Replaced <span className="text-emerald-500 font-semibold">12</span> numerical NaNs with Mean</div>
            </div>
          </Card>

          {/* Card 5: Auto-Visualization (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <BarChart3 className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-lg font-display font-bold text-foreground">Auto-Visualizer</h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  Frictionless Matplotlib rendering. Produces correlation heatmaps, histogram spreads, and bivariate scatter trends automatically.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 flex items-end justify-center h-16 gap-1.5 border-b border-border/60 pb-2">
              <div className="w-4 bg-primary/30 h-1/3 rounded-t-sm" />
              <div className="w-4 bg-primary/50 h-2/3 rounded-t-sm" />
              <div className="w-4 bg-primary h-full rounded-t-sm" />
              <div className="w-4 bg-primary/70 h-1/2 rounded-t-sm" />
            </div>
          </Card>

          {/* Card 6: Board-Ready Reports (col-span-1) */}
          <Card className="md:col-span-1 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group relative">
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
            <div className="mt-8 flex justify-center">
              <div className="border border-border/70 rounded bg-muted/30 p-1.5 flex items-center gap-1.5 shadow-sm text-[9px] font-mono">
                <FileText className="h-3 w-3 text-red-500" />
                <span>executive_summary.pdf</span>
              </div>
            </div>
          </Card>

          {/* Card 7: Context-Aware AI (col-span-2) */}
          <Card className="md:col-span-2 border border-border bg-card/45 backdrop-blur-sm shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/25 transition-all duration-300 flex flex-col justify-between group overflow-hidden relative">
            <div className="space-y-4">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                <Brain className="h-5 w-5" />
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-display font-bold text-foreground">RAG Insights Partner</h3>
                <p className="text-sm text-muted-foreground leading-relaxed max-w-md">
                  Secure RAG system. Chat with your dataset ephemerally. Extract summaries, anomalies, and cleaning options without leaking raw records.
                </p>
              </div>
            </div>

            {/* Visual Micro Mockup */}
            <div className="mt-8 border border-border/50 bg-muted/30 rounded-xl p-3 max-w-md space-y-2 text-[10px] font-sans">
              <div className="flex gap-2">
                <Badge className="bg-primary text-primary-foreground h-5 px-1 rounded text-[8px] shrink-0 font-mono">User</Badge>
                <p className="text-foreground/80 leading-snug">What does the purchasing distribution imply?</p>
              </div>
              <div className="flex gap-2 border-t border-border/30 pt-2">
                <Badge className="bg-muted text-muted-foreground border h-5 px-1 rounded text-[8px] shrink-0 font-mono">AI</Badge>
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
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">PII Masking By Design</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Sensitive strings, email handles, and phone coordinates are automatically parsed and masked before model analysis.</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
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
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
                  <div>
                    <h4 className="text-xs sm:text-sm font-semibold text-foreground">Polars Query Optimization</h4>
                    <p className="text-xs text-muted-foreground leading-relaxed">Processes datasets with 200k+ rows in sub-second times using multi-threaded Rust query structures.</p>
                  </div>
                </li>
                <li className="flex gap-3">
                  <div className="h-5 w-5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 text-xs font-bold">✓</div>
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
          <Link to="/">
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
