import { BarChart3, Brain, FileText, Layout, Wand2, ShieldCheck, ShieldAlert, Gauge, HelpCircle, Zap, Target, ArrowRight, Sparkles, Layers, Cpu, Database } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const Features = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Editorial Hero Banner */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-6 max-w-4xl">
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase tracking-wider font-mono border border-primary/20">
              <Zap className="h-3.5 w-3.5" />
              <span>Enterprise Feature Suite</span>
            </div>
            
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold tracking-tight text-foreground leading-[1.05] uppercase">
              Engineered for sub-second data audits.
            </h1>
            
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              A unified intelligence stack for analysts, engineers, and leadership. Automate column grading, explain analytical choices, track transformation DAGs, and generate board-ready reports.
            </p>

            <div className="pt-4 flex flex-wrap items-center justify-center gap-4">
              <Link to="/workspace">
                <Button size="lg" className="h-12 px-7 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all font-display font-semibold text-sm">
                  <span>Start Free Audit</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link to="/how-it-works">
                <Button size="lg" variant="outline" className="h-12 px-7 rounded-xl border-border bg-card hover:bg-muted/20 shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all font-display text-sm">
                  <span>Explore Methodology</span>
                </Button>
              </Link>
            </div>
          </div>
        </div>

        {/* Bento Grid Core Capabilities */}
        <div className="container mx-auto px-4 py-16 md:py-24 max-w-7xl">
          <div className="text-center max-w-xl mx-auto mb-16 space-y-3">
            <h2 className="text-3xl sm:text-4xl font-display font-bold text-foreground uppercase tracking-tight">Core Capabilities</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">Every subsystem is built with Rust-powered Polars memory management and deterministic fallback guarantees.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Card 1: Column Confidence Scores */}
            <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/30 transition-all duration-300 flex flex-col lg:flex-row gap-6 lg:items-center justify-between group relative overflow-hidden">
              <div className="space-y-4 flex-1">
                <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                  <Gauge className="h-5 w-5" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
                    <span>Column Confidence Scoring (A-F)</span>
                    <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20 text-[10px] font-mono rounded-full">CORE ENGINE</Badge>
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Evaluates completeness, consistency, validity, and statistical stability per variable. Instantly identify dirty columns before feeding data into ML models or executive dashboards.
                  </p>
                </div>
              </div>

              {/* Visual Micro Mockup */}
              <div className="w-full lg:w-[360px] border border-border bg-white rounded-xl p-4 font-mono text-[11px] space-y-2.5 shadow-xs shrink-0">
                <div className="flex justify-between items-center border-b border-border pb-2">
                  <span className="font-bold text-foreground uppercase tracking-wider text-[9px] text-muted-foreground">Confidence Ledger</span>
                  <span className="text-muted-foreground text-[10px]">grade</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2"><Badge className="bg-emerald-100 text-emerald-800 border-emerald-300 px-2 rounded font-bold text-[9px]">A</Badge> customer_id</span>
                  <span className="text-emerald-700 font-semibold">100%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2"><Badge className="bg-blue-100 text-blue-800 border-blue-300 px-2 rounded font-bold text-[9px]">B</Badge> total_purchase</span>
                  <span className="text-blue-700 font-semibold">89.4%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2"><Badge className="bg-red-100 text-red-800 border-red-300 px-2 rounded font-bold text-[9px]">F</Badge> referral_code</span>
                  <span className="text-red-700 font-semibold">18.5%</span>
                </div>
              </div>
            </Card>

            {/* Card 2: Interactive Issue Ledger */}
            <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/30 transition-all duration-300 flex flex-col justify-between group">
              <div className="space-y-4">
                <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                  <ShieldAlert className="h-5 w-5" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-display font-bold text-foreground">Issue Ledger</h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                    "Jira for Dirty Data". Surfaces 9 issue categories (missing values, duplicates, outliers, type mismatches) with approve/reject workflow control.
                  </p>
                </div>
              </div>
              <div className="mt-6 pt-4 border-t border-border/60 flex items-center justify-between text-xs font-mono text-muted-foreground">
                <span>9 Issue Categories</span>
                <span className="text-primary font-bold">1-Click Apply</span>
              </div>
            </Card>

            {/* Card 3: Multi-Dataset Relational Joins */}
            <Card className="md:col-span-1 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/30 transition-all duration-300 flex flex-col justify-between group">
              <div className="space-y-4">
                <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                  <Layers className="h-5 w-5" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-display font-bold text-foreground">Relational Join DAG</h3>
                  <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                    Merge up to 5 CSV/Excel datasets on key columns (`inner`, `left`, `right`, `outer`, `semi`, `anti`) with column collision renaming.
                  </p>
                </div>
              </div>
              <div className="mt-6 pt-4 border-t border-border/60 flex items-center justify-between text-xs font-mono text-muted-foreground">
                <span>Rust Streaming</span>
                <span className="text-primary font-bold">Polars Core</span>
              </div>
            </Card>

            {/* Card 4: Board-Ready PDF Generation */}
            <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-2xl p-8 hover:-translate-y-0.5 hover:border-primary/30 transition-all duration-300 flex flex-col lg:flex-row gap-6 lg:items-center justify-between group">
              <div className="space-y-4 flex-1">
                <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors">
                  <FileText className="h-5 w-5" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-display font-bold text-foreground">WeasyPrint Board-Ready PDF Reports</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    Generates executive-level reports with pre-compiled CSS styling, high-DPI compressed chart figures, statistical summaries, and audit trail lineage logs.
                  </p>
                </div>
              </div>

              <div className="border border-border bg-white rounded-xl p-4 font-mono text-[11px] space-y-2 shadow-xs shrink-0 w-full lg:w-[320px]">
                <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-bold border-b pb-1">Report Contents</div>
                <div className="text-foreground">✓ Executive Overview</div>
                <div className="text-foreground">✓ Quality Grade Matrix</div>
                <div className="text-foreground">✓ Matplotlib Charts</div>
                <div className="text-foreground">✓ Lineage Audit DAG</div>
              </div>
            </Card>

          </div>
        </div>

        {/* Feature Matrix */}
        <div className="border-t border-border/60 bg-muted/20 py-16 md:py-24">
          <div className="container mx-auto px-4 max-w-5xl">
            <div className="text-center mb-12 space-y-2">
              <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">Security & Performance Guarantees</h2>
              <p className="text-xs sm:text-sm text-muted-foreground">Built to handle sensitive datasets securely on modern cloud infrastructure.</p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="p-5 rounded-2xl border border-border bg-card shadow-sm space-y-2">
                <ShieldCheck className="h-6 w-6 text-emerald-600" />
                <h4 className="font-display font-bold text-sm text-foreground">In-Memory Security</h4>
                <p className="text-xs text-muted-foreground">Raw data is processed in isolated memory and automatically purged after session completion.</p>
              </div>

              <div className="p-5 rounded-2xl border border-border bg-card shadow-sm space-y-2">
                <Cpu className="h-6 w-6 text-primary" />
                <h4 className="font-display font-bold text-sm text-foreground">Multi-Threaded Polars</h4>
                <p className="text-xs text-muted-foreground">Sub-second vectorized string coercion and outlier scanning with minimal RAM overhead.</p>
              </div>

              <div className="p-5 rounded-2xl border border-border bg-card shadow-sm space-y-2">
                <Brain className="h-6 w-6 text-purple-600" />
                <h4 className="font-display font-bold text-sm text-foreground">Offline AI Fallback</h4>
                <p className="text-xs text-muted-foreground">Deterministic offline analyzer guarantees instant answers even if cloud LLM endpoints pause.</p>
              </div>

              <div className="p-5 rounded-2xl border border-border bg-card shadow-sm space-y-2">
                <Database className="h-6 w-6 text-blue-600" />
                <h4 className="font-display font-bold text-sm text-foreground">Multi-Format Export</h4>
                <p className="text-xs text-muted-foreground">Export remediated datasets as CSV, Parquet, JSONL, or standalone HTML audit logs.</p>
              </div>
            </div>

          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Features;
