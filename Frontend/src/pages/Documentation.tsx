import { BookOpen, FileSpreadsheet, AlertTriangle, Gauge, Brain, CheckCircle2, ShieldCheck, Lock, Layers, Terminal, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const Documentation = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1 whitespace-nowrap shrink-0">
              System Documentation & Guides
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              GetReport Technical Hub.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Comprehensive technical guides for Polars streaming ingestion, quality confidence scoring, security controls, and multi-format exports.
            </p>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-16 max-w-6xl space-y-16">
          
          {/* Quick Topic Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <a href="#supported-formats" className="p-6 border border-border bg-card rounded-2xl shadow-sm hover:shadow-md hover:border-primary/30 transition-all group space-y-3">
              <FileSpreadsheet className="h-8 w-8 text-emerald-600 group-hover:scale-105 transition-transform" />
              <h3 className="font-display font-bold text-foreground text-base">Supported File Formats</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">CSV, Excel, Parquet, TSV, JSONL, and compressed archive streaming capabilities.</p>
            </a>

            <a href="#confidence-scores" className="p-6 border border-border bg-card rounded-2xl shadow-sm hover:shadow-md hover:border-primary/30 transition-all group space-y-3">
              <Gauge className="h-8 w-8 text-primary group-hover:scale-105 transition-transform" />
              <h3 className="font-display font-bold text-foreground text-base">Column Confidence (A-F)</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">Detailed breakdown of Completeness, Consistency, Validity, and Stability scoring algorithms.</p>
            </a>

            <a href="#security" className="p-6 border border-border bg-card rounded-2xl shadow-sm hover:shadow-md hover:border-primary/30 transition-all group space-y-3">
              <ShieldCheck className="h-8 w-8 text-blue-600 group-hover:scale-105 transition-transform" />
              <h3 className="font-display font-bold text-foreground text-base">Architecture Specs & Security</h3>
              <p className="text-xs text-muted-foreground leading-relaxed">In-memory execution, HSTS, CSP, rate limiting, and zero permanent data retention protocol.</p>
            </a>
          </div>

          {/* Section 1: File Formats */}
          <section id="supported-formats" className="space-y-6 scroll-mt-28">
            <div className="flex items-center gap-3 border-b border-border pb-4">
              <FileSpreadsheet className="h-6 w-6 text-primary" />
              <h2 className="text-2xl font-display font-bold text-foreground uppercase tracking-tight">Supported Ingestion Formats</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
                <h4 className="font-display font-bold text-foreground text-sm flex items-center justify-between gap-2">
                  <span>Comma-Separated Values (.csv, .tsv)</span>
                  <Badge variant="secondary" className="font-mono text-[10px] whitespace-nowrap shrink-0">Polars Stream</Badge>
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Automatic buffer sampling for delimiter detection (<code>,</code>, <code>;</code>, <code>\t</code>, <code>|</code>) and encoding resolution (UTF-8, Latin-1, CP1252).
                </p>
              </Card>

              <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
                <h4 className="font-display font-bold text-foreground text-sm flex items-center justify-between gap-2">
                  <span>Excel Workbooks (.xlsx, .xls)</span>
                  <Badge variant="secondary" className="font-mono text-[10px] whitespace-nowrap shrink-0">calamine Engine</Badge>
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  High-speed Rust calamine parser with decompression limits to prevent zip-bomb archives and memory exhaustion.
                </p>
              </Card>

              <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
                <h4 className="font-display font-bold text-foreground text-sm flex items-center justify-between gap-2">
                  <span>Apache Parquet (.parquet)</span>
                  <Badge variant="secondary" className="font-mono text-[10px] whitespace-nowrap shrink-0">Native Rust</Badge>
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Zero-copy columnar format parsing with full column projection pushdown for sub-millisecond memory loading.
                </p>
              </Card>

              <Card className="p-6 border border-border bg-card shadow-sm space-y-3 rounded-2xl">
                <h4 className="font-display font-bold text-foreground text-sm flex items-center justify-between gap-2">
                  <span>JSON Lines (.json, .jsonl, .ndjson)</span>
                  <Badge variant="secondary" className="font-mono text-[10px] whitespace-nowrap shrink-0">Auto Schema</Badge>
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Streams line-delimited JSON with automatic nested structure flattening and scalar field coercion.
                </p>
              </Card>
            </div>
          </section>

          {/* Section 2: Confidence Scoring */}
          <section id="confidence-scores" className="space-y-6 scroll-mt-28">
            <div className="flex items-center gap-3 border-b border-border pb-4">
              <Gauge className="h-6 w-6 text-primary" />
              <h2 className="text-2xl font-display font-bold text-foreground uppercase tracking-tight">Column Confidence Grades (A-F)</h2>
            </div>

            <div className="space-y-4 text-xs sm:text-sm text-muted-foreground leading-relaxed">
              <p>
                Every column in your dataset receives an overall confidence score from 0.0% to 100.0% mapped directly to a letter grade:
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono text-center pt-2">
                <div className="p-3 rounded-xl border border-emerald-300 bg-emerald-50 text-emerald-800 font-bold">Grade A: ≥ 90%</div>
                <div className="p-3 rounded-xl border border-blue-300 bg-blue-50 text-blue-800 font-bold">Grade B: 80 - 89%</div>
                <div className="p-3 rounded-xl border border-amber-300 bg-amber-50 text-amber-800 font-bold">Grade C: 70 - 79%</div>
                <div className="p-3 rounded-xl border border-orange-300 bg-orange-50 text-orange-800 font-bold">Grade D: 60 - 69%</div>
                <div className="p-3 rounded-xl border border-red-300 bg-red-50 text-red-800 font-bold">Grade F: &lt; 60%</div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-4">
                <Card className="p-5 border border-border bg-card rounded-xl">
                  <h4 className="font-bold text-foreground mb-1 text-sm">Completeness (30% Weight)</h4>
                  <p className="text-xs">Percentage of non-null, non-empty cells in the column.</p>
                </Card>

                <Card className="p-5 border border-border bg-card rounded-xl">
                  <h4 className="font-bold text-foreground mb-1 text-sm">Consistency (25% Weight)</h4>
                  <p className="text-xs">Format uniformity and pattern adherence across string/date columns.</p>
                </Card>

                <Card className="p-5 border border-border bg-card rounded-xl">
                  <h4 className="font-bold text-foreground mb-1 text-sm">Validity (25% Weight)</h4>
                  <p className="text-xs">Values falling within plausible domain boundaries (e.g. positive prices).</p>
                </Card>

                <Card className="p-5 border border-border bg-card rounded-xl">
                  <h4 className="font-bold text-foreground mb-1 text-sm">Stability (20% Weight)</h4>
                  <p className="text-xs">Variance inflation and extreme outlier ratios (IQR & Z-Score).</p>
                </Card>
              </div>
            </div>
          </section>

          {/* Section 3: Architecture & Security */}
          <section id="security" className="space-y-6 scroll-mt-28">
            <div className="flex items-center gap-3 border-b border-border pb-4">
              <ShieldCheck className="h-6 w-6 text-emerald-600" />
              <h2 className="text-2xl font-display font-bold text-foreground uppercase tracking-tight">Architecture & Security Specifications</h2>
            </div>

            <div className="space-y-4">
              <Card className="p-6 border border-border bg-card rounded-2xl space-y-4 shadow-sm">
                <h3 className="font-display font-bold text-base text-foreground flex items-center gap-2">
                  <Lock className="h-5 w-5 text-primary" />
                  <span>In-Memory Ephemeral Execution</span>
                </h3>
                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
                  GetReport processes raw tabular data in temporary server memory. Files are never stored on persistent storage drives. All memory allocations, intermediate calculation dataframes, and synthesized PDF buffers are automatically garbage-collected and purged upon session completion (or after 1 hour of inactivity).
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-[11px] pt-2">
                  <div className="p-3 rounded-lg bg-muted/60 border border-border/40">
                    <span className="font-bold text-foreground block">HSTS Enforced</span>
                    <span className="text-muted-foreground">max-age=63072000</span>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/60 border border-border/40">
                    <span className="font-bold text-foreground block">CSP Enforced</span>
                    <span className="text-muted-foreground">frame-ancestors 'none'</span>
                  </div>
                  <div className="p-3 rounded-lg bg-muted/60 border border-border/40">
                    <span className="font-bold text-foreground block">Banner Masking</span>
                    <span className="text-muted-foreground">Server: GetReport-Secure</span>
                  </div>
                </div>
              </Card>
            </div>
          </section>

        </div>

      </main>

      <Footer />
    </div>
  );
};

export default Documentation;
