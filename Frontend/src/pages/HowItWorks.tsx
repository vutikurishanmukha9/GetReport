import { UploadCloud, Search, FileDown, ArrowRight, Brain, Sparkles, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

const steps = [
  {
    id: "01",
    icon: UploadCloud,
    title: "Ingestion & Inferred Schemas",
    description: "Drop your CSV or Excel ledger. Our Polars parser streams the file in memory, automatically resolving character encodings, field separators, and primary column data types.",
    details: ["Supports CSV, XLSX, Parquet, TSV", "Auto-detects UTF-8, Latin-1 & CP1252", "Session-scoped processing"],
  },
  {
    id: "02",
    icon: Search,
    title: "Hygiene Scoring & Issue Ledger",
    description: "Our quality algorithms calculate completeness, stability, and validity. Surfacing 9 issue categories in an interactive ledger where you approve, reject, or modify fixes before execution.",
    details: ["A-F column confidence grades", "Interactive issue overrides", "Automated Winsorization & imputation"],
  },
  {
    id: "03",
    icon: Brain,
    title: "Statistical Auditing",
    description: "Executes Pearson correlations, extreme outlier checks, skewness tests, Kurtosis shapes, VIF multicollinearity, and time-series conceptual drift. Every test run produces structured logs explaining why it ran.",
    details: ["Why-I-Did-X transparency logs", "VIF multicollinearity metrics", "Pairwise correlation detection (|r| > 0.7)"],
  },
  {
    id: "04",
    icon: Sparkles,
    title: "AI Synthesis & RAG Context",
    description: "A semantic layer extracts the business domain (e.g. Sales, Education, Healthcare) and passes statistical summaries to our RAG engine, generating deep summaries and executive insights.",
    details: ["Multi-provider LLM fallback chain", "Deterministic offline AI fallback", "Confidential row protection"],
  },
  {
    id: "05",
    icon: FileDown,
    title: "Board-Ready PDF Compilation",
    description: "Compiles a formatted executive audit report via WeasyPrint, featuring high-DPI charts, complete stats tables, and recommendations. Available as PDF, CSV, Parquet, or HTML.",
    details: ["Matplotlib chart embeddings", "Executive-style report cover", "Multi-format streaming export"],
  },
];

export const HowItWorks = () => {
  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-20">
        {/* Editorial Title Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/30 to-background py-16 md:py-24">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-4xl">
            <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary border-primary/30 px-3 py-1">
              Audit Methodology & Architecture
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.05]">
              Transparent Data Pipeline.
            </h1>
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              From raw spreadsheet fields to board-level audit reports. We expose every decision, every threshold, and every mathematical operation.
            </p>
          </div>
        </div>

        {/* Vertical Pipeline Steps */}
        <div className="container mx-auto px-4 py-16 md:py-24 max-w-5xl">
          <div className="space-y-12 relative before:absolute before:inset-0 before:left-8 sm:before:left-1/2 before:-ml-px before:w-0.5 before:bg-border/60 hidden sm:block" />

          <div className="space-y-12">
            {steps.map((step) => {
              const Icon = step.icon;
              return (
                <Card key={step.id} className="border border-border/80 bg-card p-6 md:p-8 rounded-2xl shadow-premium hover:border-primary/30 transition-all duration-300 relative group">
                  <div className="flex flex-col md:flex-row gap-6 md:items-start">
                    
                    {/* Number & Icon Badge */}
                    <div className="flex items-center gap-4 shrink-0">
                      <span className="font-mono font-bold text-2xl text-primary/40 group-hover:text-primary transition-colors">
                        {step.id}
                      </span>
                      <div className="h-12 w-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center group-hover:scale-105 transition-transform">
                        <Icon className="h-6 w-6" />
                      </div>
                    </div>

                    {/* Step Info */}
                    <div className="space-y-3 flex-1">
                      <h3 className="text-xl font-display font-bold text-foreground uppercase tracking-tight">
                        {step.title}
                      </h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        {step.description}
                      </p>

                      <div className="flex flex-wrap gap-2 pt-2">
                        {step.details.map((detail, dIdx) => (
                          <span key={dIdx} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-muted/60 text-muted-foreground text-[11px] font-mono border border-border/40">
                            <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                            {detail}
                          </span>
                        ))}
                      </div>
                    </div>

                  </div>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Architecture Comparison Table */}
        <div className="border-t border-border/60 bg-muted/20 py-16 md:py-24">
          <div className="container mx-auto px-4 max-w-5xl">
            <div className="text-center mb-12 space-y-2">
              <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">Why GetReport vs Manual Python / Excel</h2>
              <p className="text-xs sm:text-sm text-muted-foreground">Automated transparency instead of brittle custom scripts.</p>
            </div>

            <div className="border border-border bg-card rounded-2xl shadow-premium overflow-hidden">
              <div className="grid grid-cols-3 bg-muted/50 p-4 border-b border-border text-xs font-display font-bold uppercase tracking-wider text-foreground">
                <div>Capability</div>
                <div>Manual Python / Pandas</div>
                <div className="text-primary font-extrabold">GetReport Subsystem</div>
              </div>

              <div className="divide-y divide-border/60 text-xs sm:text-sm font-sans">
                <div className="grid grid-cols-3 p-4 items-center">
                  <div className="font-semibold text-foreground">Audit Speed</div>
                  <div className="text-muted-foreground">Minutes to hours</div>
                  <div className="text-emerald-700 font-bold font-mono">Sub-second (Polars Rust)</div>
                </div>

                <div className="grid grid-cols-3 p-4 items-center">
                  <div className="font-semibold text-foreground">Quality Grading</div>
                  <div className="text-muted-foreground">Manual inspection</div>
                  <div className="text-emerald-700 font-bold font-mono">A-F Column Confidence</div>
                </div>

                <div className="grid grid-cols-3 p-4 items-center">
                  <div className="font-semibold text-foreground">Data Privacy</div>
                  <div className="text-muted-foreground">Varies / raw logs</div>
                  <div className="text-emerald-700 font-bold font-mono">In-Memory Ephemeral</div>
                </div>

                <div className="grid grid-cols-3 p-4 items-center">
                  <div className="font-semibold text-foreground">Executive Output</div>
                  <div className="text-muted-foreground">Raw charts / Jupyter notebook</div>
                  <div className="text-emerald-700 font-bold font-mono">Board-Ready PDF & RAG</div>
                </div>
              </div>
            </div>

            {/* CTA */}
            <div className="text-center pt-14">
              <Link to="/workspace">
                <Button size="lg" className="h-12 px-8 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all font-display font-semibold">
                  <span>Start Audit Pipeline</span>
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
