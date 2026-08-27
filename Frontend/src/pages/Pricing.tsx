import { useState } from "react";
import { 
  Check, ArrowRight, Heart, 
  Github, ChevronDown, ChevronUp, HelpCircle
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

interface FeatureCategory {
  category: string;
  items: { name: string; description: string; included: boolean }[];
}

const matrixCategories: FeatureCategory[] = [
  {
    category: "Ingestion & Schema Resolution",
    items: [
      { name: "Multi-Format Parsing", description: "CSV, XLSX, XLS, Parquet, TSV, JSONL, Feather, Arrow", included: true },
      { name: "Zero-Copy Polars Engine", description: "Memory-mapped ingestion with automatic delimiter and encoding detection", included: true },
      { name: "Multi-Dataset Relational Joins", description: "Merge up to 5 datasets on composite primary keys with DAG lineage tracking", included: true },
      { name: "Zip-Bomb & Memory Exhaustion Guards", description: "Enforces strict decompression bounds and payload memory safety limits", included: true },
    ]
  },
  {
    category: "Hygiene & Issue Remediation",
    items: [
      { name: "A-F Column Confidence Scores", description: "Evaluates completeness (35%), consistency (25%), validity (25%), stability (15%)", included: true },
      { name: "Interactive Issue Ledger", description: "9 quality check categories with 1-click human-in-the-loop approvals", included: true },
      { name: "Automated Data Cleaning", description: "Winsorization capping, type coercion, missing value fill, and duplicate purging", included: true },
      { name: "ML Readiness Scorecard", description: "Flags extreme class imbalance, constant variables, and target leakage risks", included: true },
    ]
  },
  {
    category: "Statistical Depth & AI Synthesis",
    items: [
      { name: "VIF Multicollinearity Detection", description: "Calculates Variance Inflation Factors to eliminate collinear predictors (VIF > 5)", included: true },
      { name: "Time-Series Conceptual Drift", description: "Detects chronological mean and distribution shifts > 30%", included: true },
      { name: "Ephemeral RAG AI Companion", description: "Contextual dataset Q&A with grounded statistical citations", included: true },
      { name: "Deterministic Offline Rule Fallback", description: "Generates analytical insights even when external AI API keys are unset", included: true },
    ]
  },
  {
    category: "Executive Reporting & Export",
    items: [
      { name: "WeasyPrint Executive PDF", description: "High-DPI print-ready PDF reports with embedded Matplotlib galleries", included: true },
      { name: "Multi-Format Cleaned Exports", description: "Download remediated data as CSV, Parquet, or standalone HTML", included: true },
      { name: "Matplotlib Visual Insights Gallery", description: "Correlation heatmaps, distribution plots, and outlier boxplots", included: true },
      { name: "Cryptographic Audit Receipts", description: "Verifiable transformation logs detailing every executed data operation", included: true },
    ]
  },
  {
    category: "Security & Privacy",
    items: [
      { name: "100% In-Memory Execution", description: "Zero raw records ever written to persistent disk storage", included: true },
      { name: "Automated 60-Minute Session Purge", description: "All memory buffers and intermediate files wiped automatically", included: true },
      { name: "Zero Model Training", description: "Uploaded datasets are never used to train public or proprietary LLMs", included: true },
      { name: "TLS 1.3 & HSTS Enforcement", description: "Modern encryption standards with strict Content Security Policies", included: true },
    ]
  }
];

const faqs = [
  {
    question: "Is GetReport truly 100% free with no hidden paywalls?",
    answer: "Yes. GetReport is 100% free and open source. All capabilities—including multi-dataset relational joins, Issue Ledger approvals, RAG companion chat, and board-ready WeasyPrint PDF reports—are fully unlocked for everyone without any credit card or tier gating."
  },
  {
    question: "How does GetReport handle data privacy and confidentiality?",
    answer: "All processing occurs strictly inside ephemeral server memory using Polars Rust dataframes. Raw records are never written to permanent disk storage, are never used to train AI models, and are automatically purged from RAM within 60 minutes or immediately upon clicking 'Reset Workspace'."
  },
  {
    question: "What file formats can I audit?",
    answer: "GetReport supports CSV (.csv), Excel workbooks (.xls, .xlsx via calamine), Apache Parquet (.parquet), TSV (.tsv), JSON Lines (.jsonl), Feather, and Arrow buffers."
  },
  {
    question: "How does the AI Companion work if my dataset contains confidential records?",
    answer: "Our RAG engine transmits only high-level statistical summaries (such as column mean, standard deviation, anomaly counts, and domain taxonomy) to LLM endpoints. Raw row entries and personally identifiable records (PII) are completely omitted from prompt payloads."
  },
  {
    question: "Can I deploy GetReport on-premise in my own Docker cluster?",
    answer: "Yes. Because GetReport is open source, you can clone the repository and run the full FastAPI backend and Vite React frontend locally or inside private Kubernetes/Docker infrastructure with zero external telemetry."
  }
];

export const Pricing = () => {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  const toggleFaq = (index: number) => {
    setOpenFaqIndex(prev => prev === index ? null : index);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background animate-in fade-in duration-500">
      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-16 sm:pt-20">
        {/* Header */}
        <div className="border-b border-border/60 bg-gradient-to-b from-muted/20 via-background to-background py-8 sm:py-12">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-emerald-500/10 text-emerald-700 text-xs font-semibold uppercase tracking-wider font-mono border border-emerald-500/20 t-badge-shimmer">
              <Heart className="h-3.5 w-3.5 fill-emerald-500" />
              <span>100% Free & Open Source</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase leading-[1.08]">
              Full Platform Access. <span className="text-primary block mt-0.5">Zero Paywalls.</span>
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              No subscriptions, no tiered limitations, and no credit card required. Enterprise-grade data quality scoring, multi-dataset joins, and executive PDF generation for everyone.
            </p>
          </div>
        </div>

        {/* Section 1: Hero Offering Card */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-4xl">
          <Card className="border-2 border-primary/30 bg-card rounded-2xl sm:rounded-3xl p-5 sm:p-8 shadow-premium relative overflow-hidden space-y-6 t-card-lift">
            
            {/* Top Row: Tier Name & Price */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border/60 pb-6">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <Badge className="bg-primary text-primary-foreground text-[10px] font-mono font-bold uppercase tracking-wider px-3 py-0.5 rounded-md">
                    COMMUNITY & ENTERPRISE
                  </Badge>
                  <span className="text-xs font-mono text-emerald-600 font-semibold flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 t-pulse-dot" /> Active Release
                  </span>
                </div>
                <h2 className="text-xl sm:text-2xl font-display font-extrabold text-foreground uppercase tracking-tight">
                  GetReport Open Edition
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground font-sans">
                  Instant in-memory dataset audits with zero account creation or setup friction.
                </p>
              </div>

              <div className="flex flex-col items-start sm:items-end shrink-0">
                <div className="flex items-baseline gap-1.5">
                  <span className="text-4xl sm:text-5xl font-display font-extrabold text-foreground">$0</span>
                  <span className="text-xs text-emerald-700 font-mono font-bold uppercase tracking-wider">FREE FOREVER</span>
                </div>
                <span className="text-[10px] text-muted-foreground font-mono mt-0.5">Apache 2.0 / Open Source</span>
              </div>
            </div>

            {/* Core Feature Bullet Grid */}
            <div className="space-y-3">
              <span className="text-xs font-mono uppercase tracking-wider text-primary font-bold block">
                Everything Unlocked Right Now:
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs sm:text-sm text-foreground font-sans">
                {[
                  "Polars Rust zero-copy streaming ingestion",
                  "A-F Column Confidence Scoring (4 dimensions)",
                  "Interactive Issue Ledger with 1-click approvals",
                  "Multi-dataset relational joins (up to 5 datasets)",
                  "WeasyPrint executive PDF report downloads",
                  "VIF multicollinearity & conceptual drift checks",
                  "Ephemeral RAG AI Companion for dataset Q&A",
                  "100% in-memory RAM execution (zero disk writes)"
                ].map((feat, fIdx) => (
                  <div key={fIdx} className="flex items-start gap-2.5 p-2.5 rounded-xl bg-muted/30 border border-border/40 text-xs sm:text-sm">
                    <Check className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                    <span className="leading-snug">{feat}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Launch CTA Bar */}
            <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
              <Link to="/workspace" className="flex-1">
                <Button size="lg" className="w-full h-11 rounded-xl shadow-premium t-card-lift t-spring-press font-display font-semibold text-sm">
                  <span>Launch Free Audit Workspace</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <a 
                href="https://github.com/vutikurishanmukha9/GetReport" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="sm:w-auto"
              >
                <Button size="lg" variant="outline" className="w-full sm:w-auto h-11 px-6 rounded-xl border-border bg-card hover:bg-muted/20 shadow-premium t-card-lift t-spring-press font-display text-sm gap-2">
                  <Github className="h-4 w-4" />
                  <span>Star on GitHub</span>
                </Button>
              </a>
            </div>

          </Card>
        </div>

        {/* Section 2: Full Capability & Feature Matrix */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-5xl space-y-6">
          <div className="text-center max-w-2xl mx-auto space-y-1.5">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Complete Feature & Architecture Matrix
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans">
              Every single algorithm, metric, and data engine component included in the free release.
            </p>
          </div>

          <div className="border border-border/80 bg-card rounded-2xl shadow-premium overflow-hidden">
            {matrixCategories.map((cat, cIdx) => (
              <div key={cat.category} className={cIdx !== 0 ? "border-t border-border/60" : ""}>
                <div className="bg-muted/40 px-5 py-2.5 border-b border-border/40">
                  <h3 className="font-display font-bold text-xs uppercase tracking-wider text-primary">
                    {cat.category}
                  </h3>
                </div>
                <div className="divide-y divide-border/40">
                  {cat.items.map((item, iIdx) => (
                    <div key={iIdx} className="px-5 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2 hover:bg-muted/10 transition-colors">
                      <div className="space-y-0.5 max-w-2xl">
                        <span className="font-sans font-semibold text-xs sm:text-sm text-foreground block">
                          {item.name}
                        </span>
                        <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
                          {item.description}
                        </p>
                      </div>
                      <div className="flex items-center gap-1.5 text-emerald-700 font-mono text-xs font-semibold shrink-0">
                        <Check className="h-3.5 w-3.5 text-emerald-600" />
                        <span>Included</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Section 3: Interactive FAQ Accordion */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 max-w-4xl space-y-6">
          <div className="text-center max-w-2xl mx-auto space-y-1.5">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-mono font-bold uppercase tracking-wider">
              <HelpCircle className="h-3.5 w-3.5" />
              <span>Frequently Asked Questions</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Everything You Need to Know
            </h2>
          </div>

          <div className="space-y-2.5">
            {faqs.map((faq, fIdx) => {
              const isOpen = openFaqIndex === fIdx;
              return (
                <Card 
                  key={fIdx}
                  className={`border transition-all duration-200 rounded-xl overflow-hidden ${
                    isOpen ? "border-primary/40 bg-card shadow-xs" : "border-border bg-card/60 hover:bg-card"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => toggleFaq(fIdx)}
                    className="w-full px-5 py-3.5 flex items-center justify-between gap-4 text-left cursor-pointer bg-transparent border-0 select-none"
                  >
                    <span className="font-display font-bold text-xs sm:text-sm text-foreground">
                      {faq.question}
                    </span>
                    {isOpen ? (
                      <ChevronUp className="h-4 w-4 text-primary shrink-0 transition-transform" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0 transition-transform" />
                    )}
                  </button>

                  {isOpen && (
                    <div className="px-5 pb-4 text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed border-t border-border/40 pt-2.5 animate-in slide-in-from-top-1 duration-200">
                      {faq.answer}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        </div>

        {/* Section 4: Bottom Conversion Bar */}
        <div className="border-t border-border/60 bg-muted/20 py-10 sm:py-12">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-display font-bold text-foreground uppercase tracking-tight">
              Start auditing datasets in seconds
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              Drop your spreadsheet or Parquet file. No registration required.
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

export default Pricing;
