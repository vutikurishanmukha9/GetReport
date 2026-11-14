import { useState } from "react";
import { 
  Check, ArrowRight, 
  ChevronDown, ChevronUp,
  FileSpreadsheet, SlidersHorizontal, BarChart3, FileText,
  ShieldCheck
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
      { name: "Executive PDF Reports", description: "High-DPI print-ready PDF reports with embedded analytical charts", included: true },
      { name: "Multi-Format Cleaned Exports", description: "Download remediated data as CSV, Parquet, or standalone HTML", included: true },
      { name: "Visual Insights Gallery", description: "Correlation heatmaps, distribution plots, and outlier boxplots", included: true },
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
    answer: "Yes. GetReport is 100% free and open source. All capabilities (including multi-dataset relational joins, Issue Ledger approvals, RAG companion chat, and executive PDF reports) are fully unlocked for everyone without any credit card or tier gating."
  },
  {
    question: "How does GetReport handle data privacy and confidentiality?",
    answer: "All processing occurs strictly inside ephemeral server memory using Polars dataframes. Raw records are never written to permanent disk storage, are never used to train AI models, and are automatically purged from RAM within 60 minutes or immediately upon clicking 'Reset Workspace'."
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
    <div className="min-h-screen flex flex-col bg-[#F5F2EB] text-stone-900 dark:bg-[#07080a] dark:text-zinc-100 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.05)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        {/* Header */}
        <div className="border-b border-stone-300/80 bg-[#FAF8F5]/90 dark:border-white/[0.08] dark:bg-[#07080a]/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-4xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-stone-300 bg-stone-100 text-stone-700 dark:border-white/10 dark:bg-white/[0.04] dark:text-zinc-300 text-xs font-sans font-medium">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>Open Source • Free for Personal & Enterprise Use</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-sans font-black text-stone-900 dark:text-white tracking-[-0.03em] leading-[1.12]">
              Full platform access. <span className="text-stone-500 dark:text-zinc-400 block mt-1">Zero paywalls.</span>
            </h1>
            
            <p className="text-sm sm:text-base text-stone-600 dark:text-zinc-400 max-w-2xl mx-auto leading-relaxed font-sans">
              No subscriptions, no tiered limitations, and no credit card required. Enterprise-grade data quality scoring, multi-dataset joins, and executive PDF generation for everyone.
            </p>
          </div>
        </div>

        {/* Section 1: Hero Offering Card */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-4xl">
          <Card className="border border-stone-300/80 bg-[#FAF8F5] dark:border-white/15 dark:bg-[#0c0d12]/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-5 sm:p-8 shadow-md dark:shadow-2xl relative overflow-hidden space-y-6">
            
            {/* Top Row: Tier Name & Price */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-stone-300/80 dark:border-white/10 pb-6">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <Badge className="bg-[#800000] text-white dark:bg-white dark:text-black text-[10px] font-sans font-bold uppercase tracking-[0.15em] px-3 py-0.5 rounded-md">
                    COMMUNITY & ENTERPRISE
                  </Badge>
                  <span className="text-xs font-sans text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full font-semibold">
                    Active Release
                  </span>
                </div>
                <h2 className="text-xl sm:text-2xl font-sans font-bold text-stone-900 dark:text-white tracking-tight">
                  GetReport Open Edition
                </h2>
                <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 font-sans">
                  Instant in-memory dataset audits with zero account creation or setup friction.
                </p>
              </div>

              <div className="flex flex-col items-start sm:items-end shrink-0">
                <div className="flex items-baseline gap-1.5">
                  <span className="text-4xl sm:text-5xl font-sans font-black text-stone-900 dark:text-white tracking-tight tabular-nums">$0</span>
                  <span className="text-xs text-emerald-700 dark:text-emerald-400 font-sans font-bold uppercase tracking-[0.15em]">FREE FOREVER</span>
                </div>
                <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-sans uppercase tracking-wider mt-0.5">Apache 2.0 / Open Source</span>
              </div>
            </div>

            {/* Core Feature Bullet Grid */}
            <div className="space-y-3">
              <span className="text-xs font-sans uppercase tracking-[0.15em] text-stone-800 dark:text-zinc-300 font-bold block">
                Everything Unlocked Right Now:
              </span>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 text-xs sm:text-sm font-sans">
                {[
                  "Polars zero-copy streaming ingestion",
                  "A-F Column Confidence Scoring (4 dimensions)",
                  "Interactive Issue Ledger with 1-click approvals",
                  "Multi-dataset relational joins (up to 5 datasets)",
                  "Executive PDF report downloads",
                  "VIF multicollinearity & conceptual drift checks",
                  "Ephemeral RAG AI Companion for dataset Q&A",
                  "100% in-memory RAM execution (zero disk writes)"
                ].map((feat, fIdx) => (
                  <div key={fIdx} className="flex items-start gap-2.5 p-3 rounded-xl bg-stone-100/90 border border-stone-200 text-stone-800 dark:bg-white/[0.03] dark:border-white/5 dark:text-zinc-200">
                    <Check className="h-4 w-4 text-emerald-700 dark:text-emerald-400 shrink-0 mt-0.5" />
                    <span className="leading-snug">{feat}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Launch CTA Bar */}
            <div className="pt-2 flex items-center">
              <Link to="/workspace" className="w-full">
                <Button size="lg" className="w-full h-11 rounded-xl bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:text-black dark:hover:bg-zinc-200 font-sans font-semibold text-sm shadow-md transition-all active:scale-[0.99] cursor-pointer">
                  <span>Launch Free Audit Workspace</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>

          </Card>
        </div>

        {/* Section 2: 4-Category Capability Bento Grid */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-6xl space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-2">
            <h2 className="text-2xl sm:text-3xl font-sans font-bold text-stone-900 dark:text-white tracking-tight">
              Platform capabilities & architecture
            </h2>
            <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
              Every single algorithm, metric, and data engine component included in GetReport Open Edition.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-6">
            {matrixCategories.map((cat, cIdx) => {
              const icons = [FileSpreadsheet, SlidersHorizontal, BarChart3, FileText];
              const CategoryIcon = icons[cIdx % icons.length];
              return (
                <Card key={cat.category} className="border border-stone-200/90 bg-[#FAF8F5] dark:border-white/10 dark:bg-[#0c0d12]/80 backdrop-blur-md rounded-2xl p-5 sm:p-6 shadow-sm flex flex-col justify-between">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between border-b border-stone-200 dark:border-white/10 pb-3">
                      <div className="flex items-center gap-2.5">
                        <div className="h-8 w-8 rounded-lg bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/[0.06] dark:text-white dark:border-white/10 flex items-center justify-center font-bold">
                          <CategoryIcon className="h-4 w-4" />
                        </div>
                        <h3 className="font-sans font-bold text-sm sm:text-base text-stone-900 dark:text-white tracking-tight">
                          {cat.category}
                        </h3>
                      </div>
                      <Badge variant="outline" className="text-[10px] font-sans font-semibold uppercase tracking-wider bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/20">
                        Unlocked
                      </Badge>
                    </div>

                    <div className="space-y-3">
                      {cat.items.map((item, iIdx) => (
                        <div key={iIdx} className="flex items-start gap-2.5">
                          <div className="h-4 w-4 rounded-full bg-emerald-100 text-emerald-800 dark:bg-emerald-500/10 dark:text-emerald-400 flex items-center justify-center shrink-0 mt-0.5 border border-emerald-300 dark:border-emerald-500/20">
                            <Check className="h-3 w-3" />
                          </div>
                          <div className="space-y-0.5">
                            <span className="font-sans font-semibold text-xs sm:text-sm text-stone-900 dark:text-zinc-200 block">
                              {item.name}
                            </span>
                            <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-sans leading-relaxed">
                              {item.description}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Section 3: Interactive FAQ Accordion */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-4xl space-y-6">
          <div className="text-center max-w-2xl mx-auto space-y-1.5">
            <h2 className="text-xl sm:text-2xl font-sans font-bold text-stone-900 dark:text-white tracking-tight">
              Everything you need to know
            </h2>
          </div>

          <div className="space-y-2.5">
            {faqs.map((faq, fIdx) => {
              const isOpen = openFaqIndex === fIdx;
              return (
                <Card 
                  key={fIdx}
                  className={`border transition-all duration-200 rounded-xl overflow-hidden ${
                    isOpen ? "border-stone-300 bg-[#EDE8DE] dark:border-white/20 dark:bg-[#12141c]/90 shadow-md" : "border-stone-200 bg-[#FAF8F5] hover:bg-[#F3EFE6] dark:border-white/10 dark:bg-[#0c0d12]/70 dark:hover:bg-[#0c0d12]"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => toggleFaq(fIdx)}
                    className="w-full px-5 py-4 flex items-center justify-between gap-4 text-left cursor-pointer bg-transparent border-0 select-none"
                  >
                    <span className="font-sans font-semibold text-xs sm:text-sm text-stone-900 dark:text-zinc-100">
                      {faq.question}
                    </span>
                    {isOpen ? (
                      <ChevronUp className="h-4 w-4 text-stone-900 dark:text-white shrink-0 transition-transform" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-stone-500 dark:text-zinc-400 shrink-0 transition-transform" />
                    )}
                  </button>

                  {isOpen && (
                    <div className="px-5 pb-4 text-xs sm:text-sm text-stone-600 dark:text-zinc-400 font-sans leading-relaxed border-t border-stone-200 dark:border-white/5 pt-3 animate-in slide-in-from-top-1 duration-200">
                      {faq.answer}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        </div>

        {/* Section 4: Bottom Conversion Bar */}
        <div className="border-t border-stone-300/80 bg-[#FAF8F5]/90 dark:border-white/10 dark:bg-[#0c0d12]/60 py-10 sm:py-14">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-sans font-bold text-stone-900 dark:text-white tracking-tight">
              Start auditing datasets in seconds
            </h2>
            <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 font-sans max-w-lg mx-auto">
              Drop your spreadsheet or Parquet file. No registration required.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl bg-[#800000] text-white hover:bg-[#6b0000] dark:bg-white dark:text-black dark:hover:bg-zinc-200 font-sans font-semibold text-sm shadow-md transition-all active:scale-[0.99] cursor-pointer">
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
