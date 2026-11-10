import { useState } from "react";
import { 
  Check, ArrowRight, ShieldCheck,
  FileSpreadsheet, FileText, SlidersHorizontal, BarChart3,
  ChevronDown, ChevronUp
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";

export const Pricing = () => {
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);

  const toggleFaq = (idx: number) => {
    setOpenFaqIndex((prev) => (prev === idx ? null : idx));
  };

  const matrixCategories = [
    {
      category: "Ingestion & Streaming Engine",
      items: [
        { name: "Single & Multi-File Drag-and-Drop", description: "CSV, XLSX, XLS, Parquet, JSON, and TSV formats" },
        { name: "Zero-Copy Ephemeral Processing", description: "In-memory Polars execution with zero disk writes" },
        { name: "Direct Snowflake Data Warehouse Sync", description: "Stream warehouse queries without intermediate caching" },
        { name: "Multi-Dataset Relational Joins", description: "Merge up to 5 datasets on common keys with schema checks" },
      ],
    },
    {
      category: "Data Quality & Assurance Pipeline",
      items: [
        { name: "Interactive Review Ledger", description: "Full transparency over proposed column fixes before commit" },
        { name: "Column-Level Trust Scoring (0-100%)", description: "Completeness, consistency, validity, and stability metrics" },
        { name: "Statistical Multicollinearity & VIF Scan", description: "Automated Variance Inflation Factor checks across features" },
        { name: "Silent Formula Corruption Prevention", description: "Protect spreadsheet formulas against destructive cleanups" },
      ],
    },
    {
      category: "Interactive Exploration & Visual Studio",
      items: [
        { name: "Multivariate HiPlot Exploration", description: "Parallel coordinates, scatter projections, and correlations" },
        { name: "Instant Column Slice & Filtering", description: "Sub-second filtering across millions of rows" },
        { name: "Correlation Matrices & Heatmaps", description: "Identify hidden linear and nonlinear feature associations" },
        { name: "Executive Confidence Certificate", description: "Letter grade A through F statistical audit receipt" },
      ],
    },
    {
      category: "Export Formats & Enterprise Outputs",
      items: [
        { name: "Boardroom-Ready PDF Reports", description: "High-DPI vector documents compiled via WeasyPrint" },
        { name: "Cleaned Excel Workbook (.xlsx)", description: "Sanitized data with intact original formulas" },
        { name: "Columnar Apache Parquet (.parquet)", description: "Production-ready files ready for downstream ETL engines" },
        { name: "Reproducible Cleaning DAG (.json)", description: "Export deterministic transformation rules for your team" },
      ],
    },
  ];

  const faqs = [
    {
      question: "Is GetReport completely free to use?",
      answer: "Yes. GetReport is an open-source, local-first data intelligence engine. There are no subscriptions, paywalls, or usage caps on dataset sizes or export counts."
    },
    {
      question: "Does GetReport store my uploaded data on any cloud servers?",
      answer: "No. GetReport operates entirely within your browser session and ephemeral local server memory. All computations use in-memory Polars buffers that are immediately flushed when your session ends."
    },
    {
      question: "Can I use GetReport for enterprise or client projects?",
      answer: "Yes. All exports—including WeasyPrint executive PDF briefs, clean Excel sheets, and Parquet files—are yours to keep, present to stakeholders, or integrate into your pipelines."
    },
    {
      question: "How does GetReport handle large files over 500MB?",
      answer: "Because GetReport uses an Arrow-native Polars core in Rust, it processes hundreds of thousands of rows in milliseconds with minimal RAM overhead, preventing the crashes common to standard spreadsheet software."
    }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200 animate-in fade-in duration-500 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.04)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-50 dark:opacity-40" />
      <div className="hero-grain-overlay" />

      <Header onReset={() => {}} showReset={false} />

      <main className="flex-1 pt-24 sm:pt-28 relative z-10">
        
        {/* Header Section */}
        <div className="border-b border-border bg-background/90 py-8 sm:py-14 relative z-10">
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl text-center space-y-3 sm:space-y-4">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border bg-muted/50 text-muted-foreground text-xs font-mono font-medium">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>100% Free & Open Architecture</span>
            </div>

            <h1 className="text-3xl sm:text-4xl md:text-5xl font-display font-black text-foreground tracking-[-0.03em] leading-[1.12]">
              Transparent access for all teams.
            </h1>
            
            <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed font-sans">
              No subscriptions, no tiered limitations, and no credit card required. Enterprise-grade data quality scoring, multi-dataset joins, and executive PDF generation for everyone.
            </p>
          </div>
        </div>

        {/* Section 1: Hero Offering Card */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-4xl">
          <Card className="border border-border dark:border-white/15 bg-card dark:bg-[#0c0d12]/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl p-5 sm:p-8 shadow-sm dark:shadow-2xl relative overflow-hidden space-y-6 text-card-foreground">
            
            {/* Top Row: Tier Name & Price */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border pb-6">
              <div className="space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <Badge className="bg-zinc-950 text-white dark:bg-white dark:text-black text-[10px] font-sans font-bold uppercase tracking-[0.15em] px-3 py-0.5 rounded-md">
                    COMMUNITY & ENTERPRISE
                  </Badge>
                  <span className="text-xs font-sans text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full font-semibold">
                    Active Release
                  </span>
                </div>
                <h2 className="text-xl sm:text-2xl font-sans font-bold text-foreground tracking-tight">
                  GetReport Open Edition
                </h2>
                <p className="text-xs sm:text-sm text-muted-foreground font-sans">
                  Instant in-memory dataset audits with zero account creation or setup friction.
                </p>
              </div>

              <div className="text-left sm:text-right shrink-0">
                <div className="flex items-baseline gap-1">
                  <span className="text-3xl sm:text-4xl font-display font-black text-foreground tracking-tight">$0</span>
                  <span className="text-xs text-muted-foreground font-sans">/ forever</span>
                </div>
                <span className="text-[11px] font-mono text-muted-foreground block mt-0.5">
                  Apache 2.0 Open Source Core
                </span>
              </div>
            </div>

            {/* Core Capability Checklist */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs sm:text-sm font-sans">
              {[
                "Unlimited dataset uploads and row volume",
                "Instant Polars streaming engine (<50ms execution)",
                "Full human-in-the-loop Issue Ledger and DAG review",
                "Direct Snowflake Data Warehouse connector",
                "Boardroom-ready vector PDF report exports",
                "100% ephemeral in-memory processing (zero storage)",
                "Cleaned Excel, CSV, and Apache Parquet downloads",
                "Multivariate HiPlot exploration and correlation heatmaps"
              ].map((feature, idx) => (
                <div key={idx} className="flex items-center gap-2.5">
                  <div className="h-4 w-4 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 border border-emerald-500/20">
                    <Check className="h-2.5 w-2.5 stroke-[3]" />
                  </div>
                  <span className="text-foreground/90 font-medium text-xs">{feature}</span>
                </div>
              ))}
            </div>

            {/* Primary Action Button */}
            <div className="pt-2 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-4">
              <span className="text-xs text-muted-foreground font-sans">
                Works directly in your modern browser • No signup required
              </span>
              <Link to="/workspace" className="w-full sm:w-auto">
                <Button size="lg" className="w-full sm:w-auto h-11 px-8 rounded-xl font-sans font-semibold text-sm shadow-sm cursor-pointer">
                  <span>Start Free Audit</span>
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
            </div>
          </Card>
        </div>

        {/* Section 2: Comprehensive Feature Matrix */}
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 max-w-6xl space-y-8">
          <div className="text-center max-w-2xl mx-auto space-y-2">
            <h2 className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-tight">
              Complete capability breakdown
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans">
              All features are accessible directly in the browser with zero artificial paywalls.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 lg:gap-6">
            {matrixCategories.map((cat, cIdx) => {
              const icons = [FileSpreadsheet, SlidersHorizontal, BarChart3, FileText];
              const CategoryIcon = icons[cIdx % icons.length];
              return (
                <Card key={cat.category} className="border border-border dark:border-white/10 bg-card dark:bg-[#0c0d12]/80 backdrop-blur-md rounded-2xl p-5 sm:p-6 shadow-sm flex flex-col justify-between text-card-foreground">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between border-b border-border pb-3">
                      <div className="flex items-center gap-2.5">
                        <div className="h-8 w-8 rounded-lg bg-muted dark:bg-white/[0.06] text-foreground dark:text-white flex items-center justify-center border border-border dark:border-white/10">
                          <CategoryIcon className="h-4 w-4" />
                        </div>
                        <h3 className="font-sans font-bold text-sm sm:text-base text-foreground tracking-tight">
                          {cat.category}
                        </h3>
                      </div>
                      <Badge variant="outline" className="text-[10px] font-sans font-semibold uppercase tracking-wider bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-500/20">
                        Unlocked
                      </Badge>
                    </div>

                    <div className="space-y-3">
                      {cat.items.map((item, iIdx) => (
                        <div key={iIdx} className="flex items-start gap-2.5">
                          <div className="h-4 w-4 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 mt-0.5 border border-emerald-500/20">
                            <Check className="h-3 w-3 stroke-[2.5]" />
                          </div>
                          <div className="space-y-0.5">
                            <span className="font-sans font-semibold text-xs sm:text-sm text-foreground block">
                              {item.name}
                            </span>
                            <p className="text-[11px] text-muted-foreground font-sans leading-relaxed">
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
            <h2 className="text-xl sm:text-2xl font-sans font-bold text-foreground tracking-tight">
              Everything you need to know
            </h2>
          </div>

          <div className="space-y-2.5">
            {faqs.map((faq, fIdx) => {
              const isOpen = openFaqIndex === fIdx;
              return (
                <Card 
                  key={fIdx}
                  className={`border transition-all duration-200 rounded-xl overflow-hidden shadow-2xs ${
                    isOpen ? "border-border/80 bg-muted/40 dark:bg-[#12141c]/90 dark:border-white/20 shadow-sm" : "border-border bg-card dark:bg-[#0c0d12]/70 hover:bg-muted/30"
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => toggleFaq(fIdx)}
                    className="w-full px-5 py-4 flex items-center justify-between gap-4 text-left cursor-pointer bg-transparent border-0 select-none"
                  >
                    <span className="font-sans font-semibold text-xs sm:text-sm text-foreground">
                      {faq.question}
                    </span>
                    {isOpen ? (
                      <ChevronUp className="h-4 w-4 text-foreground shrink-0 transition-transform" />
                    ) : (
                      <ChevronDown className="h-4 w-4 text-muted-foreground shrink-0 transition-transform" />
                    )}
                  </button>

                  {isOpen && (
                    <div className="px-5 pb-4 text-xs sm:text-sm text-muted-foreground font-sans leading-relaxed border-t border-border mt-1 pt-3 animate-in slide-in-from-top-1 duration-200">
                      {faq.answer}
                    </div>
                  )}
                </Card>
              );
            })}
          </div>
        </div>

        {/* Section 4: Bottom Conversion Bar */}
        <div className="border-t border-border bg-muted/30 dark:bg-[#0c0d12]/60 py-10 sm:py-14">
          <div className="container mx-auto px-4 text-center space-y-4 max-w-3xl">
            <h2 className="text-xl sm:text-2xl font-sans font-bold text-foreground tracking-tight">
              Start auditing datasets in seconds
            </h2>
            <p className="text-xs sm:text-sm text-muted-foreground font-sans max-w-lg mx-auto">
              Drop your spreadsheet or Parquet file. No registration required.
            </p>
            <div className="pt-1 flex flex-wrap items-center justify-center gap-3">
              <Link to="/workspace">
                <Button size="lg" className="h-11 px-7 rounded-xl font-sans font-semibold text-sm shadow-sm transition-all active:scale-[0.99] cursor-pointer">
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
