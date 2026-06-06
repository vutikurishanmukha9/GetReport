import { BarChart3, FileSpreadsheet, ArrowRight, ShieldCheck, MessageSquareCode, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const HeroSection = () => {
  return (
    <section className="relative pt-16 pb-16 sm:pt-20 sm:pb-20 lg:pt-28 lg:pb-24 overflow-hidden text-center">
      {/* Subtle tint background */}
      <div className="absolute inset-0 bg-gradient-to-b from-muted/30 to-transparent -z-10" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="flex flex-col items-center space-y-6 sm:space-y-8">
            
            {/* Tagline */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border/80 bg-muted/30 text-muted-foreground text-xs font-mono uppercase tracking-wider shadow-sm">
              <FileSpreadsheet className="h-3.5 w-3.5 text-primary" />
              <span>Auditable Report Generator</span>
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-display font-semibold tracking-[-0.03em] text-foreground leading-[1.05] uppercase">
              Turn raw data into
              <span className="block text-primary mt-2 font-display tracking-[-0.03em]">publication-ready reports</span>
            </h1>

            {/* Subheadline */}
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-2xl leading-relaxed font-sans mx-auto">
              Upload your CSV or Excel files. Review quality issues inside the interactive ledger,
              and receive a comprehensive PDF report with statistical charts and RAG insights.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-wrap justify-center gap-4 pt-4 pb-8">
              <Link to="/workspace">
                <Button size="lg" className="h-14 px-8 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all font-semibold font-display text-base">
                  Start Free Audit
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </Link>
              <Link to="/how-it-works">
                <Button size="lg" variant="outline" className="h-14 px-8 rounded-xl border-border bg-card hover:bg-muted/10 shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all font-display text-base">
                  How it Works
                </Button>
              </Link>
            </div>

            {/* Valuable Information - No Cards, just clean typography */}
            <div className="pt-12 grid grid-cols-1 sm:grid-cols-3 gap-8 text-left w-full max-w-4xl mx-auto border-t border-border/60">
              
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-foreground font-display font-semibold text-lg">
                  <ShieldCheck className="h-5 w-5 text-emerald-600" /> 
                  Zero-Trust Security
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Data is processed securely with strict magic-number signature checks. Operations happen seamlessly without permanently storing your sensitive datasets.
                </p>
                <ul className="text-xs font-mono text-muted-foreground/80 space-y-1.5 pt-1">
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500"/> Stateless API Processing</li>
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-emerald-500"/> Payload Sanitization</li>
                </ul>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 text-foreground font-display font-semibold text-lg">
                  <BarChart3 className="h-5 w-5 text-primary" /> 
                  Automated Statistics
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Eliminate manual EDA. Instantly identify outliers, calculate correlation matrices, verify collinearity, and compute descriptive statistics for every column.
                </p>
                <ul className="text-xs font-mono text-muted-foreground/80 space-y-1.5 pt-1">
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-primary"/> Outlier Capping Engine</li>
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-primary"/> High-Cardinality Detection</li>
                </ul>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 text-foreground font-display font-semibold text-lg">
                  <MessageSquareCode className="h-5 w-5 text-indigo-500" /> 
                  RAG Synthesized PDF
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Beyond raw numbers, our engine generates publication-ready PDFs combining your statistical outputs with context-aware, RAG-powered narrative insights.
                </p>
                <ul className="text-xs font-mono text-muted-foreground/80 space-y-1.5 pt-1">
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-indigo-400"/> Dual-Engine Compilation</li>
                  <li className="flex items-center gap-1.5"><CheckCircle2 className="h-3 w-3 text-indigo-400"/> Narrative Insight Generation</li>
                </ul>
              </div>

            </div>

        </div>
      </div>
    </section>
  );
};
