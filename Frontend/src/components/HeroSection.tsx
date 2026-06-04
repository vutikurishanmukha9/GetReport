import { BarChart3, FileSpreadsheet } from "lucide-react";

export const HeroSection = () => {
  return (
    <section className="relative pt-24 pb-12 sm:pt-28 sm:pb-16 md:pt-32 md:pb-20 lg:pt-36 lg:pb-24 overflow-hidden">
      {/* Subtle tint background */}
      <div className="absolute inset-0 bg-gradient-to-b from-muted/30 to-transparent -z-10" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
          
          {/* Left Column: Left-aligned content */}
          <div className="lg:col-span-7 text-left space-y-6 sm:space-y-8">
            
            {/* Tagline */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border/80 bg-muted/30 text-muted-foreground text-xs font-mono uppercase tracking-wider">
              <FileSpreadsheet className="h-3.5 w-3.5 text-primary" />
              <span>Auditable Report Generator</span>
            </div>

            {/* Apple-style Display Headline */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-semibold tracking-[-0.03em] text-foreground leading-[1.05] max-w-xl uppercase">
              Turn raw data into
              <span className="block text-primary mt-1 font-display tracking-[-0.03em]">publication-ready reports</span>
            </h1>

            {/* Subheadline */}
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-lg leading-relaxed font-sans">
              Upload your CSV or Excel files. Review quality issues inside the interactive ledger,
              and receive a comprehensive PDF report with statistical charts and RAG insights.
            </p>

            {/* Stats row */}
            <div className="grid grid-cols-3 gap-4 max-w-md pt-2">
              <div className="flex flex-col p-4 rounded-xl border border-border bg-card shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/20">
                <span className="text-lg sm:text-xl font-display font-semibold tracking-tight text-foreground">&lt; 1 Min</span>
                <span className="text-xs text-muted-foreground font-medium mt-1 font-mono uppercase tracking-wider text-[9px]">generation</span>
              </div>
              <div className="flex flex-col p-4 rounded-xl border border-border bg-card shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/20">
                <span className="text-lg sm:text-xl font-display font-semibold tracking-tight text-foreground">PDF</span>
                <span className="text-xs text-muted-foreground font-medium mt-1 font-mono uppercase tracking-wider text-[9px]">ready report</span>
              </div>
              <div className="flex flex-col p-4 rounded-xl border border-border bg-card shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/20">
                <span className="text-lg sm:text-xl font-display font-semibold tracking-tight text-foreground">A-F</span>
                <span className="text-xs text-muted-foreground font-medium mt-1 font-mono uppercase tracking-wider text-[9px]">data grades</span>
              </div>
            </div>

          </div>

          {/* Right Column: Custom CSS Illustration of Pipeline */}
          <div className="lg:col-span-5 w-full max-w-sm mx-auto lg:max-w-none">
            <div className="relative border border-border bg-card rounded-2xl p-6 shadow-premium overflow-hidden min-h-[365px] flex flex-col justify-between select-none hover:-translate-y-0.5 transition-transform duration-300">
              
              {/* Header */}
              <div className="flex items-center justify-between border-b pb-4 mb-4 border-border/60">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
                  <span className="text-[10px] font-mono tracking-wider uppercase text-muted-foreground font-semibold">Active processing logs</span>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">dataset_v2.csv</span>
              </div>
              
              {/* Pipeline stages list */}
              <div className="space-y-3 flex-1 font-mono text-[11px] sm:text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <span className="text-emerald-500 font-bold">✓</span> 01_Polars_Ingest
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">402ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <span className="text-emerald-500 font-bold">✓</span> 02_Integrity_Score
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">128ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <span className="text-emerald-500 font-bold">✓</span> 03_VIF_Collinearity
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">215ms</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-foreground font-semibold flex items-center gap-2">
                    <span className="h-2 w-2 bg-primary animate-pulse rounded-full" /> 04_RAG_Synthesis
                  </span>
                  <span className="text-primary font-bold text-[9px] animate-pulse">Running…</span>
                </div>
                <div className="flex items-center justify-between text-muted-foreground/40">
                  <span className="flex items-center gap-2">
                    <span>○</span> 05_PDF_Compilation
                  </span>
                  <span className="text-[9px]">Queued</span>
                </div>
              </div>

              {/* A floating mini-chart & Grade A badge */}
              <div className="mt-6 pt-4 border-t border-border/60 flex items-center justify-between gap-4">
                <div className="flex-1 space-y-1.5">
                  <span className="text-[9px] font-mono uppercase text-muted-foreground flex items-center gap-1">
                    <BarChart3 className="h-3 w-3 text-primary" /> Data Health Score
                  </span>
                  <div className="flex items-end gap-1.5 h-10 pt-1">
                    <div className="w-full bg-muted rounded-t h-4" />
                    <div className="w-full bg-muted rounded-t h-6" />
                    <div className="w-full bg-muted rounded-t h-8" />
                    <div className="w-full bg-primary/45 rounded-t h-10" />
                    <div className="w-full bg-primary rounded-t h-9" />
                  </div>
                </div>
                <div className="flex flex-col items-center justify-center h-14 w-14 rounded-xl border border-border bg-background shadow-xs shrink-0 select-none">
                  <span className="text-[8px] font-mono text-muted-foreground leading-none font-bold uppercase">GRADE</span>
                  <span className="text-xl font-display font-bold text-primary mt-0.5">A</span>
                </div>
              </div>
              
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};
