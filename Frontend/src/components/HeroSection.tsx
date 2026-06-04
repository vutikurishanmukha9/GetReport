/* Hallmark · component: hero · genre: modern-minimal · theme: Quiet
 * macrostructure: Split diptych knobs: ratio=7/5, right-side=proof-visual, divider=negative-space
 * contrast: pass
 */

import { BarChart3, FileSpreadsheet, ShieldAlert } from "lucide-react";

export const HeroSection = () => {
  return (
    <section className="relative pt-24 pb-12 sm:pt-28 sm:pb-16 md:pt-32 md:pb-20 lg:pt-36 lg:pb-24 overflow-hidden">
      {/* Subtle tint background - no mesh gradients */}
      <div className="absolute inset-0 bg-gradient-to-b from-muted/20 to-transparent -z-10" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
          
          {/* Left Column: Left-aligned content */}
          <div className="lg:col-span-7 text-left space-y-6 sm:space-y-8">
            
            {/* Semantic Tagline (no sparkles) */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border bg-muted/40 text-muted-foreground text-xs font-mono uppercase tracking-wide">
              <FileSpreadsheet className="h-3.5 w-3.5" />
              <span>Auditable Report Generator</span>
            </div>

            {/* High-weight elegant headline (no gradient fill) */}
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-foreground leading-[1.1] max-w-xl">
              Turn raw data into
              <span className="block text-primary mt-2">publication-ready reports</span>
            </h1>

            {/* Subheadline with breathing room */}
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-lg leading-relaxed">
              Upload your CSV or Excel files. Review quality issues inside the interactive ledger,
              and receive a comprehensive PDF report with statistical charts and RAG insights.
            </p>

            {/* Stats row - solid layout and linear hover shift */}
            <div className="grid grid-cols-3 gap-4 max-w-md pt-2">
              <div className="flex flex-col p-3 rounded-lg border bg-card/50 transition-all duration-200 hover:-translate-y-1 hover:border-primary/30">
                <span className="text-lg sm:text-xl font-bold text-foreground">&lt; 1 Min</span>
                <span className="text-xs text-muted-foreground font-medium mt-1">Generation</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border bg-card/50 transition-all duration-200 hover:-translate-y-1 hover:border-primary/30">
                <span className="text-lg sm:text-xl font-bold text-foreground">PDF</span>
                <span className="text-xs text-muted-foreground font-medium mt-1">Ready Report</span>
              </div>
              <div className="flex flex-col p-3 rounded-lg border bg-card/50 transition-all duration-200 hover:-translate-y-1 hover:border-primary/30">
                <span className="text-lg sm:text-xl font-bold text-foreground">A-F</span>
                <span className="text-xs text-muted-foreground font-medium mt-1">Data Grades</span>
              </div>
            </div>

          </div>

          {/* Right Column: Custom CSS Illustration of Report Sheet */}
          <div className="lg:col-span-5 w-full max-w-sm mx-auto lg:max-w-none">
            <div className="relative border bg-card/60 rounded-2xl p-5 sm:p-6 shadow-sm overflow-hidden min-h-[320px] flex flex-col justify-between select-none">
              
              {/* Audited header */}
              <div className="flex items-center justify-between border-b pb-4 mb-4">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-[10px] font-mono tracking-wider uppercase text-muted-foreground">Inspection Active</span>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground max-w-[120px] truncate">dataset_v2.csv</span>
              </div>
              
              {/* Data checks visual list */}
              <div className="space-y-2.5 flex-1">
                <div className="flex items-center justify-between text-xs sm:text-sm border-b border-dashed pb-2">
                  <span className="text-muted-foreground">Magic Signatures</span>
                  <span className="font-mono text-emerald-600 text-xs">✓ Validated</span>
                </div>
                <div className="flex items-center justify-between text-xs sm:text-sm border-b border-dashed pb-2">
                  <span className="text-muted-foreground">Missing Values</span>
                  <span className="font-mono text-amber-600 text-xs">⚠ Cleaned (Mean)</span>
                </div>
                <div className="flex items-center justify-between text-xs sm:text-sm border-b border-dashed pb-2">
                  <span className="text-muted-foreground">Outliers Detected</span>
                  <span className="font-mono text-muted-foreground text-xs">0 found</span>
                </div>
                <div className="flex items-center justify-between text-xs sm:text-sm border-b border-dashed pb-2">
                  <span className="text-muted-foreground">Domain Detection</span>
                  <span className="font-mono text-emerald-600 text-xs">✓ Finance</span>
                </div>
              </div>

              {/* A floating mini-chart & Grade A badge */}
              <div className="mt-4 pt-4 border-t flex items-center justify-between gap-4">
                <div className="flex-1 space-y-1.5">
                  <span className="text-[9px] font-mono uppercase text-muted-foreground flex items-center gap-1">
                    <BarChart3 className="h-3 w-3" /> Column Stability
                  </span>
                  <div className="flex items-end gap-1 h-10 pt-1">
                    <div className="w-full bg-muted-foreground/15 rounded-t h-4" />
                    <div className="w-full bg-muted-foreground/15 rounded-t h-6" />
                    <div className="w-full bg-muted-foreground/15 rounded-t h-8" />
                    <div className="w-full bg-primary/40 rounded-t h-10" />
                    <div className="w-full bg-primary/75 rounded-t h-9" />
                  </div>
                </div>
                <div className="flex flex-col items-center justify-center h-14 w-14 rounded-xl border bg-background shadow-xs">
                  <span className="text-[9px] font-mono text-muted-foreground leading-none">GRADE</span>
                  <span className="text-xl font-bold text-primary mt-0.5">A</span>
                </div>
              </div>
              
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};
