import { CheckCircle2, ArrowRight, ShieldCheck, Database, Layers, FileSpreadsheet } from "lucide-react";
import { Link } from "react-router-dom";

export const ComparisonSection = () => {
  return (
    <section className="py-12 sm:py-16 bg-[#07080a] border-b border-white/[0.08] relative overflow-hidden text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-2.5 mb-8 sm:mb-12">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.04] border border-white/10 text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span>Engineering Architecture</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-extrabold text-white tracking-[-0.03em] leading-tight">
            Traditional Manual Verification vs. GetReport Automated Assurance
          </h2>
          <p className="text-xs sm:text-base text-zinc-400 max-w-2xl mx-auto leading-relaxed font-normal font-sans">
            Replace vulnerable spreadsheet lookups, silent formula corruption, and untracked cell edits with a cryptographically audited data hygiene pipeline.
          </p>
        </div>

        {/* 2-Column Comparison Architecture */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6 items-stretch">
          
          {/* Left Column: Legacy Manual Inspection */}
          <div className="rounded-2xl sm:rounded-3xl bg-[#0c0d12] border border-white/10 p-5 sm:p-8 flex flex-col justify-between shadow-xl">
            <div className="space-y-5 sm:space-y-6">
              <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-xl bg-white/[0.04] border border-white/10 text-zinc-400 flex items-center justify-center font-bold shrink-0">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base sm:text-lg text-white">Manual Desktop Inspection</h3>
                    <p className="text-[11px] font-mono text-zinc-400 font-medium">Standard spreadsheet applications</p>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-zinc-800/80 border border-white/10 text-zinc-300 text-[10px] font-mono font-semibold uppercase tracking-wider shrink-0">
                  Vulnerable
                </span>
              </div>

              <ul className="space-y-3 text-xs sm:text-sm text-zinc-300 font-sans">
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.06]">
                  <span className="h-2 w-2 rounded-full bg-zinc-500 shrink-0 mt-1.5" />
                  <div>
                    <strong className="text-white block font-medium">Disk Thrashing & Memory Caps:</strong>
                    <span className="text-zinc-400 text-xs">Spreadsheet software crashes on datasets exceeding 500k rows, forcing arbitrary sampling or sluggish UI freezes.</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.06]">
                  <span className="h-2 w-2 rounded-full bg-zinc-500 shrink-0 mt-1.5" />
                  <div>
                    <strong className="text-white block font-medium">Silent Type Coercion:</strong>
                    <span className="text-zinc-400 text-xs">Automated date conversions and stripped leading zeros silently corrupt account IDs, UPC codes, and currency fields.</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.06]">
                  <span className="h-2 w-2 rounded-full bg-zinc-500 shrink-0 mt-1.5" />
                  <div>
                    <strong className="text-white block font-medium">Untracked Cell Modifications:</strong>
                    <span className="text-zinc-400 text-xs">Ad-hoc cell overwrites and manual deletions leave zero lineage, making audit recreation mathematically impossible.</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.06]">
                  <span className="h-2 w-2 rounded-full bg-zinc-500 shrink-0 mt-1.5" />
                  <div>
                    <strong className="text-white block font-medium">Unverified Slide Exports:</strong>
                    <span className="text-zinc-400 text-xs">Static chart screenshots in decks lack underlying statistical verification, confidence bounds, or reproducibility.</span>
                  </div>
                </li>
              </ul>
            </div>

            <div className="mt-6 pt-4 border-t border-white/10 text-[11px] font-mono text-zinc-400 flex items-center justify-between">
              <span>Risk: Undetected schema drift & audit failures</span>
              <span className="text-zinc-400 bg-white/[0.04] border border-white/10 px-2 py-0.5 rounded text-[10px]">No Lineage DAG</span>
            </div>
          </div>

          {/* Right Column: The GetReport Engine Standard */}
          <div className="rounded-2xl sm:rounded-3xl bg-[#0c0d12] border border-white/15 p-5 sm:p-8 flex flex-col justify-between shadow-2xl relative overflow-hidden">
            <div className="space-y-5 sm:space-y-6 relative z-10">
              <div className="flex items-center justify-between gap-2 border-b border-white/10 pb-4">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 rounded-xl bg-white/[0.06] border border-white/15 text-white flex items-center justify-center shrink-0">
                    <ShieldCheck className="h-5 w-5 text-emerald-400" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base sm:text-lg text-white">GetReport Assurance Pipeline</h3>
                    <p className="text-[11px] font-mono text-zinc-400 font-medium">Automated Polars in-memory engine</p>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-[10px] font-mono font-semibold uppercase tracking-wider shrink-0">
                  Certified Lineage
                </span>
              </div>

              <ul className="space-y-3 text-xs sm:text-sm text-zinc-300 font-sans">
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.08]">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block font-medium">Zero-Copy In-Memory Polars Core:</strong>
                    <span className="text-zinc-400 text-xs">Processes millions of rows in milliseconds with strict Arrow type safety and zero persistent disk retention.</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.08]">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block font-medium">Multivariate Anomaly & VIF Analysis:</strong>
                    <span className="text-zinc-400 text-xs">Automatically flags multicollinearity (VIF &gt; 5.0), skewness, and partial duplicates before data reaches stakeholders.</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.08]">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block font-medium">Human-in-the-Loop Issue Ledger:</strong>
                    <span className="text-zinc-400 text-xs">Review, approve, or reject every proposed transformation in an immutable Directed Acyclic Graph (DAG).</span>
                  </div>
                </li>
                <li className="flex items-start gap-3 p-3.5 rounded-xl bg-[#121216] border border-white/[0.08]">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <div>
                    <strong className="text-white block font-medium">Board-Ready WeasyPrint PDF & Parquet:</strong>
                    <span className="text-zinc-400 text-xs">Export clean tabular datasets alongside publication-quality PDF audit reports containing full statistical certificates.</span>
                  </div>
                </li>
              </ul>
            </div>

            <div className="mt-6 pt-4 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between gap-3 relative z-10">
              <div className="text-[11px] font-mono text-zinc-400 w-full sm:w-auto text-left">
                Latency: <strong className="text-emerald-400 font-semibold font-mono">&lt; 50ms streaming</strong>
              </div>
              <Link to="/workspace" className="w-full sm:w-auto">
                <button
                  type="button"
                  className="w-full sm:w-auto min-h-[40px] h-10 px-5 rounded-full bg-white hover:bg-zinc-100 text-black font-sans font-semibold text-xs tracking-tight shadow-sm border border-white transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <span>Launch Ingestion Studio</span>
                  <ArrowRight className="h-3.5 w-3.5 text-black" />
                </button>
              </Link>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};

export default ComparisonSection;