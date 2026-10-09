import { XCircle, CheckCircle2, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

export const ComparisonSection = () => {
  return (
    <section className="py-10 sm:py-14 bg-[#07080a] border-b border-white/[0.08] relative overflow-hidden text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-2.5 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10 text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span>The Clear Difference</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-white tracking-[-0.02em] leading-tight">
            Manual spreadsheet stress vs. automated peace of mind
          </h2>
          <p className="text-xs sm:text-base text-zinc-400 max-w-xl mx-auto leading-relaxed font-normal font-sans">
            Teams lose hours manually checking rows, fixing broken formulas, and worrying about costly mistakes before high-stakes presentations.
          </p>
        </div>

        {/* 2-Column Comparison Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-8 items-stretch">
          
          {/* Left Column: The Broken Status Quo */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-b from-[#130d10] via-[#0f0b0d] to-[#0a0709] border border-rose-500/25 p-5 sm:p-8 flex flex-col justify-between shadow-2xl shadow-rose-950/20">
            <div className="space-y-5 sm:space-y-6">
              <div className="flex items-center justify-between gap-2 border-b border-rose-500/20 pb-3.5 sm:pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-rose-500/10 border border-rose-500/25 text-rose-400 flex items-center justify-center font-bold shrink-0">
                    <XCircle className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base sm:text-lg text-white">The Old Manual Way</h3>
                    <p className="text-[10px] sm:text-[11px] font-mono text-rose-300/80 font-medium">Manual scrolling, copy-pasting & guessing</p>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 text-[10px] font-mono font-semibold uppercase tracking-wider shrink-0">
                  High Risk
                </span>
              </div>

              <ul className="space-y-2.5 sm:space-y-3 text-xs sm:text-sm text-zinc-300 font-sans">
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#160f14]/80 border border-rose-500/15">
                  <XCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Hidden Blank Cells:</strong> Missing values slip past unnoticed, causing inaccurate business decisions and reporting errors.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#160f14]/80 border border-rose-500/15">
                  <XCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Accidental Data Loss:</strong> Modifying numbers or deleting rows by mistake with no record of what was altered.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#160f14]/80 border border-rose-500/15">
                  <XCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Broken Formulas:</strong> Corrupted formula characters and unexpected error codes break spreadsheets when shared with clients.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#160f14]/80 border border-rose-500/15">
                  <XCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Unverified Screenshots:</strong> Copy-pasting blurry charts into slide decks with no way to verify if the numbers are accurate.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#160f14]/80 border border-rose-500/15">
                  <XCircle className="h-4 w-4 text-rose-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Hours of Wasted Time:</strong> Manually scrolling through thousands of rows just to check if data is clean before a presentation.</span>
                </li>
              </ul>
            </div>

            <div className="mt-6 sm:mt-8 pt-3.5 sm:pt-4 border-t border-rose-500/20 text-[10px] sm:text-[11px] font-mono text-zinc-400 flex flex-col xs:flex-row gap-1.5 xs:items-center justify-between">
              <span>Risk: Unnoticed errors & wasted hours</span>
              <span className="text-rose-300 font-semibold bg-rose-500/10 border border-rose-500/20 px-2.5 py-0.5 rounded w-fit">No Audit History</span>
            </div>
          </div>

          {/* Right Column: The GetReport Engine Standard */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-b from-[#0e1322] via-[#0a0d18] to-[#070911] border border-indigo-500/35 p-5 sm:p-8 flex flex-col justify-between shadow-2xl shadow-indigo-950/25 relative overflow-hidden">
            <div className="space-y-5 sm:space-y-6 relative z-10">
              <div className="flex items-center justify-between gap-2 border-b border-indigo-500/20 pb-3.5 sm:pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 flex items-center justify-center shrink-0">
                    <CheckCircle2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-bold text-base sm:text-lg text-white">The GetReport Way</h3>
                    <p className="text-[10px] sm:text-[11px] font-mono text-indigo-300/80 font-medium">Fast, automated & completely verified</p>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-[10px] font-mono font-semibold uppercase tracking-wider shrink-0">
                  100% Verified
                </span>
              </div>

              <ul className="space-y-2.5 sm:space-y-3 text-xs sm:text-sm text-zinc-300 font-sans">
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#10162a]/80 border border-indigo-500/20">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Instant Results:</strong> Scan thousands or millions of rows in seconds without crashing your computer.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#10162a]/80 border border-indigo-500/20">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">You Stay in Control:</strong> Review every suggested fix before applying it, so your data is never altered without approval.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#10162a]/80 border border-indigo-500/20">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Safe & Clean Exports:</strong> Automatically removes corrupt formula characters so your spreadsheets open safely in Excel.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#10162a]/80 border border-indigo-500/20">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Comprehensive Checks:</strong> Automatically spots duplicate entries, unusual outliers, and conflicting columns.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-3 rounded-xl bg-[#10162a]/80 border border-indigo-500/20">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span><strong className="text-white font-medium">Executive PDF Reports:</strong> Download clean Excel files and polished executive summaries ready to present with one click.</span>
                </li>
              </ul>
            </div>

            <div className="mt-6 sm:mt-8 pt-4 sm:pt-6 border-t border-indigo-500/20 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4 relative z-10">
              <div className="text-[10px] sm:text-[11px] font-mono text-zinc-400 w-full sm:w-auto text-left">
                Average Scan Time: <strong className="text-emerald-400 font-semibold">Under 2 Seconds</strong>
              </div>
              <Link to="/workspace" className="w-full sm:w-auto">
                <button
                  type="button"
                  className="w-full sm:w-auto min-h-[44px] h-11 px-5 rounded-full bg-white hover:bg-zinc-100 text-black font-sans font-semibold text-xs tracking-tight shadow-[0_2px_12px_rgba(255,255,255,0.18),inset_0_1px_0_rgba(255,255,255,1)] border border-white transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <span>Start Checking Your Data Free</span>
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
