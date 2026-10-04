import { XCircle, CheckCircle2, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const ComparisonSection = () => {
  return (
    <section className="py-10 sm:py-14 bg-gradient-to-b from-[#EAEFF5]/80 via-[#F1F5F9]/60 to-[#E8EDF5]/80 backdrop-blur-xs border-b border-slate-300/80 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-3 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-200/70 border border-slate-300 text-slate-700 text-[11px] font-mono font-semibold tracking-wide">
            <span>The Clear Difference</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-foreground tracking-[-0.02em] leading-tight">
            Manual spreadsheet stress vs. automated peace of mind
          </h2>
          <p className="text-xs sm:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed font-normal font-sans">
            Teams lose hours manually checking rows, fixing broken formulas, and worrying about costly mistakes before big presentations.
          </p>
        </div>

        {/* 2-Column Comparison Grid with Separate Distinct Card Colors */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-8 items-stretch">
          
          {/* Left Column: The Broken Status Quo (Distinct Rose/Coral Card) */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-rose-100 via-rose-50 to-red-100/80 border-2 border-rose-300 p-5 sm:p-8 flex flex-col justify-between shadow-xl shadow-rose-500/5">
            <div className="space-y-5 sm:space-y-6">
              <div className="flex items-center justify-between gap-2 border-b border-rose-200 pb-3.5 sm:pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-rose-600 text-white flex items-center justify-center font-bold shadow-md shadow-rose-600/20 shrink-0">
                    <XCircle className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-black text-base sm:text-lg text-rose-950">The Old Manual Way</h3>
                    <p className="text-[10px] sm:text-[11px] font-mono text-rose-700 font-medium">Manual scrolling, copy-pasting & guessing</p>
                  </div>
                </div>
                <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-full bg-rose-200/80 text-rose-800 text-[9px] sm:text-[10px] font-mono font-bold uppercase tracking-wider shrink-0">
                  High Risk
                </span>
              </div>

              <ul className="space-y-2.5 sm:space-y-3 text-xs sm:text-sm text-slate-800 font-sans">
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-rose-200 shadow-xs">
                  <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                  <span><strong className="text-rose-950">Hidden Blank Cells:</strong> Missing values slip past unnoticed, causing inaccurate business decisions and reporting errors.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-rose-200 shadow-xs">
                  <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                  <span><strong className="text-rose-950">Accidental Data Loss:</strong> Modifying numbers or deleting rows by mistake with no record of what was altered.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-rose-200 shadow-xs">
                  <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                  <span><strong className="text-rose-950">Broken Formulas:</strong> Corrupted formula characters and unexpected error codes break spreadsheets when shared with clients.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-rose-200 shadow-xs">
                  <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                  <span><strong className="text-rose-950">Unverified Screenshots:</strong> Copy-pasting blurry charts into slide decks with no way to verify if the numbers are accurate.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-rose-200 shadow-xs">
                  <XCircle className="h-4 w-4 text-rose-600 shrink-0 mt-0.5" />
                  <span><strong className="text-rose-950">Hours of Wasted Time:</strong> Manually scrolling through thousands of rows just to check if the data is clean before a presentation.</span>
                </li>
              </ul>
            </div>

            <div className="mt-6 sm:mt-8 pt-3.5 sm:pt-4 border-t border-rose-200 text-[10px] sm:text-[11px] font-mono text-rose-800 flex flex-col xs:flex-row gap-1.5 xs:items-center justify-between">
              <span>Risk: Unnoticed errors & wasted time</span>
              <span className="text-rose-700 font-bold bg-rose-200/60 px-2 py-0.5 rounded w-fit">No Error History</span>
            </div>
          </div>

          {/* Right Column: The GetReport Engine Standard (Distinct Electric Blue Card) */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-blue-100 via-sky-50 to-indigo-100/90 border-2 border-blue-400 p-5 sm:p-8 flex flex-col justify-between shadow-xl shadow-blue-500/10 relative overflow-hidden">
            <div className="space-y-5 sm:space-y-6 relative z-10">
              <div className="flex items-center justify-between gap-2 border-b border-blue-200 pb-3.5 sm:pb-4">
                <div className="flex items-center gap-2.5">
                  <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-600/25 font-bold shrink-0">
                    <CheckCircle2 className="h-5 w-5" />
                  </div>
                  <div>
                    <h3 className="font-display font-black text-base sm:text-lg text-blue-950">The GetReport Way</h3>
                    <p className="text-[10px] sm:text-[11px] font-mono text-blue-700 font-semibold">Fast, automated & completely verified</p>
                  </div>
                </div>
                <span className="px-2 py-0.5 sm:px-2.5 sm:py-1 rounded-full bg-blue-200/80 text-blue-800 text-[9px] sm:text-[10px] font-mono font-bold uppercase tracking-wider shrink-0">
                  100% Verified
                </span>
              </div>

              <ul className="space-y-2.5 sm:space-y-3 text-xs sm:text-sm text-slate-800 font-sans">
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-blue-200 shadow-xs">
                  <CheckCircle2 className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
                  <span><strong className="text-blue-950">Instant Results:</strong> Scan thousands or millions of rows in seconds without crashing your computer.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-blue-200 shadow-xs">
                  <CheckCircle2 className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
                  <span><strong className="text-blue-950">You Stay in Control:</strong> Review every suggested fix before applying it, so your data is never altered without your approval.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-blue-200 shadow-xs">
                  <CheckCircle2 className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
                  <span><strong className="text-blue-950">Safe & Clean Exports:</strong> Automatically removes corrupt formula characters so your spreadsheets open safely in Excel.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-blue-200 shadow-xs">
                  <CheckCircle2 className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
                  <span><strong className="text-blue-950">Comprehensive Checks:</strong> Automatically spots duplicate entries, unusual outliers, and conflicting columns.</span>
                </li>
                <li className="flex items-start gap-2.5 sm:gap-3 p-2.5 sm:p-3 rounded-xl bg-white/90 border border-blue-200 shadow-xs">
                  <CheckCircle2 className="h-4 w-4 text-blue-600 shrink-0 mt-0.5" />
                  <span><strong className="text-blue-950">Executive PDF Reports:</strong> Download clean Excel files and polished executive summaries ready to present with one click.</span>
                </li>
              </ul>
            </div>

            <div className="mt-6 sm:mt-8 pt-4 sm:pt-6 border-t border-blue-200 flex flex-col sm:flex-row items-center justify-between gap-3 sm:gap-4 relative z-10">
              <div className="text-[10px] sm:text-[11px] font-mono text-blue-900 w-full sm:w-auto text-left">
                Average Scan Time: <strong className="text-blue-950 font-bold">Under 2 Seconds</strong>
              </div>
              <Link to="/workspace" className="w-full sm:w-auto">
                <Button size="sm" className="w-full sm:w-auto h-11 sm:h-9 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-semibold text-xs gap-1.5 shadow-md shadow-violet-600/25 ring-1 ring-white/20 active:scale-95 transition-all">
                  <span>Start Checking Your Data Free</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Button>
              </Link>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};

export default ComparisonSection;
