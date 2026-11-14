import { useState } from "react";
import { 
  SlidersHorizontal, 
  Terminal,
  Activity,
  Sliders,
  CheckCircle2,
  Trash2,
  Check,
  Cpu,
  FileSpreadsheet
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

export const FeaturesSection = () => {
  // Interactive Ledger state demonstrating core GetReport human-in-the-loop feature
  const [ledger, setLedger] = useState([
    { id: "1", column: "account_balance", issue: "Negative Balances (< $0.00)", fix: "Set to $0.00 or review manually", approved: true },
    { id: "2", column: "signup_date", issue: "Unreadable Date Numbers", fix: "Convert to standard date (YYYY-MM-DD)", approved: true },
    { id: "3", column: "customer_id", issue: "Duplicate Customer Entries", fix: "Keep the most recent record", approved: false },
    { id: "4", column: "company_notes", issue: "Dangerous Formula Symbols", fix: "Safely format as plain text", approved: true },
  ]);

  const handleSetItemStatus = (id: string, approved: boolean) => {
    setLedger((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, approved } : item
      )
    );
  };

  const handleSaveAll = () => {
    setLedger((prev) => prev.map((item) => ({ ...item, approved: true })));
  };

  const handleDiscardAll = () => {
    setLedger((prev) => prev.map((item) => ({ ...item, approved: false })));
  };

  const approvedCount = ledger.filter((i) => i.approved).length;
  const healthScore = Math.round(((12 + approvedCount) / 16) * 100);

  return (
    <section id="features" className="py-10 sm:py-14 bg-[#07080a] border-b border-white/[0.08] relative overflow-hidden text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-2.5 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10 text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span>Transparency & Control</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-stone-900 dark:text-white tracking-[-0.02em] leading-tight">
            You stay in complete control of every change
          </h2>
          <p className="text-xs sm:text-base text-stone-600 dark:text-zinc-400 max-w-xl mx-auto leading-relaxed font-sans font-normal">
            GetReport never alters your data behind your back. Every error is highlighted, and you choose what to fix before generating reports.
          </p>
        </div>

        {/* Bento Grid with Dual-Mode Surfaces */}
        <div className="space-y-4 sm:space-y-5">
          
          {/* Top Bento Row: 2-col Ledger + 1-col Stacked Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8 items-stretch">
            
            {/* Tile 1: Interactive Issue Ledger (Spans 2 columns) */}
            <div className="lg:col-span-2 rounded-[1.75rem] sm:rounded-[2rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-7 lg:p-8 shadow-md dark:shadow-2xl flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-stone-200/90 dark:border-white/10 pb-3.5 sm:pb-4">
                  <div className="flex items-center gap-2.5 sm:gap-3">
                    <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-violet-100 text-violet-800 border border-violet-200 dark:bg-violet-500/10 dark:border-violet-500/25 dark:text-violet-400 flex items-center justify-center font-bold shrink-0">
                      <SlidersHorizontal className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-display font-bold text-base sm:text-lg text-stone-900 dark:text-white">Interactive Review Gate</h3>
                      <p className="text-[11px] sm:text-xs text-stone-600 dark:text-zinc-400 font-medium">Review suggested fixes with a single tap before applying them</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 self-start sm:self-auto">
                    <span className="text-xs font-mono text-stone-500 dark:text-zinc-400 font-medium">Health Score:</span>
                    <span className="text-base sm:text-lg font-display font-black text-violet-900 bg-violet-100 border border-violet-300 dark:text-white dark:bg-violet-600/30 dark:border-violet-500/40 px-2.5 py-0.5 rounded-full">
                      {healthScore}%
                    </span>
                  </div>
                </div>

                {/* Action Toolbar with Save All & Discard All Buttons */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                  <p className="text-xs sm:text-sm text-stone-700 dark:text-zinc-300 font-sans font-medium">
                    Review and test the live action buttons below:
                  </p>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={handleSaveAll}
                      className="min-h-[44px] sm:min-h-0 h-9 px-4 rounded-full font-sans text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-200 hover:-translate-y-0.5 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 bg-[#800000] hover:bg-[#6b0000] text-white border border-[#800000] dark:bg-emerald-600 dark:hover:bg-emerald-500 dark:border-emerald-500/80 shadow-sm group"
                      title="Approve and apply all suggested fixes"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-white stroke-[2.2]" />
                      <span>Save All</span>
                      {approvedCount < ledger.length && (
                        <span className="ml-0.5 px-1.5 py-0.2 bg-white/20 text-white text-[10px] font-mono rounded-full font-bold">
                          +{ledger.length - approvedCount}
                        </span>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={handleDiscardAll}
                      disabled={approvedCount === 0}
                      className={`min-h-[44px] sm:min-h-0 h-9 px-3.5 rounded-full font-sans text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 group ${
                        approvedCount === 0
                          ? "bg-stone-200/50 text-stone-400 border border-stone-200 cursor-not-allowed opacity-50 shadow-none dark:bg-white/[0.03] dark:text-zinc-600 dark:border-white/5"
                          : "bg-stone-200/80 hover:bg-rose-100 text-stone-700 hover:text-rose-900 border border-stone-300 hover:border-rose-300 dark:bg-white/[0.06] dark:hover:bg-rose-500/15 dark:text-zinc-300 dark:hover:text-rose-300 dark:border-white/10 dark:hover:border-rose-500/30 hover:-translate-y-0.5 active:scale-95"
                      }`}
                      title={approvedCount === 0 ? "All fixes already discarded" : "Discard all suggested fixes"}
                    >
                      <Trash2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${approvedCount === 0 ? "text-stone-400 dark:text-zinc-600" : "text-stone-600 dark:text-zinc-400 group-hover:text-rose-600 dark:group-hover:text-rose-400 stroke-[2]"}`} />
                      <span>Discard All</span>
                    </button>
                  </div>
                </div>

                {/* Interactive Ledger Rows */}
                <div className="space-y-2.5 pt-1">
                  {ledger.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 sm:p-3.5 rounded-xl border transition-all duration-150 flex flex-col xs:flex-row xs:items-center justify-between gap-2.5 sm:gap-3 select-none ${
                        item.approved
                          ? "bg-emerald-50 border-emerald-300 ring-1 ring-emerald-300/40 dark:bg-[#0d1620]/90 dark:border-emerald-500/40 dark:ring-emerald-500/15"
                          : "bg-stone-100/90 border-stone-200/90 dark:bg-[#11131c]/90 dark:border-white/10"
                      }`}
                    >
                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-mono text-xs font-semibold text-stone-900 bg-stone-200/90 px-2 py-0.5 rounded border border-stone-300 dark:text-zinc-200 dark:bg-white/[0.06] dark:border-white/10">
                            {item.column}
                          </span>
                          <Badge variant="outline" className="text-[10px] font-sans px-2 py-0.5 bg-violet-100 text-violet-800 border-violet-200 dark:bg-violet-500/10 dark:border-violet-500/25 dark:text-violet-300 font-medium rounded-full">
                            {item.issue}
                          </Badge>
                        </div>
                        <div className="text-[11px] sm:text-xs font-sans text-stone-600 dark:text-zinc-400">
                          Suggested Action: <strong className="text-stone-900 dark:text-white font-medium">{item.fix}</strong>
                        </div>
                      </div>

                      {/* High-End Save Fix and Delete Action Buttons */}
                      <div className="flex items-center gap-2 self-end xs:self-auto shrink-0">
                        <button
                          type="button"
                          onClick={() => handleSetItemStatus(item.id, true)}
                          className={`min-h-[44px] sm:min-h-0 h-8.5 px-3.5 rounded-full font-sans text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                            item.approved
                              ? "bg-emerald-700 hover:bg-emerald-600 text-white border border-emerald-700 dark:bg-emerald-600 dark:hover:bg-emerald-500 dark:border-emerald-500 font-bold shadow-xs"
                              : "bg-stone-200/90 hover:bg-emerald-100 text-stone-700 hover:text-emerald-900 border border-stone-300 hover:border-emerald-300 dark:bg-white/[0.06] dark:hover:bg-emerald-500/15 dark:text-zinc-300 dark:hover:text-emerald-300 dark:border-white/10 dark:hover:border-emerald-500/30 group"
                          }`}
                          title="Save & apply this fix"
                        >
                          <CheckCircle2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${item.approved ? "text-white" : "text-stone-500 group-hover:text-emerald-700 dark:text-zinc-400 dark:group-hover:text-emerald-400"}`} />
                          <span>Save Fix</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => handleSetItemStatus(item.id, false)}
                          className={`min-h-[44px] sm:min-h-0 h-8.5 px-3.5 rounded-full font-sans text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 ${
                            !item.approved
                              ? "bg-rose-700 hover:bg-rose-600 text-white border border-rose-700 dark:bg-rose-600 dark:hover:bg-rose-500 dark:border-rose-500 font-bold shadow-xs"
                              : "bg-stone-200/90 hover:bg-rose-100 text-stone-700 hover:text-rose-900 border border-stone-300 hover:border-rose-300 dark:bg-white/[0.06] dark:hover:bg-rose-500/15 dark:text-zinc-300 dark:hover:text-rose-300 dark:border-white/10 dark:hover:border-rose-500/30 group"
                          }`}
                          title="Delete or skip this fix"
                        >
                          <Trash2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${!item.approved ? "text-white" : "text-stone-500 group-hover:text-rose-700 dark:text-zinc-400 dark:group-hover:text-rose-400"}`} />
                          <span>Delete</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-5 pt-3.5 border-t border-stone-200/90 dark:border-white/10 text-[10px] sm:text-[11px] font-sans text-stone-600 dark:text-zinc-400 flex flex-col xs:flex-row gap-1.5 xs:items-center justify-between">
                <span>{approvedCount} of {ledger.length} fixes approved</span>
                <span className="text-violet-800 dark:text-violet-300 font-semibold w-fit">Every change is clearly documented</span>
              </div>
            </div>

            {/* Right Column: 2 Stacked Cards */}
            <div className="lg:col-span-1 flex flex-col gap-5 sm:gap-6 justify-between">
              
              {/* Card 2A: Instant Filters & Search */}
              <div className="rounded-[1.5rem] sm:rounded-[1.75rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-6 shadow-md dark:shadow-xl flex flex-col justify-between flex-1">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="h-8 w-8 rounded-xl bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/[0.05] dark:border-white/10 dark:text-zinc-300 flex items-center justify-center font-bold">
                      <Terminal className="h-4 w-4" />
                    </div>
                    <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-stone-800 bg-stone-200/90 px-2 py-0.5 rounded-full border border-stone-300 dark:text-zinc-300 dark:bg-white/[0.05] dark:border-white/10">
                      Sub-second
                    </span>
                  </div>

                  <div>
                    <h3 className="font-display font-bold text-base text-stone-900 dark:text-white">Instant Filters & Search</h3>
                    <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-medium mt-0.5">Filter and slice any column in milliseconds</p>
                  </div>

                  <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                    Slice customer tiers or revenue categories without waiting or crashing your spreadsheet.
                  </p>

                  <div className="p-2.5 bg-stone-100/90 rounded-xl border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 space-y-1 font-sans text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-stone-900 dark:text-zinc-200 font-semibold text-[10.5px]">High-Value Accounts</span>
                      <span className="text-[10px] font-mono text-emerald-900 bg-emerald-100 border border-emerald-300 dark:text-emerald-300 dark:bg-emerald-500/10 dark:border-emerald-500/20 font-bold px-1.5 py-0.2 rounded">
                        500 rows verified
                      </span>
                    </div>
                    <div className="text-stone-600 dark:text-zinc-400 text-[11px]">
                      Sorted by: <span className="font-semibold text-stone-900 dark:text-white">Highest Revenue First</span>
                    </div>
                  </div>
                </div>

                <div className="pt-3 mt-3 border-t border-stone-200/90 dark:border-white/10 text-[10px] font-mono text-stone-600 dark:text-zinc-400 font-medium">
                  Instant Local Processing
                </div>
              </div>

              {/* Card 2B: Multi-Column Visual Discovery */}
              <div className="rounded-[1.5rem] sm:rounded-[1.75rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-6 shadow-md dark:shadow-xl flex flex-col justify-between flex-1">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="h-8 w-8 rounded-xl bg-stone-200 text-stone-800 border border-stone-300 dark:bg-white/[0.05] dark:border-white/10 dark:text-zinc-300 flex items-center justify-center font-bold">
                      <Sliders className="h-4 w-4" />
                    </div>
                    <span className="text-[10px] font-mono font-semibold uppercase tracking-wider text-stone-800 bg-stone-200/90 px-2 py-0.5 rounded-full border border-stone-300 dark:text-zinc-300 dark:bg-white/[0.05] dark:border-white/10">
                      Visual Sparklines
                    </span>
                  </div>

                  <div>
                    <h3 className="font-display font-bold text-base text-stone-900 dark:text-white">Multi-Column Discovery</h3>
                    <p className="text-[11px] text-stone-600 dark:text-zinc-400 font-medium mt-0.5">Compare multiple metrics on a single screen</p>
                  </div>

                  <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                    Slide across dates, revenue, and categories to spot anomalies and correlations instantly.
                  </p>

                  <div className="p-2.5 bg-stone-100/90 rounded-xl border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 space-y-1.5 font-sans text-xs">
                    <div className="flex items-center justify-between text-[10.5px] text-stone-800 dark:text-zinc-300 font-medium">
                      <span>Cohort Distribution</span>
                      <span className="font-mono text-emerald-800 dark:text-emerald-400 font-bold">+24.8% Active</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1.5 h-6 items-end pt-0.5">
                      <div className="bg-emerald-300 dark:bg-emerald-500/30 rounded-t h-[40%]" />
                      <div className="bg-emerald-400 dark:bg-emerald-500/50 rounded-t h-[65%]" />
                      <div className="bg-emerald-500 dark:bg-emerald-500/70 rounded-t h-[85%]" />
                      <div className="bg-emerald-600 dark:bg-emerald-500 rounded-t h-[100%]" />
                    </div>
                  </div>
                </div>

                <div className="pt-3 mt-3 border-t border-stone-200/90 dark:border-white/10 text-[10px] font-mono text-stone-600 dark:text-zinc-400 font-medium">
                  Interactive Visual Charts
                </div>
              </div>

            </div>
          </div>

          {/* Bottom Bento Row: 3 Equal Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8">
            
            {/* Tile 4: Outlier & Anomaly Detection */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-7 shadow-md dark:shadow-xl flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-amber-100 text-amber-900 border border-amber-300 dark:bg-amber-500/10 dark:border-amber-500/25 dark:text-amber-400 flex items-center justify-center font-bold">
                  <Activity className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-bold text-base sm:text-lg text-stone-900 dark:text-white">Outlier & Anomaly Detection</h3>
                  <p className="text-[11px] sm:text-xs text-stone-600 dark:text-zinc-400 mt-0.5 sm:mt-1 font-medium">Automatic quality and consistency scoring</p>
                </div>

                <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                  Automatically spots extreme outliers, unexpected data spikes, and conflicting fields before they skew your calculations.
                </p>

                <div className="grid grid-cols-2 gap-2 text-xs font-sans">
                  <div className="p-2.5 rounded-xl bg-stone-100/90 border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 text-center">
                    <span className="text-[10px] text-stone-500 dark:text-zinc-400 block font-medium">Data Balance</span>
                    <span className="font-bold text-stone-900 dark:text-white text-xs sm:text-sm mt-0.5 block">Healthy & Normal</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-stone-100/90 border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 text-center">
                    <span className="text-[10px] text-stone-500 dark:text-zinc-400 block font-medium">Conflict Check</span>
                    <span className="font-bold text-emerald-800 dark:text-emerald-400 text-xs sm:text-sm mt-0.5 block">0 Conflicts</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-stone-200/90 dark:border-white/10 text-[10px] sm:text-[11px] font-mono text-amber-800 dark:text-amber-300 font-medium">
                Automated Error Detection
              </div>
            </div>

            {/* Tile 5: Formula Protection & Safety Guard */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-7 shadow-md dark:shadow-xl flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-teal-100 text-teal-900 border border-teal-300 dark:bg-teal-500/10 dark:border-teal-500/25 dark:text-teal-400 flex items-center justify-center font-bold">
                  <Cpu className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-bold text-base sm:text-lg text-stone-900 dark:text-white">Formula Protection Guard</h3>
                  <p className="text-[11px] sm:text-xs text-stone-600 dark:text-zinc-400 mt-0.5 sm:mt-1 font-medium">Zero broken #REF! or formula corruptions</p>
                </div>

                <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                  Protects calculation trees and formulas during cell cleanups so your team never inherits broken workbook sheets.
                </p>

                <div className="grid grid-cols-2 gap-2 text-xs font-sans">
                  <div className="p-2.5 rounded-xl bg-stone-100/90 border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 text-center">
                    <span className="text-[10px] text-stone-500 dark:text-zinc-400 block font-medium">Formula Trees</span>
                    <span className="font-bold text-stone-900 dark:text-white text-xs sm:text-sm mt-0.5 block">100% Protected</span>
                  </div>
                  <div className="p-2.5 rounded-xl bg-stone-100/90 border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 text-center">
                    <span className="text-[10px] text-stone-500 dark:text-zinc-400 block font-medium">Syntax Validation</span>
                    <span className="font-bold text-emerald-800 dark:text-emerald-400 text-xs sm:text-sm mt-0.5 block">Zero Errors</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-stone-200/90 dark:border-white/10 text-[10px] sm:text-[11px] font-mono text-teal-800 dark:text-teal-300 font-medium">
                Safe Calculation Architecture
              </div>
            </div>

            {/* Tile 6: Export in Any Format */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-[#FAF8F5] border border-stone-200/90 dark:bg-[#0c0d12] dark:border-white/10 p-5 sm:p-7 shadow-md dark:shadow-xl flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-emerald-100 text-emerald-900 border border-emerald-300 dark:bg-emerald-500/10 dark:border-emerald-500/25 dark:text-emerald-400 flex items-center justify-center font-bold">
                  <FileSpreadsheet className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-bold text-base sm:text-lg text-stone-900 dark:text-white">Export in Any Format</h3>
                  <p className="text-[11px] sm:text-xs text-stone-600 dark:text-zinc-400 mt-0.5 sm:mt-1 font-medium">Excel, PDF, and automated validation</p>
                </div>

                <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                  Download cleaned spreadsheets ready for your team, alongside executive PDF briefs and automated pipeline quality rules.
                </p>

                <div className="p-2.5 sm:p-3 bg-stone-100/90 border border-stone-200/90 dark:bg-[#121215] dark:border-white/10 rounded-xl space-y-1.5 font-sans text-[11px]">
                  <div className="text-stone-800 dark:text-zinc-200 font-medium flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400 shrink-0 stroke-[2.5]" />
                    <span>Clean Excel Spreadsheet (.xlsx)</span>
                  </div>
                  <div className="text-stone-800 dark:text-zinc-200 font-medium flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400 shrink-0 stroke-[2.5]" />
                    <span>Executive Summary PDF (.pdf)</span>
                  </div>
                  <div className="text-stone-800 dark:text-zinc-300 font-medium flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-700 dark:text-emerald-400 shrink-0 stroke-[2.5]" />
                    <span>Automated Pipeline Rules (.json)</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-stone-200/90 dark:border-white/10 text-[10px] sm:text-[11px] font-mono text-emerald-800 dark:text-emerald-300 font-medium">
                Ready for Teams & Executives
              </div>
            </div>

          </div>
        </div>

      </div>
    </section>
  );
};

export default FeaturesSection;
