import { useState } from "react";
import { 
  SlidersHorizontal, 
  ShieldCheck, 
  Terminal,
  Activity,
  Sliders,
  CheckCircle2,
  XCircle,
  Trash2,
  Check,
  Cpu,
  FileSpreadsheet
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

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
    <section id="features" className="py-10 sm:py-14 bg-gradient-to-b from-[#EEF2F6]/75 via-[#F1F5F9]/55 to-[#E2E8F0]/75 backdrop-blur-xs border-b border-slate-300/80 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-3 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-violet-100 border border-violet-200 text-violet-800 text-[11px] font-mono font-semibold tracking-wide">
            <span>Transparency & Control</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-foreground tracking-[-0.02em] leading-tight">
            You stay in complete control of every change
          </h2>
          <p className="text-xs sm:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed font-sans font-normal">
            GetReport never alters your data behind your back. Every error is highlighted, and you choose what to fix before generating reports.
          </p>
        </div>

        {/* Bento Grid with Separate Distinct Card Colors & Zero Empty Space */}
        <div className="space-y-5 sm:space-y-6 lg:space-y-8">
          
          {/* Top Bento Row: 2-col Ledger + 1-col Stacked Cards (Filters & Discovery) */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8 items-stretch">
            
            {/* Tile 1: Interactive Issue Ledger (Spans 2 columns) */}
            <div className="lg:col-span-2 rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-violet-100 via-purple-50 to-indigo-100/90 border-2 border-violet-300 p-5 sm:p-7 lg:p-8 shadow-lg flex flex-col justify-between">
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-violet-200 pb-3.5 sm:pb-4">
                  <div className="flex items-center gap-2.5 sm:gap-3">
                    <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-violet-600 text-white flex items-center justify-center font-bold shadow-md shadow-violet-600/20 shrink-0">
                      <SlidersHorizontal className="h-5 w-5" />
                    </div>
                    <div>
                      <h3 className="font-display font-black text-base sm:text-lg text-violet-950">Interactive Review Gate</h3>
                      <p className="text-[11px] sm:text-xs text-violet-800 font-medium">Review suggested fixes with a single tap before applying them</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 self-start sm:self-auto">
                    <span className="text-xs font-mono text-violet-800 font-semibold">Health Score:</span>
                    <span className="text-base sm:text-lg font-display font-black text-white bg-violet-600 px-2.5 py-0.5 rounded-full shadow-xs">
                      {healthScore}%
                    </span>
                  </div>
                </div>

                {/* Action Toolbar with Save All & Discard All Buttons */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                  <p className="text-xs sm:text-sm text-violet-950 font-sans font-medium">
                    Review and test the live action buttons below:
                  </p>
                  <div className="flex items-center gap-2 shrink-0">
                    <button
                      type="button"
                      onClick={handleSaveAll}
                      className="h-8.5 px-4 rounded-full font-hero text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-200 hover:-translate-y-0.5 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 bg-gradient-to-r from-emerald-600 via-emerald-500 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-[0_2px_8px_-1px_rgba(16,185,129,0.36),inset_0_1px_0_rgba(255,255,255,0.25)] border border-emerald-500/80 group"
                      title="Approve and apply all suggested fixes"
                    >
                      <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-white stroke-[2.2]" />
                      <span>Save All</span>
                      {approvedCount < ledger.length && (
                        <span className="ml-0.5 px-1.5 py-0.2 bg-white/25 text-white text-[10px] font-mono rounded-full font-bold">
                          +{ledger.length - approvedCount}
                        </span>
                      )}
                    </button>

                    <button
                      type="button"
                      onClick={handleDiscardAll}
                      disabled={approvedCount === 0}
                      className={`h-8.5 px-3.5 rounded-full font-hero text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 group ${
                        approvedCount === 0
                          ? "bg-white/60 text-slate-400 border border-slate-200/60 cursor-not-allowed opacity-60 shadow-none"
                          : "bg-white/95 hover:bg-rose-50/90 text-slate-700 hover:text-rose-700 border border-slate-200/90 hover:border-rose-300 shadow-2xs hover:shadow-xs hover:-translate-y-0.5 active:scale-95"
                      }`}
                      title={approvedCount === 0 ? "All fixes already discarded" : "Discard all suggested fixes"}
                    >
                      <Trash2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${approvedCount === 0 ? "text-slate-300" : "text-slate-400 group-hover:text-rose-600 stroke-[2]"}`} />
                      <span>Discard All</span>
                    </button>
                  </div>
                </div>

                {/* Interactive Ledger Rows */}
                <div className="space-y-2.5 pt-1">
                  {ledger.map((item) => (
                    <div
                      key={item.id}
                      className={`p-3 sm:p-3.5 rounded-xl border transition-all duration-150 flex flex-col xs:flex-row xs:items-center justify-between gap-2.5 sm:gap-3 shadow-xs select-none ${
                        item.approved
                          ? "bg-white/95 border-emerald-300 shadow-xs ring-1 ring-emerald-500/10"
                          : "bg-white/85 border-slate-200/90 shadow-2xs"
                      }`}
                    >
                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-mono text-xs font-bold text-slate-900 bg-slate-100/90 px-1.5 py-0.5 rounded border border-slate-200/80">
                            {item.column}
                          </span>
                          <Badge variant="outline" className="text-[10px] font-hero px-2 py-0.5 bg-violet-50 border-violet-200 text-violet-900 font-semibold rounded-full">
                            {item.issue}
                          </Badge>
                        </div>
                        <div className="text-[11px] sm:text-xs font-hero text-slate-600">
                          Suggested Action: <strong className="text-slate-900 font-semibold">{item.fix}</strong>
                        </div>
                      </div>

                      {/* Dedicated High-End Save Fix and Delete Action Buttons */}
                      <div className="flex items-center gap-2 self-end xs:self-auto shrink-0">
                        <button
                          type="button"
                          onClick={() => handleSetItemStatus(item.id, true)}
                          className={`h-8 px-3.5 rounded-full font-hero text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${
                            item.approved
                              ? "bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white shadow-[0_2px_8px_-1px_rgba(16,185,129,0.36),inset_0_1px_0_rgba(255,255,255,0.25)] border border-emerald-500/80 font-bold"
                              : "bg-white hover:bg-emerald-50/90 text-slate-600 hover:text-emerald-700 border border-slate-200 hover:border-emerald-300 shadow-2xs group"
                          }`}
                          title="Save & apply this fix"
                        >
                          <CheckCircle2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${item.approved ? "text-white" : "text-slate-400 group-hover:text-emerald-600"}`} />
                          <span>Save Fix</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => handleSetItemStatus(item.id, false)}
                          className={`h-8 px-3.5 rounded-full font-hero text-xs font-semibold tracking-tight flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 ${
                            !item.approved
                              ? "bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white shadow-[0_2px_8px_-1px_rgba(244,63,94,0.36),inset_0_1px_0_rgba(255,255,255,0.25)] border border-rose-500/80 font-bold"
                              : "bg-white hover:bg-rose-50/90 text-slate-600 hover:text-rose-700 border border-slate-200 hover:border-rose-300 shadow-2xs group"
                          }`}
                          title="Delete or skip this fix"
                        >
                          <Trash2 className={`h-3.5 w-3.5 shrink-0 transition-colors ${!item.approved ? "text-white" : "text-slate-400 group-hover:text-rose-600"}`} />
                          <span>Delete</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-5 pt-3.5 border-t border-violet-200 text-[10px] sm:text-[11px] font-hero text-violet-900 flex flex-col xs:flex-row gap-1.5 xs:items-center justify-between">
                <span>{approvedCount} of {ledger.length} fixes approved</span>
                <span className="text-violet-700 font-bold w-fit">Every change is clearly documented</span>
              </div>
            </div>

            {/* Right Column: 2 Stacked Cards (Accommodates the Empty Space!) */}
            <div className="lg:col-span-1 flex flex-col gap-5 sm:gap-6 justify-between">
              
              {/* Card 2A: Instant Filters & Search */}
              <div className="rounded-[1.5rem] sm:rounded-[1.75rem] bg-gradient-to-br from-blue-100 via-indigo-50 to-sky-100/90 border-2 border-blue-300 p-5 sm:p-6 shadow-md flex flex-col justify-between flex-1">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="h-8 w-8 rounded-xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-600/20">
                      <Terminal className="h-4 w-4" />
                    </div>
                    <span className="text-[10px] font-hero font-bold uppercase tracking-wider text-blue-700 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-200">
                      Sub-second
                    </span>
                  </div>

                  <div>
                    <h3 className="font-display font-black text-base text-blue-950">Instant Filters & Search</h3>
                    <p className="text-[11px] text-blue-700 font-medium mt-0.5">Filter and slice any column in milliseconds</p>
                  </div>

                  <p className="text-xs text-blue-950 leading-relaxed font-sans font-normal">
                    Slice customer tiers or revenue categories without waiting or crashing your spreadsheet.
                  </p>

                  <div className="p-2.5 bg-white/95 rounded-xl border border-blue-200 space-y-1 shadow-2xs font-hero text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-blue-700 font-semibold text-[10.5px]">High-Value Accounts</span>
                      <span className="text-[10px] font-mono text-emerald-700 font-bold bg-emerald-50 px-1.5 py-0.2 rounded border border-emerald-200">
                        500 rows verified
                      </span>
                    </div>
                    <div className="text-slate-600 text-[11px]">
                      Sorted by: <span className="font-semibold text-slate-900">Highest Revenue First</span>
                    </div>
                  </div>
                </div>

                <div className="pt-3 mt-3 border-t border-blue-200 text-[10px] font-hero text-blue-900 font-bold">
                  Instant Local Processing
                </div>
              </div>

              {/* Card 2B: Multi-Column Visual Discovery */}
              <div className="rounded-[1.5rem] sm:rounded-[1.75rem] bg-gradient-to-br from-sky-100 via-cyan-50 to-blue-100/90 border-2 border-sky-300 p-5 sm:p-6 shadow-md flex flex-col justify-between flex-1">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="h-8 w-8 rounded-xl bg-sky-600 text-white flex items-center justify-center shadow-md shadow-sky-600/20">
                      <Sliders className="h-4 w-4" />
                    </div>
                    <span className="text-[10px] font-hero font-bold uppercase tracking-wider text-sky-700 bg-sky-50 px-2 py-0.5 rounded-full border border-sky-200">
                      Visual Sparklines
                    </span>
                  </div>

                  <div>
                    <h3 className="font-display font-black text-base text-sky-950">Multi-Column Discovery</h3>
                    <p className="text-[11px] text-sky-700 font-medium mt-0.5">Compare multiple metrics on a single screen</p>
                  </div>

                  <p className="text-xs text-sky-950 leading-relaxed font-sans font-normal">
                    Slide across dates, revenue, and categories to spot anomalies and correlations instantly.
                  </p>

                  <div className="p-2.5 bg-white/95 rounded-xl border border-sky-200/90 shadow-2xs space-y-1.5 font-hero text-xs">
                    <div className="flex items-center justify-between text-[10.5px] text-sky-900 font-semibold">
                      <span>Cohort Distribution</span>
                      <span className="font-mono text-sky-700 font-bold">+24.8% Active</span>
                    </div>
                    <div className="grid grid-cols-4 gap-1.5 h-6 items-end pt-0.5">
                      <div className="bg-sky-200 rounded-t h-[40%]" />
                      <div className="bg-sky-300 rounded-t h-[65%]" />
                      <div className="bg-sky-400 rounded-t h-[85%]" />
                      <div className="bg-sky-600 rounded-t h-[100%]" />
                    </div>
                  </div>
                </div>

                <div className="pt-3 mt-3 border-t border-sky-200 text-[10px] font-hero text-sky-800 font-bold">
                  Interactive Visual Charts
                </div>
              </div>

            </div>
          </div>

          {/* Bottom Bento Row: 3 Equal Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8">
            
            {/* Tile 4: Outlier & Anomaly Detection */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-amber-100 via-orange-50 to-amber-50 border-2 border-amber-300 p-5 sm:p-7 shadow-md flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-amber-600 text-white flex items-center justify-center shadow-md shadow-amber-600/20">
                  <Activity className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-black text-base sm:text-lg text-amber-950">Outlier & Anomaly Detection</h3>
                  <p className="text-[11px] sm:text-xs text-amber-800 mt-0.5 sm:mt-1 font-semibold">Automatic quality and consistency scoring</p>
                </div>

                <p className="text-xs sm:text-sm text-amber-950 leading-relaxed font-sans font-medium">
                  Automatically spots extreme outliers, unexpected data spikes, and conflicting fields before they skew your calculations.
                </p>

                <div className="grid grid-cols-2 gap-2 text-xs font-hero">
                  <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-amber-200 text-center shadow-xs">
                    <span className="text-[10px] text-amber-700 block font-medium">Data Balance</span>
                    <span className="font-bold text-amber-950 text-xs sm:text-sm mt-0.5 block">Healthy & Normal</span>
                  </div>
                  <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-amber-200 text-center shadow-xs">
                    <span className="text-[10px] text-amber-700 block font-medium">Conflict Check</span>
                    <span className="font-bold text-emerald-800 text-xs sm:text-sm mt-0.5 block">0 Conflicts</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-amber-200 text-[10px] sm:text-[11px] font-hero text-amber-900 font-bold">
                Automated Error Detection
              </div>
            </div>

            {/* Tile 5: Formula Protection & Safety Guard */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-teal-100 via-emerald-50 to-cyan-100/90 border-2 border-teal-300 p-5 sm:p-7 shadow-md flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-teal-600 text-white flex items-center justify-center shadow-md shadow-teal-600/20">
                  <Cpu className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-black text-base sm:text-lg text-teal-950">Formula Protection Guard</h3>
                  <p className="text-[11px] sm:text-xs text-teal-800 mt-0.5 sm:mt-1 font-semibold">Zero broken #REF! or formula corruptions</p>
                </div>

                <p className="text-xs sm:text-sm text-teal-950 leading-relaxed font-sans font-medium">
                  Protects calculation trees and formulas during cell cleanups so your team never inherits broken workbook sheets.
                </p>

                <div className="grid grid-cols-2 gap-2 text-xs font-hero">
                  <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-teal-200 text-center shadow-xs">
                    <span className="text-[10px] text-teal-700 block font-medium">Formula Trees</span>
                    <span className="font-bold text-teal-950 text-xs sm:text-sm mt-0.5 block">100% Protected</span>
                  </div>
                  <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-teal-200 text-center shadow-xs">
                    <span className="text-[10px] text-teal-700 block font-medium">Syntax Validation</span>
                    <span className="font-bold text-emerald-800 text-xs sm:text-sm mt-0.5 block">Zero Errors</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-teal-200 text-[10px] sm:text-[11px] font-hero text-teal-900 font-bold">
                Safe Calculation Architecture
              </div>
            </div>

            {/* Tile 6: Export in Any Format */}
            <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-emerald-100 via-teal-50 to-emerald-50 border-2 border-emerald-300 p-5 sm:p-7 shadow-md flex flex-col justify-between">
              <div className="space-y-4">
                <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-emerald-600 text-white flex items-center justify-center shadow-md shadow-emerald-600/20">
                  <FileSpreadsheet className="h-5 w-5" />
                </div>

                <div>
                  <h3 className="font-display font-black text-base sm:text-lg text-emerald-950">Export in Any Format</h3>
                  <p className="text-[11px] sm:text-xs text-emerald-700 mt-0.5 sm:mt-1 font-semibold">Excel, PDF, and automated validation</p>
                </div>

                <p className="text-xs sm:text-sm text-emerald-950 leading-relaxed font-sans font-medium">
                  Download cleaned spreadsheets ready for your team, alongside executive PDF briefs and automated pipeline quality rules.
                </p>

                <div className="p-2.5 sm:p-3 bg-white/95 border border-emerald-200 rounded-xl space-y-1.5 font-hero text-[11px] shadow-xs">
                  <div className="text-emerald-800 font-semibold flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-600 shrink-0 stroke-[2.5]" />
                    <span>Clean Excel Spreadsheet (.xlsx)</span>
                  </div>
                  <div className="text-emerald-800 font-semibold flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-600 shrink-0 stroke-[2.5]" />
                    <span>Executive Summary PDF (.pdf)</span>
                  </div>
                  <div className="text-emerald-700 font-medium flex items-center gap-1.5">
                    <Check className="h-3.5 w-3.5 text-emerald-600 shrink-0 stroke-[2.5]" />
                    <span>Automated Pipeline Rules (.json)</span>
                  </div>
                </div>
              </div>

              <div className="pt-5 sm:pt-6 mt-4 border-t border-emerald-200 text-[10px] sm:text-[11px] font-hero text-emerald-900 font-bold">
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
