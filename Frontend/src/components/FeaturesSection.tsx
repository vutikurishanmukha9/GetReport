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
  Check
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

        {/* Bento Grid with Separate Distinct Card Colors */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 sm:gap-6 lg:gap-8">
          
          {/* Tile 1: Interactive Issue Ledger (Distinct Rich Lavender/Purple Card) */}
          <div className="lg:col-span-2 rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-violet-100 via-purple-50 to-indigo-100/90 border-2 border-violet-300 p-5 sm:p-8 shadow-lg flex flex-col justify-between">
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
                  <Button
                    variant="saveAll"
                    size="sm"
                    onClick={handleSaveAll}
                    className="h-9 px-3.5 rounded-xl gap-1.5 shadow-sm text-xs cursor-pointer font-bold"
                  >
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Save All</span>
                  </Button>
                  <Button
                    variant="delete"
                    size="sm"
                    onClick={handleDiscardAll}
                    className="h-9 px-3.5 rounded-xl gap-1.5 text-xs cursor-pointer font-bold"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Discard All</span>
                  </Button>
                </div>
              </div>

              {/* Interactive Ledger Rows */}
              <div className="space-y-2.5 pt-1">
                {ledger.map((item) => (
                  <div
                    key={item.id}
                    className={`p-3 sm:p-3.5 rounded-xl border transition-all duration-150 flex flex-col xs:flex-row xs:items-center justify-between gap-2.5 sm:gap-3 shadow-xs select-none ${
                      item.approved
                        ? "bg-white/95 border-emerald-300 shadow-sm"
                        : "bg-white/70 border-slate-300 opacity-80"
                    }`}
                  >
                    <div className="space-y-0.5 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-mono text-xs font-bold text-slate-900">{item.column}</span>
                        <Badge variant="outline" className="text-[9.5px] sm:text-[10px] font-mono px-1.5 py-0 bg-violet-50 border-violet-200 text-violet-900 font-semibold">
                          {item.issue}
                        </Badge>
                      </div>
                      <div className="text-[10.5px] sm:text-[11px] font-mono text-slate-700">
                        Suggested Action: <strong className="text-slate-950">{item.fix}</strong>
                      </div>
                    </div>

                    {/* Dedicated Save Fix and Delete Action Buttons */}
                    <div className="flex items-center gap-2 self-end xs:self-auto shrink-0">
                      <button
                        type="button"
                        onClick={() => handleSetItemStatus(item.id, true)}
                        className={`h-8.5 px-3 rounded-xl font-mono text-[11px] font-bold flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer ${
                          item.approved
                            ? "bg-emerald-600 text-white shadow-sm shadow-emerald-600/30 border border-emerald-500/80 ring-1 ring-white/20"
                            : "bg-white text-slate-600 hover:text-emerald-700 hover:bg-emerald-50 border border-slate-300 shadow-2xs"
                        }`}
                        title="Save & apply this fix"
                      >
                        <CheckCircle2 className={`h-3.5 w-3.5 ${item.approved ? "text-white" : "text-emerald-600"}`} />
                        <span>Save Fix</span>
                      </button>

                      <button
                        type="button"
                        onClick={() => handleSetItemStatus(item.id, false)}
                        className={`h-8.5 px-3 rounded-xl font-mono text-[11px] font-bold flex items-center gap-1.5 transition-all duration-150 active:scale-95 cursor-pointer ${
                          !item.approved
                            ? "bg-rose-600 text-white shadow-sm shadow-rose-600/30 border border-rose-500/80 ring-1 ring-white/20"
                            : "bg-white text-slate-500 hover:text-rose-700 hover:bg-rose-50 border border-slate-300 shadow-2xs"
                        }`}
                        title="Delete or skip this fix"
                      >
                        <Trash2 className={`h-3.5 w-3.5 ${!item.approved ? "text-white" : "text-rose-600"}`} />
                        <span>Delete</span>
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-6 pt-3.5 sm:pt-4 border-t border-violet-200 text-[10px] sm:text-[11px] font-mono text-violet-900 flex flex-col xs:flex-row gap-1.5 xs:items-center justify-between">
              <span>{approvedCount} of {ledger.length} fixes approved</span>
              <span className="text-violet-700 font-bold w-fit">Every change is clearly documented</span>
            </div>
          </div>

          {/* Tile 2: Instant Data Search & Filters */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-blue-100 via-indigo-50 to-sky-100/90 border-2 border-blue-300 p-5 sm:p-8 shadow-md flex flex-col justify-between">
            <div className="space-y-4">
              <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-600/20">
                <Terminal className="h-5 w-5" />
              </div>

              <div>
                <h3 className="font-display font-black text-base sm:text-lg text-blue-950">Instant Filters & Search</h3>
                <p className="text-[11px] sm:text-xs text-blue-700 mt-0.5 sm:mt-1 font-semibold">Filter and slice any column in milliseconds</p>
              </div>

              <p className="text-xs sm:text-sm text-blue-950 leading-relaxed font-sans font-medium">
                Quickly filter, sort, and inspect specific customer tiers or revenue categories without waiting or crashing your computer.
              </p>

              <div className="p-3 bg-white/95 rounded-xl border border-blue-200 space-y-1 shadow-xs font-mono text-xs overflow-x-auto">
                <span className="text-blue-600 block text-[9.5px] uppercase font-bold">Instant Filter Summary</span>
                <div className="text-slate-900 font-semibold">Active Filter: High-Value Customer Accounts</div>
                <div className="text-slate-900 font-semibold">Sorted by: Highest Revenue First</div>
                <div className="text-emerald-700 text-[10px] pt-1 font-bold">✓ 500 rows verified in under 1 second</div>
              </div>
            </div>

            <div className="pt-5 sm:pt-6 mt-4 border-t border-blue-200 text-[10px] sm:text-[11px] font-mono text-blue-900 font-bold">
              Instant Local Processing
            </div>
          </div>

          {/* Tile 3: Multi-Column Visual Discovery */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-sky-100 via-cyan-50 to-blue-100/90 border-2 border-sky-300 p-5 sm:p-8 shadow-md flex flex-col justify-between">
            <div className="space-y-4">
              <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-sky-600 text-white flex items-center justify-center shadow-md shadow-sky-600/20">
                <Sliders className="h-5 w-5" />
              </div>

              <div>
                <h3 className="font-display font-black text-base sm:text-lg text-sky-950">Multi-Column Visual Discovery</h3>
                <p className="text-[11px] sm:text-xs text-sky-700 mt-0.5 sm:mt-1 font-semibold">Compare multiple metrics on a single screen</p>
              </div>

              <p className="text-xs sm:text-sm text-sky-950 leading-relaxed font-sans font-medium">
                Interactive visual charts let you easily slide across dates, revenue, scores, and categories to spot hidden patterns.
              </p>

              <div className="h-16 sm:h-20 rounded-xl bg-white/95 border border-sky-200 flex items-center justify-center text-sky-900 font-mono text-[11px] font-bold shadow-xs">
                Interactive Visual Trend Explorer
              </div>
            </div>

            <div className="pt-5 sm:pt-6 mt-4 border-t border-sky-200 text-[10px] sm:text-[11px] font-mono text-sky-800 font-semibold">
              Interactive Visual Charts
            </div>
          </div>

          {/* Tile 4: Outlier & Anomaly Detection */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-amber-100 via-orange-50 to-amber-50 border-2 border-amber-300 p-5 sm:p-8 shadow-md flex flex-col justify-between">
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

              <div className="grid grid-cols-2 gap-2 font-mono text-xs">
                <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-amber-200 text-center shadow-xs">
                  <span className="text-[9.5px] sm:text-[10px] text-amber-700 block font-semibold">Data Balance</span>
                  <span className="font-bold text-amber-950 text-xs sm:text-sm">Healthy & Normal</span>
                </div>
                <div className="p-2 sm:p-2.5 rounded-xl bg-white/95 border border-amber-200 text-center shadow-xs">
                  <span className="text-[9.5px] sm:text-[10px] text-amber-700 block font-semibold">Conflict Check</span>
                  <span className="font-bold text-emerald-800 text-xs sm:text-sm">0 Conflicts Found</span>
                </div>
              </div>
            </div>

            <div className="pt-5 sm:pt-6 mt-4 border-t border-amber-200 text-[10px] sm:text-[11px] font-mono text-amber-900 font-semibold">
              Automated Error Detection
            </div>
          </div>

          {/* Tile 5: Export in Any Format */}
          <div className="rounded-[1.75rem] sm:rounded-[2rem] bg-gradient-to-br from-emerald-100 via-teal-50 to-emerald-50 border-2 border-emerald-300 p-5 sm:p-8 shadow-md flex flex-col justify-between">
            <div className="space-y-4">
              <div className="h-9 w-9 sm:h-10 sm:w-10 rounded-xl sm:rounded-2xl bg-emerald-600 text-white flex items-center justify-center shadow-md shadow-emerald-600/20">
                <ShieldCheck className="h-5 w-5" />
              </div>

              <div>
                <h3 className="font-display font-black text-base sm:text-lg text-emerald-950">Export in Any Format</h3>
                <p className="text-[11px] sm:text-xs text-emerald-700 mt-0.5 sm:mt-1 font-semibold">Excel, PDF, and automated validation</p>
              </div>

              <p className="text-xs sm:text-sm text-emerald-950 leading-relaxed font-sans font-medium">
                Download cleaned spreadsheets ready for your team, alongside executive PDF briefs and automated pipeline quality rules.
              </p>

              <div className="p-2.5 sm:p-3 bg-white/95 border border-emerald-200 rounded-xl space-y-1 font-mono text-[10.5px] sm:text-[11px] shadow-xs">
                <div className="text-emerald-800 font-bold">✓ Clean Excel Spreadsheet (.xlsx)</div>
                <div className="text-emerald-800 font-bold">✓ Executive Summary PDF (.pdf)</div>
                <div className="text-emerald-700">✓ Automated Pipeline Rules (.json)</div>
              </div>
            </div>

            <div className="pt-5 sm:pt-6 mt-4 border-t border-emerald-200 text-[10px] sm:text-[11px] font-mono text-emerald-900 font-bold">
              Ready for Teams & Executives
            </div>
          </div>

        </div>

      </div>
    </section>
  );
};

export default FeaturesSection;
