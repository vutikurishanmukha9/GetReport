import { useState } from "react";
import { 
  ArrowRight, ShieldCheck, Terminal, Grid
} from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

interface SampleDataset {
  id: string;
  name: string;
  category: string;
  rows: string;
  columns: number;
  grade: string;
  confidence: number;
  gradeColor: string;
  issuesSummary: string;
  issuesCount: number;
  sampleSql: string;
  sqlColumns: string[];
  sqlRows: string[][];
  correlations: { pair: string; r: number; type: "pos" | "neg" }[];
}

const SAMPLE_DATASETS: SampleDataset[] = [
  {
    id: "fintech",
    name: "Sales & Payment Transactions",
    category: "Online Sales",
    rows: "14,200",
    columns: 18,
    grade: "B+",
    confidence: 87.4,
    gradeColor: "bg-violet-50 text-violet-700 border-violet-200",
    issuesSummary: "14 duplicate transaction IDs • 12 negative amount errors • 3 conflicting columns flagged",
    issuesCount: 3,
    sampleSql: "Grouping by category: Sales Subscriptions, Cloud Infrastructure, Processing, Licenses",
    sqlColumns: ["Category", "Transactions", "Avg Amount"],
    sqlRows: [
      ["SaaS Subscriptions", "4,820", "$42.50"],
      ["Cloud Infrastructure", "3,110", "$890.12"],
      ["Payment Processing", "2,940", "$14.20"],
      ["Enterprise Licenses", "1,880", "$2,450.00"],
    ],
    correlations: [
      { pair: "Order Amount ↔ Transaction Fee", r: 0.94, type: "pos" },
      { pair: "Risk Score ↔ Dispute Chance", r: 0.82, type: "pos" },
      { pair: "Account Age ↔ Refund Rate", r: -0.68, type: "neg" },
    ],
  },
  {
    id: "clinical",
    name: "Patient Health Survey",
    category: "Healthcare Cohort",
    rows: "6,850",
    columns: 24,
    grade: "A",
    confidence: 96.2,
    gradeColor: "bg-emerald-50 text-emerald-700 border-emerald-200",
    issuesSummary: "0 date errors • 100% complete records • Zero missing patient values",
    issuesCount: 0,
    sampleSql: "Grouping by treatment group: Active 50mg, Active 100mg, Control group",
    sqlColumns: ["Treatment Group", "Patients", "Recovery Rate"],
    sqlRows: [
      ["Active Program (50mg)", "2,280", "+34.2%"],
      ["Active Program (100mg)", "2,290", "+58.7%"],
      ["Control Group", "2,280", "-1.4%"],
    ],
    correlations: [
      { pair: "Dosage Level ↔ Improvement", r: 0.89, type: "pos" },
      { pair: "Attendance Rate ↔ Outcome", r: 0.78, type: "pos" },
      { pair: "Initial Severity ↔ Recovery Days", r: -0.54, type: "neg" },
    ],
  },
  {
    id: "saas",
    name: "Customer Subscriptions & Revenue",
    category: "Subscription Accounts",
    rows: "9,400",
    columns: 16,
    grade: "C",
    confidence: 68.5,
    gradeColor: "bg-amber-50 text-amber-700 border-amber-200",
    issuesSummary: "Negative revenue values caught • 4 broken date formats • 18 blank renewal dates flagged",
    issuesCount: 4,
    sampleSql: "Grouping by plan tier: Enterprise Annual, Growth Monthly, Starter Tier",
    sqlColumns: ["Plan Tier", "Total Revenue", "Seat Usage"],
    sqlRows: [
      ["Enterprise Annual", "$4,820,000", "88.4%"],
      ["Growth Monthly", "$1,450,000", "72.1%"],
      ["Starter Tier", "$320,000", "44.0%"],
    ],
    correlations: [
      { pair: "Seat Usage ↔ Renewal Rate", r: 0.86, type: "pos" },
      { pair: "Support Tickets ↔ Cancellation Risk", r: 0.74, type: "pos" },
      { pair: "Contract Length ↔ Discount Given", r: -0.61, type: "neg" },
    ],
  },
];

export const HeroSection = () => {
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>("fintech");
  const [activeViewMode, setActiveViewMode] = useState<"seal" | "matrix" | "sql">("seal");

  const activeDataset = SAMPLE_DATASETS.find(d => d.id === selectedDatasetId) || SAMPLE_DATASETS[0];

  return (
    <section className="relative pt-20 sm:pt-24 pb-8 sm:pb-12 bg-mesh-hero text-slate-900 border-b border-slate-200/90 overflow-hidden">
      {/* Precision micro-dot grid */}
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(#94A3B8_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_20%,#000_60%,transparent_100%)] opacity-35 pointer-events-none" />
      
      {/* Multi-color dynamic animated ambient glowing auras with color transitions */}
      <div className="absolute top-4 left-1/4 -translate-x-1/2 w-[600px] h-[500px] bg-gradient-to-tr from-violet-500/25 via-indigo-500/20 to-purple-500/15 rounded-full blur-3xl pointer-events-none -z-10 animate-aurora-1" />
      <div className="absolute top-10 right-1/4 translate-x-1/2 w-[560px] h-[480px] bg-gradient-to-bl from-sky-400/25 via-teal-400/15 to-emerald-400/15 rounded-full blur-3xl pointer-events-none -z-10 animate-aurora-2" />
      <div className="absolute -top-12 left-1/2 -translate-x-1/2 w-[520px] h-[420px] bg-gradient-to-b from-rose-400/15 via-purple-400/15 to-transparent rounded-full blur-3xl pointer-events-none -z-10 animate-aurora-3" />
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[750px] h-[300px] bg-gradient-to-t from-emerald-400/15 via-sky-400/15 to-transparent rounded-full blur-3xl pointer-events-none -z-10 animate-aurora-4" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-8 xl:gap-12 items-center">
          
          {/* Left Column: Editorial Value Proposition */}
          <div className="lg:col-span-7 space-y-5 sm:space-y-6 text-left">

            {/* Display Headline */}
            <h1 className="text-2xl xs:text-3xl sm:text-4xl lg:text-[2.5rem] xl:text-[2.85rem] font-hero font-extrabold tracking-[-0.03em] text-slate-900 leading-[1.16] sm:leading-[1.12] text-balance">
              <span className="block lg:whitespace-nowrap">
                The data audit platform built for
              </span>
              <span className="block mt-1 sm:mt-1.5 bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent lg:whitespace-nowrap">
                when numbers actually matter.
              </span>
            </h1>

            {/* Subheadline */}
            <p className="text-sm sm:text-base lg:text-lg text-slate-600 leading-relaxed max-w-xl font-hero font-normal">
              Catch hidden spreadsheet errors, fix bad data with one click, and generate professional PDF reports before sharing with your team or clients.
            </p>

            {/* Primary & Secondary Nested CTAs with Non-Blue Royal Iris Button */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-1 sm:pt-2">
              <Link to="/workspace" className="w-full sm:w-auto">
                <button
                  type="button"
                  className="w-full sm:w-auto h-12 px-7 rounded-xl bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-hero font-bold text-sm tracking-tight shadow-[0_4px_16px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] border border-violet-400/40 ring-1 ring-white/20 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2.5 cursor-pointer group"
                >
                  <span>Start Checking Your Data Free</span>
                  <ArrowRight className="h-4 w-4 text-white/90 transition-transform duration-200 group-hover:translate-x-1" />
                </button>
              </Link>

              <Link to="/how-it-works" className="w-full sm:w-auto">
                <Button
                  size="lg"
                  variant="outline"
                  className="w-full sm:w-auto h-12 px-6 rounded-xl border-slate-300/90 bg-white/90 hover:bg-white text-slate-800 hover:text-violet-900 shadow-sm hover:border-violet-300 hover:shadow-md transition-all duration-150 hover:-translate-y-0.5 active:scale-95 font-hero font-semibold text-sm"
                >
                  <span>See How It Works</span>
                </Button>
              </Link>
            </div>

            {/* Technical Proof Strip */}
            <div className="pt-4 border-t border-slate-200/90 grid grid-cols-3 gap-2 sm:gap-3 max-w-lg text-[10px] sm:text-[11px] font-mono">
              <div className="p-2 sm:p-2.5 rounded-xl bg-white/75 backdrop-blur-xs border border-slate-200/80 shadow-2xs space-y-0.5">
                <span className="block font-bold text-slate-900 text-xs sm:text-sm">&lt; 2 Seconds</span>
                <span className="leading-tight block text-slate-500 text-[10px]">Instant Scan</span>
              </div>
              <div className="p-2 sm:p-2.5 rounded-xl bg-white/75 backdrop-blur-xs border border-slate-200/80 shadow-2xs space-y-0.5">
                <span className="block font-bold text-slate-900 text-xs sm:text-sm">100% Private</span>
                <span className="leading-tight block text-slate-500 text-[10px]">Never Stored</span>
              </div>
              <div className="p-2 sm:p-2.5 rounded-xl bg-white/75 backdrop-blur-xs border border-slate-200/80 shadow-2xs space-y-0.5">
                <span className="block font-bold text-slate-900 text-xs sm:text-sm">Safe Export</span>
                <span className="leading-tight block text-slate-500 text-[10px]">Formula Protected</span>
              </div>
            </div>

          </div>

          {/* Right Column: Interactive Live Audit Studio (The "Working Artifact") */}
          <div className="lg:col-span-5 w-full">
            <div className="rounded-[1.75rem] sm:rounded-[2.25rem] bg-gradient-to-b from-white/95 via-slate-50/90 to-indigo-50/30 p-1.5 sm:p-2.5 ring-1 ring-slate-200/90 shadow-[0_20px_50px_-10px_rgba(30,58,138,0.12)] backdrop-blur-xl">
              <div className="rounded-[calc(1.75rem-0.375rem)] sm:rounded-[calc(2.25rem-0.5rem)] bg-white/95 border border-slate-200/80 p-4 sm:p-6 shadow-sm space-y-4 sm:space-y-5 text-slate-800">
                
                {/* Header: Sample Dataset Switcher Pills */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 sm:gap-3 border-b border-slate-100 pb-3 sm:pb-4">
                  <div>
                    <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 block">
                      Interactive Demo • Choose an Example:
                    </span>
                    <span className="text-xs sm:text-sm font-bold text-slate-900 font-hero">
                      {activeDataset.name}
                    </span>
                  </div>

                  {/* 3 Dataset Selector Buttons */}
                  <div className="grid grid-cols-3 sm:flex items-center gap-1 bg-slate-100/90 p-1 rounded-xl border border-slate-200 w-full sm:w-auto">
                    {SAMPLE_DATASETS.map((d) => (
                      <button
                        key={d.id}
                        type="button"
                        onClick={() => setSelectedDatasetId(d.id)}
                        className={`h-9 sm:h-7.5 px-3 rounded-lg text-xs sm:text-[11px] font-mono font-semibold transition-all active:scale-95 touch-manipulation flex items-center justify-center ${
                          selectedDatasetId === d.id
                            ? "bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 text-white shadow-xs"
                            : "text-slate-600 hover:text-slate-900 hover:bg-white/50"
                        }`}
                      >
                        {d.id === "fintech" ? "Sales" : d.id === "clinical" ? "Healthcare" : "Customers"}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Sub-view Mode Switcher */}
                <div className="grid grid-cols-3 gap-1 bg-slate-100/90 p-1 rounded-xl border border-slate-200 text-xs font-mono">
                  <button
                    type="button"
                    onClick={() => setActiveViewMode("seal")}
                    className={`h-10 sm:h-8.5 rounded-lg text-center font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[11px] touch-manipulation ${
                      activeViewMode === "seal" ? "bg-white text-slate-900 border border-slate-200 shadow-xs" : "text-slate-500 hover:text-slate-900"
                    }`}
                  >
                    <ShieldCheck className="h-3.5 w-3.5 text-blue-600 shrink-0" />
                    <span>Quality Score</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveViewMode("matrix")}
                    className={`h-10 sm:h-8.5 rounded-lg text-center font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[11px] touch-manipulation ${
                      activeViewMode === "matrix" ? "bg-white text-slate-900 border border-slate-200 shadow-xs" : "text-slate-500 hover:text-slate-900"
                    }`}
                  >
                    <Grid className="h-3.5 w-3.5 text-indigo-600 shrink-0" />
                    <span>Relationships</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveViewMode("sql")}
                    className={`h-10 sm:h-8.5 rounded-lg text-center font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[11px] touch-manipulation ${
                      activeViewMode === "sql" ? "bg-white text-slate-900 border border-slate-200 shadow-xs" : "text-slate-500 hover:text-slate-900"
                    }`}
                  >
                    <Terminal className="h-3.5 w-3.5 text-violet-600 shrink-0" />
                    <span>Data Table</span>
                  </button>
                </div>

                {/* VIEW 1: SEAL & HEALTH SCORECARD */}
                {activeViewMode === "seal" && (
                  <div className="space-y-3 sm:space-y-4 animate-in fade-in duration-200">
                    <div className={`flex items-center gap-3.5 sm:gap-5 p-3.5 sm:p-4 rounded-xl sm:rounded-2xl border-2 transition-all ${
                      activeDataset.id === "fintech" 
                        ? "bg-gradient-to-br from-blue-100/90 via-sky-50 to-indigo-100/70 border-blue-300 shadow-sm text-blue-950" 
                        : activeDataset.id === "clinical" 
                        ? "bg-gradient-to-br from-emerald-100/90 via-teal-50 to-emerald-50 border-emerald-300 shadow-sm text-emerald-950" 
                        : "bg-gradient-to-br from-amber-100/90 via-orange-50 to-amber-50 border-amber-300 shadow-sm text-amber-950"
                    }`}>
                      {/* Concentric Grade Seal */}
                      <div className={`flex flex-col items-center justify-center h-16 w-16 sm:h-20 sm:w-20 rounded-xl sm:rounded-2xl ${activeDataset.gradeColor} border-2 shadow-sm shrink-0 bg-white`}>
                        <span className="text-[8px] sm:text-[9px] font-mono uppercase font-bold opacity-80">GRADE</span>
                        <span className="text-2xl sm:text-3xl font-hero font-extrabold tracking-tight leading-none my-0.5">
                          {activeDataset.grade}
                        </span>
                        <span className="text-[9px] sm:text-[10px] font-mono font-bold opacity-90">{activeDataset.confidence}%</span>
                      </div>

                      <div className="space-y-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <h4 className="font-hero font-bold text-xs sm:text-sm text-slate-900 truncate">
                            {activeDataset.category}
                          </h4>
                          <Badge variant="outline" className="text-[9px] font-mono px-1.5 py-0 border-slate-300 bg-white font-bold text-slate-800 shadow-2xs">
                            {activeDataset.rows} rows
                          </Badge>
                        </div>
                        <p className="text-[10px] sm:text-[11px] font-sans text-slate-700 line-clamp-2 font-medium">
                          {activeDataset.issuesSummary}
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-1.5 sm:gap-2.5 text-center font-mono text-[10px] sm:text-[11px]">
                      <div className="p-2 sm:p-2.5 rounded-xl border border-violet-200 bg-violet-50/90 shadow-xs">
                        <span className="text-[8.5px] sm:text-[9px] text-violet-700 block uppercase font-bold">Total Columns</span>
                        <span className="font-bold text-violet-950 text-xs sm:text-[11px]">{activeDataset.columns}</span>
                      </div>
                      <div className="p-2 sm:p-2.5 rounded-xl border border-rose-200 bg-rose-50/90 shadow-xs">
                        <span className="text-[8.5px] sm:text-[9px] text-rose-700 block uppercase font-bold">Issues Found</span>
                        <span className="font-bold text-rose-950 text-xs sm:text-[11px]">{activeDataset.issuesCount}</span>
                      </div>
                      <div className="p-2 sm:p-2.5 rounded-xl border border-emerald-200 bg-emerald-50/90 shadow-xs">
                        <span className="text-[8.5px] sm:text-[9px] text-emerald-700 block uppercase font-bold">Export Format</span>
                        <span className="font-bold text-emerald-950 text-xs sm:text-[11px]">Excel & PDF</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* VIEW 2: CORRELATION MATRIX PREVIEW */}
                {activeViewMode === "matrix" && (
                  <div className="space-y-3 font-mono text-xs animate-in fade-in duration-200">
                    <div className="flex items-center justify-between text-[10px] text-slate-500 uppercase pb-1 border-b border-slate-200">
                      <span>Compared Fields</span>
                      <span>Relationship Strength</span>
                    </div>
                    {activeDataset.correlations.map((c) => (
                      <div key={c.pair} className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                        <span className="font-semibold text-slate-800 text-[11px] truncate max-w-[200px]">{c.pair}</span>
                        <Badge
                          variant="secondary"
                          className={`text-[10px] font-bold ${
                            c.type === "pos" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"
                          }`}
                        >
                          {c.type === "pos" ? "+" : ""}{(c.r * 100).toFixed(0)}% Match
                        </Badge>
                      </div>
                    ))}
                    <div className="text-[10px] text-slate-500 text-center pt-1 font-sans">
                      Shows which numbers move together so you can spot patterns easily
                    </div>
                  </div>
                )}

                {/* VIEW 3: DATA TABLE PREVIEW */}
                {activeViewMode === "sql" && (
                  <div className="space-y-3 animate-in fade-in duration-200 font-mono text-xs">
                    <div className="bg-slate-950 rounded-xl p-3 text-emerald-400 text-[11px] leading-relaxed border border-slate-800 overflow-x-auto">
                      <span className="text-slate-400 block text-[9px] uppercase font-bold mb-1">Instant Data Summary</span>
                      {activeDataset.sampleSql}
                    </div>

                    <div className="flex items-center justify-between text-[10.5px] font-mono text-slate-500 px-0.5">
                      <span>Spreadsheet Preview</span>
                      <span className="sm:hidden text-violet-600 font-semibold flex items-center gap-1">
                        Swipe sideways →
                      </span>
                    </div>

                    <div className="border border-slate-200 rounded-xl overflow-x-auto shadow-2xs">
                      <table className="w-full text-left text-[11px] min-w-[320px]">
                        <thead className="bg-slate-100 text-slate-700 font-semibold border-b border-slate-200">
                          <tr>
                            {activeDataset.sqlColumns.map((col) => (
                              <th key={col} className="px-3 py-1.5">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 bg-white">
                          {activeDataset.sqlRows.map((row, rIdx) => (
                            <tr key={`hero-row-${rIdx}`} className="hover:bg-slate-50">
                              {row.map((cell, cIdx) => (
                                <td key={`hero-cell-${rIdx}-${cIdx}`} className="px-3 py-1.5 text-slate-700">
                                  {cell}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Bottom Card Action */}
                <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-xs">
                  <span className="text-[11px] font-mono text-slate-500">
                    Try this with your own spreadsheet
                  </span>
                  <Link to="/workspace" className="inline-flex items-center gap-1 font-mono font-bold text-violet-600 hover:text-violet-700 hover:underline text-xs">
                    <span>Open Free Workspace</span>
                    <ArrowRight className="h-3 w-3" />
                  </Link>
                </div>

              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default HeroSection;
