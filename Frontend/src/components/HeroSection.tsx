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
    <section className="relative pt-28 sm:pt-28 lg:pt-28 pb-14 sm:pb-16 lg:pb-20 bg-hero-custom-gradient border-b border-slate-200/80 overflow-hidden">
      {/* 1. Subtle Ethereal Micro-Dot Grid */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.22)_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_25%,#000_65%,transparent_100%)] opacity-35 pointer-events-none" />

      {/* 2. Film Grain / Tactile Noise Texture Filter (opacity: .08 overlay to soften gradient) */}
      <div className="hero-grain-overlay" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-6 xl:px-8 max-w-7xl relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-6 xl:gap-10 items-start lg:pt-1">
          
          {/* Left Column: Value Proposition & High-Contrast Typography */}
          <div className="lg:col-span-7 space-y-6 sm:space-y-7 text-left">

            {/* Display Headline - Crisp white on deepened orange bowl with soft text-shadow */}
            <h1 
              className="font-hero font-extrabold tracking-[-0.03em] text-white leading-[1.14] sm:leading-[1.1]"
              style={{ 
                fontSize: 'clamp(1.25rem, 3.2vw, 2.65rem)',
                textShadow: '0 1px 18px rgba(120, 40, 0, 0.25)' 
              }}
            >
              <span className="block text-white whitespace-nowrap">
                The data audit platform built for
              </span>
              <span className="block mt-1 sm:mt-1.5 text-white whitespace-nowrap">
                when numbers actually matter.
              </span>
            </h1>

            {/* Subheadline - High-contrast white text covering the extended orange bowl */}
            <p 
              className="text-base sm:text-lg lg:text-[1.125rem] text-white/95 leading-relaxed max-w-xl font-hero font-medium [text-wrap:pretty]"
              style={{ textShadow: '0 1px 10px rgba(120, 40, 0, 0.20)' }}
            >
              Catch hidden spreadsheet errors, fix bad data with one click, and generate professional PDF reports before sharing with your team or clients.
            </p>

            {/* Primary & Secondary Nested CTAs */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-1">
              <Link to="/workspace" className="w-full sm:w-auto">
                <button
                  type="button"
                  className="w-full sm:w-auto h-12 px-7 rounded-xl bg-gradient-to-r from-[#4338ca] via-[#3730a3] to-[#312e81] hover:from-[#3730a3] hover:to-[#312e81] text-white font-hero font-bold text-sm tracking-tight shadow-[0_4px_20px_-2px_rgba(67,56,202,0.45)] border border-indigo-400/30 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2.5 cursor-pointer group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 focus-visible:ring-offset-2"
                >
                  <span>Start Checking Your Data Free</span>
                  <ArrowRight className="h-4 w-4 text-indigo-200 transition-transform duration-200 group-hover:translate-x-1" />
                </button>
              </Link>

              <Link to="/how-it-works" className="w-full sm:w-auto">
                <button
                  type="button"
                  className="w-full sm:w-auto h-12 px-6 rounded-xl bg-white hover:bg-slate-50 text-[#1d1d2b] font-hero font-bold text-sm shadow-sm border border-black/10 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2"
                >
                  <span>See How It Works</span>
                </button>
              </Link>
            </div>

            {/* Technical Proof Strip - Clean Frosted Cards with single typography family on periwinkle base */}
            <div className="pt-5 mt-2 border-t border-white/45 grid grid-cols-3 gap-2.5 sm:gap-3 max-w-lg">
              <div className="p-3 sm:p-3.5 rounded-xl bg-white/90 hover:bg-white backdrop-blur-md border border-white/70 shadow-xs space-y-1 transition-all">
                <span className="block font-hero font-extrabold text-slate-900 text-sm sm:text-base leading-tight">&lt; 2 Seconds</span>
                <span className="leading-tight block text-slate-600 text-xs sm:text-[13px] font-hero font-medium">Instant Scan</span>
              </div>
              <div className="p-3 sm:p-3.5 rounded-xl bg-white/90 hover:bg-white backdrop-blur-md border border-white/70 shadow-xs space-y-1 transition-all">
                <span className="block font-hero font-extrabold text-slate-900 text-sm sm:text-base leading-tight">100% Private</span>
                <span className="leading-tight block text-slate-600 text-xs sm:text-[13px] font-hero font-medium">Never Stored</span>
              </div>
              <div className="p-3 sm:p-3.5 rounded-xl bg-white/90 hover:bg-white backdrop-blur-md border border-white/70 shadow-xs space-y-1 transition-all">
                <span className="block font-hero font-extrabold text-slate-900 text-sm sm:text-base leading-tight">Safe Export</span>
                <span className="leading-tight block text-slate-600 text-xs sm:text-[13px] font-hero font-medium">Formula Protected</span>
              </div>
            </div>

          </div>

          {/* Right Column: Clean Single-Border Interactive Demo Card */}
          <div className="lg:col-span-5 w-full">
            <div 
              className="rounded-3xl p-4 sm:p-6 shadow-sm space-y-4 sm:space-y-5 text-slate-800"
              style={{
                backgroundColor: 'rgba(255, 255, 255, 0.92)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border: '1px solid rgba(255, 255, 255, 0.85)',
                boxShadow: '0 24px 50px -15px rgba(249, 115, 22, 0.18), 0 12px 30px -10px rgba(188, 202, 248, 0.32)',
              }}
            >
              
              {/* Header: Sample Dataset Switcher */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 sm:gap-3 border-b border-slate-100 pb-3 sm:pb-4">
                <div>
                  <span className="text-[13px] font-hero font-semibold text-slate-700 block leading-tight">
                    Try an example
                  </span>
                  <span className="text-sm sm:text-base font-hero font-bold text-slate-900 block mt-0.5">
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
                      className={`h-9 sm:h-8 px-3 rounded-lg text-xs sm:text-[12px] font-hero font-semibold transition-all active:scale-95 touch-manipulation flex items-center justify-center focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 ${
                        selectedDatasetId === d.id
                          ? "bg-gradient-to-r from-[#4338ca] via-[#3730a3] to-[#312e81] text-white shadow-xs"
                          : "text-slate-600 hover:text-slate-900 hover:bg-white/60"
                      }`}
                    >
                      {d.id === "fintech" ? "Sales" : d.id === "clinical" ? "Healthcare" : "Customers"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sub-view Mode Switcher */}
              <div className="grid grid-cols-3 gap-1 bg-slate-100/90 p-1 rounded-xl border border-slate-200">
                <button
                  type="button"
                  onClick={() => setActiveViewMode("seal")}
                  className={`h-10 sm:h-8.5 rounded-lg text-center font-hero font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[12px] touch-manipulation focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 ${
                    activeViewMode === "seal" ? "bg-white text-slate-900 border border-slate-200 shadow-xs" : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  <ShieldCheck className="h-3.5 w-3.5 text-blue-600 shrink-0" />
                  <span>Quality Score</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveViewMode("matrix")}
                  className={`h-10 sm:h-8.5 rounded-lg text-center font-hero font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[12px] touch-manipulation focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 ${
                    activeViewMode === "matrix" ? "bg-white text-slate-900 border border-slate-200 shadow-xs" : "text-slate-500 hover:text-slate-900"
                  }`}
                >
                  <Grid className="h-3.5 w-3.5 text-indigo-600 shrink-0" />
                  <span>Relationships</span>
                </button>

                <button
                  type="button"
                  onClick={() => setActiveViewMode("sql")}
                  className={`h-10 sm:h-8.5 rounded-lg text-center font-hero font-semibold transition-all flex items-center justify-center gap-1.5 active:scale-95 text-xs sm:text-[12px] touch-manipulation focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-1 ${
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
                  <div className={`flex items-center gap-3.5 sm:gap-5 p-3.5 sm:p-4 rounded-xl sm:rounded-2xl border transition-all ${
                    activeDataset.id === "fintech" 
                      ? "bg-gradient-to-br from-[#7d5a9e]/12 via-purple-50/60 to-white border-[#7d5a9e]/30 shadow-xs text-purple-950" 
                      : activeDataset.id === "clinical" 
                      ? "bg-gradient-to-br from-[#6fb85a]/14 via-emerald-50/60 to-white border-[#6fb85a]/30 shadow-xs text-emerald-950" 
                      : "bg-gradient-to-br from-amber-500/12 via-amber-50/60 to-white border-amber-400/30 shadow-xs text-amber-950"
                  }`}>
                    {/* Concentric Grade Seal */}
                    <div className={`flex flex-col items-center justify-center h-16 w-16 sm:h-20 sm:w-20 rounded-xl sm:rounded-2xl ${
                      activeDataset.id === "fintech"
                        ? "bg-white text-[#4c2d6b] border border-[#7d5a9e]/30 shadow-xs"
                        : activeDataset.id === "clinical"
                        ? "bg-white text-[#245717] border border-[#6fb85a]/35 shadow-xs"
                        : "bg-white text-amber-800 border border-amber-300 shadow-xs"
                    } shrink-0`}>
                      <span className="text-[11px] font-hero font-semibold text-slate-500 leading-none">Grade</span>
                      <span className="text-2xl sm:text-3xl font-hero font-extrabold tracking-tight leading-none my-0.5">
                        {activeDataset.grade}
                      </span>
                      <span className="text-[10px] sm:text-[11px] font-mono font-bold opacity-85">{activeDataset.confidence}%</span>
                    </div>

                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <h4 className="font-hero font-bold text-xs sm:text-sm text-slate-900 truncate">
                          {activeDataset.category}
                        </h4>
                        <Badge variant="outline" className="text-[10px] font-hero px-1.5 py-0 border-slate-300 bg-white font-medium text-slate-800 shadow-2xs">
                          <span className="font-mono font-bold mr-1">{activeDataset.rows}</span> rows
                        </Badge>
                      </div>
                      <p className="text-[11px] sm:text-xs font-hero text-slate-700 line-clamp-2 font-normal leading-relaxed">
                        {activeDataset.issuesSummary}
                      </p>
                    </div>
                  </div>

                  {/* Summary Tiles: Harmonized light tints, colored labels, and consistent border strength */}
                  <div className="grid grid-cols-3 gap-2 sm:gap-2.5 text-center">
                    <div className="p-2 sm:p-2.5 rounded-xl border border-violet-200/90 bg-violet-50/90 shadow-2xs">
                      <span className="text-[11px] sm:text-[11.5px] text-violet-700 block font-hero font-semibold">Total columns</span>
                      <span className="font-mono font-bold text-violet-950 text-xs sm:text-sm mt-0.5 block">{activeDataset.columns}</span>
                    </div>
                    <div className="p-2 sm:p-2.5 rounded-xl border border-rose-200/90 bg-rose-50/90 shadow-2xs">
                      <span className="text-[11px] sm:text-[11.5px] text-rose-700 block font-hero font-semibold">Issues found</span>
                      <span className="font-mono font-bold text-rose-950 text-xs sm:text-sm mt-0.5 block">{activeDataset.issuesCount}</span>
                    </div>
                    <div className="p-2 sm:p-2.5 rounded-xl border border-emerald-200/90 bg-emerald-50/90 shadow-2xs">
                      <span className="text-[11px] sm:text-[11.5px] text-emerald-700 block font-hero font-semibold">Export format</span>
                      <span className="font-hero font-bold text-emerald-950 text-xs sm:text-sm mt-0.5 block">Excel & PDF</span>
                    </div>
                  </div>
                </div>
              )}

              {/* VIEW 2: CORRELATION MATRIX PREVIEW */}
              {activeViewMode === "matrix" && (
                <div className="space-y-3 text-xs animate-in fade-in duration-200">
                  <div className="flex items-center justify-between text-[11px] font-hero font-medium text-slate-500 pb-1 border-b border-slate-200">
                    <span>Compared fields</span>
                    <span>Relationship strength</span>
                  </div>
                  {activeDataset.correlations.map((c) => (
                    <div key={c.pair} className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 border border-slate-200">
                      <span className="font-hero font-semibold text-slate-800 text-[11.5px] truncate max-w-[200px]">{c.pair}</span>
                      <Badge
                        variant="secondary"
                        className={`text-[10px] font-bold ${
                          c.type === "pos" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-amber-50 text-amber-700 border-amber-200"
                        }`}
                      >
                        <span className="font-mono font-bold">{c.type === "pos" ? "+" : ""}{(c.r * 100).toFixed(0)}%</span>
                        <span className="ml-1 font-hero font-medium">Match</span>
                      </Badge>
                    </div>
                  ))}
                  <div className="text-[11px] text-slate-500 text-center pt-1 font-hero">
                    Shows which numbers move together so you can spot patterns easily
                  </div>
                </div>
              )}

              {/* VIEW 3: DATA TABLE PREVIEW */}
              {activeViewMode === "sql" && (
                <div className="space-y-3 animate-in fade-in duration-200 text-xs">
                  <div className="bg-slate-950 rounded-xl p-3 text-emerald-400 text-[11px] leading-relaxed border border-slate-800 overflow-x-auto">
                    <span className="text-slate-400 block text-[10px] font-hero font-semibold mb-1">Instant Data Summary</span>
                    <span className="font-mono">{activeDataset.sampleSql}</span>
                  </div>

                  <div className="flex items-center justify-between text-[11px] font-hero font-medium text-slate-500 px-0.5">
                    <span>Spreadsheet preview</span>
                    <span className="sm:hidden text-violet-600 font-semibold flex items-center gap-1 font-hero">
                      Swipe sideways →
                    </span>
                  </div>

                  <div className="border border-slate-200 rounded-xl overflow-x-auto shadow-2xs">
                    <table className="w-full text-left text-[11px] min-w-[320px]">
                      <thead className="bg-slate-100 text-slate-700 font-hero font-semibold border-b border-slate-200">
                        <tr>
                          {activeDataset.sqlColumns.map((col) => (
                            <th key={col} className="px-3 py-1.5">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 bg-white font-mono">
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
              <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-xs sm:text-sm">
                <span className="text-[13px] sm:text-sm font-hero font-semibold text-slate-800">
                  Try this with your own spreadsheet
                </span>
                <Link 
                  to="/workspace" 
                  className="inline-flex items-center gap-1.5 font-hero font-bold text-[#4338ca] hover:text-[#312e81] hover:underline text-xs sm:text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 rounded-md"
                >
                  <span>Open Free Workspace</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>
              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default HeroSection;
