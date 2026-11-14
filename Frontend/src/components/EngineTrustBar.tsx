import { 
  ArrowDownLeft, 
  ArrowUpRight, 
  ArrowLeftRight,
  FileSpreadsheet,
  FileText,
  Database,
  FileCode,
  Table
} from "lucide-react";

interface FormatDocument {
  id: string;
  type: "BOTH" | "IN" | "OUT";
  ext: string;
  name: string;
  subtitle: string;
  engine: string;
  previewType: "sheet" | "report" | "code";
  icon: typeof FileSpreadsheet;
}

const DOCUMENTS: FormatDocument[] = [
  // ─── UNIQUE FORMATS (NO DUPLICATES) ─────────────────────────
  {
    id: "fmt-xlsx",
    type: "BOTH",
    ext: ".XLSX",
    name: "Excel Workbook",
    subtitle: "Full support for both input & cleaned output",
    engine: "Calamine parser + formula-safe writer",
    previewType: "sheet",
    icon: FileSpreadsheet,
  },
  {
    id: "fmt-csv",
    type: "BOTH",
    ext: ".CSV",
    name: "Comma-Separated",
    subtitle: "Raw file ingestion & sanitized output",
    engine: "Auto-detect RFC4180 delimiter engine",
    previewType: "sheet",
    icon: Table,
  },
  {
    id: "fmt-pdf",
    type: "OUT",
    ext: ".PDF",
    name: "Boardroom Brief",
    subtitle: "Executive printable vector PDF report",
    engine: "WeasyPrint vector layout engine",
    previewType: "report",
    icon: FileText,
  },
  {
    id: "fmt-parquet",
    type: "IN",
    ext: ".PARQUET",
    name: "Apache Parquet",
    subtitle: "High-density columnar database files",
    engine: "Zero-copy Polars memory scanner",
    previewType: "sheet",
    icon: Database,
  },
  {
    id: "fmt-html",
    type: "OUT",
    ext: ".HTML",
    name: "Interactive Audit",
    subtitle: "Standalone offline HTML briefing",
    engine: "Self-contained responsive bundle",
    previewType: "report",
    icon: FileText,
  },
  {
    id: "fmt-tsv",
    type: "IN",
    ext: ".TSV",
    name: "Tab-Separated",
    subtitle: "Tabular database tables & TSV dumps",
    engine: "Fast vectorized row streaming",
    previewType: "sheet",
    icon: Table,
  },
  {
    id: "fmt-json",
    type: "OUT",
    ext: ".JSON",
    name: "Issue Ledger Log",
    subtitle: "Structured machine-readable audit logs",
    engine: "Full remediation trace schema",
    previewType: "code",
    icon: FileCode,
  },
  {
    id: "fmt-jsonl",
    type: "IN",
    ext: ".JSONL",
    name: "JSON Lines",
    subtitle: "Streaming event & telemetry records",
    engine: "Streaming schema inference engine",
    previewType: "code",
    icon: FileCode,
  },
  {
    id: "fmt-arrow",
    type: "IN",
    ext: ".ARROW",
    name: "Arrow / Feather",
    subtitle: "In-memory columnar memory buffers",
    engine: "Direct RAM zero-copy IPC",
    previewType: "sheet",
    icon: Database,
  },
];

// Duplicate document array for seamless unbroken infinite horizontal flow with unique keys
const FLOWING_DOCUMENTS = [
  ...DOCUMENTS.map((d) => ({ ...d, uniqueFlowKey: `${d.id}-flow-a` })),
  ...DOCUMENTS.map((d) => ({ ...d, uniqueFlowKey: `${d.id}-flow-b` })),
];

export const EngineTrustBar = () => {

  const renderMiniPreview = (previewType: "sheet" | "report" | "code", ext: string) => {
    if (previewType === "sheet") {
      return (
        <div className="rounded-lg bg-stone-100/90 border border-stone-200/90 p-2 space-y-1.5 font-mono text-[9px] select-none transition-all duration-250 group-hover/card:bg-stone-200/70 group-hover/card:border-stone-300 dark:bg-black/40 dark:border-white/5 dark:group-hover/card:bg-black/70 dark:group-hover/card:border-white/10 shadow-xs">
          <div className="grid grid-cols-3 gap-1 text-stone-500 border-b border-stone-200 pb-1 font-semibold dark:text-zinc-400 dark:border-white/5">
            <span>A: ID</span>
            <span>B: VALUE</span>
            <span>C: STAT</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-stone-700 dark:text-zinc-300">
            <span>#1042</span>
            <span className="text-emerald-700 dark:text-emerald-400 font-semibold">$48,200</span>
            <span className="text-stone-500 dark:text-zinc-400">OK</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-stone-700 dark:text-zinc-300">
            <span>#1043</span>
            <span className="text-emerald-700 dark:text-emerald-400 font-semibold">$32,150</span>
            <span className="text-stone-500 dark:text-zinc-400">OK</span>
          </div>
        </div>
      );
    }
    if (previewType === "report") {
      return (
        <div className="rounded-lg bg-stone-100/90 border border-stone-200/90 p-2 space-y-1.5 font-mono text-[9px] select-none transition-all duration-250 group-hover/card:bg-stone-200/70 group-hover/card:border-stone-300 dark:bg-black/40 dark:border-white/5 dark:group-hover/card:bg-black/70 dark:group-hover/card:border-white/10 shadow-xs">
          <div className="flex items-center justify-between border-b border-stone-200 pb-1 dark:border-white/5">
            <span className="text-stone-800 dark:text-zinc-300 font-bold">AUDIT REPORT</span>
            <span className="text-emerald-800 dark:text-emerald-400 text-[8px] px-1 rounded bg-emerald-100 dark:bg-emerald-500/10 font-bold border border-emerald-200 dark:border-transparent">99.4%</span>
          </div>
          <div className="space-y-1 pt-0.5">
            <div className="h-1.5 bg-stone-300 dark:bg-zinc-500/30 rounded w-full group-hover/card:bg-stone-400 dark:group-hover/card:bg-zinc-500/40 transition-colors duration-250" />
            <div className="h-1.5 bg-violet-400/50 dark:bg-violet-500/30 rounded w-3/4 group-hover/card:bg-violet-400/70 dark:group-hover/card:bg-violet-500/40 transition-colors duration-250" />
            <div className="h-1.5 bg-emerald-400/60 dark:bg-emerald-500/30 rounded w-5/6 group-hover/card:bg-emerald-400/80 dark:group-hover/card:bg-emerald-500/40 transition-colors duration-250" />
          </div>
        </div>
      );
    }
    return (
      <div className="rounded-lg bg-stone-100/90 border border-stone-200/90 p-2 font-mono text-[8.5px] text-stone-600 dark:text-zinc-400 space-y-0.5 select-none transition-all duration-250 group-hover/card:bg-stone-200/70 group-hover/card:border-stone-300 dark:bg-black/40 dark:border-white/5 dark:group-hover/card:bg-black/70 dark:group-hover/card:border-white/10 shadow-xs">
        <div><span className="text-violet-700 dark:text-violet-400">&#123;</span> &quot;format&quot;: <span className="text-emerald-700 dark:text-emerald-300 font-semibold">&quot;{ext}&quot;</span>,</div>
        <div className="pl-2">&quot;valid_rows&quot;: <span className="text-amber-700 dark:text-amber-300 font-semibold">140000</span>,</div>
        <div className="pl-2">&quot;errors&quot;: <span className="text-emerald-700 dark:text-emerald-300 font-semibold">0</span> <span className="text-violet-700 dark:text-violet-400">&#125;</span></div>
      </div>
    );
  };

  return (
    <section className="border-b border-stone-300/80 bg-[#FAF8F5] dark:border-white/[0.08] dark:bg-[#07080a] py-6 sm:py-10 overflow-hidden text-stone-900 dark:text-zinc-200">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header with Clear Both-Support Mention */}
        <div className="text-center max-w-3xl mx-auto mb-4 sm:mb-8 space-y-2 sm:space-y-2.5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-stone-200/80 border border-stone-300 text-stone-700 dark:bg-white/[0.05] dark:border-white/10 dark:text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400 animate-pulse" />
            <span>Universal Ingestion & Export Engine</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-stone-900 dark:text-white tracking-[-0.02em]">
            Supported File Formats: Input & Output
          </h2>
          <p className="text-xs sm:text-sm text-stone-600 dark:text-zinc-400 max-w-2xl mx-auto font-sans font-normal leading-relaxed">
            Formats like Excel (.XLSX) and CSV (.CSV) are fully supported for both input ingestion and cleaned export, alongside specialized database inputs and executive PDF deliverables.
          </p>
        </div>

      </div>

      {/* Infinite Horizontal Moving Stream of Deduplicated Document Cards */}
      <div className="relative w-full overflow-hidden [mask-image:linear-gradient(to_right,transparent_0%,black_6%,black_94%,transparent_100%)] [-webkit-mask-image:linear-gradient(to_right,transparent_0%,black_6%,black_94%,transparent_100%)] group py-4 sm:py-5">
        <div 
          className="animate-marquee-flow flex items-center gap-4 sm:gap-5"
          // SAFETY: inline CSS custom property styling requires cast to React.CSSProperties
          style={{ "--marquee-speed": "9s" } as React.CSSProperties}
        >
          {FLOWING_DOCUMENTS.map((doc) => {
            const IconComponent = doc.icon;
            const isBoth = doc.type === "BOTH";
            const isInput = doc.type === "IN";

            const themeStyles = isBoth
              ? {
                  cardBorder: "border-stone-200/90 hover:border-violet-500/50 dark:border-white/10 dark:hover:border-violet-500/50",
                  cardBg: "bg-[#FAF8F5] hover:bg-[#F3EFE6] dark:bg-[#0c0d12] dark:hover:bg-[#15131a]",
                  cardShadow: "shadow-md hover:shadow-xl dark:shadow-xl dark:hover:shadow-[0_20px_40px_-15px_rgba(139,92,246,0.35),0_0_24px_rgba(139,92,246,0.12)]",
                  sheenBg: "bg-gradient-to-b from-violet-500/10 via-violet-500/2 to-transparent dark:from-violet-500/12 dark:via-violet-500/3 dark:to-transparent",
                  badge: "bg-violet-100 text-violet-800 border-violet-200 group-hover/card:border-violet-400 dark:bg-violet-500/20 dark:text-violet-300 dark:border-violet-500/40 dark:group-hover/card:border-violet-400/60",
                  icon: "text-violet-700 group-hover/card:text-violet-800 dark:text-violet-400 dark:group-hover/card:text-violet-300",
                  dot: "bg-violet-600 group-hover/card:bg-violet-700 dark:bg-violet-400 dark:group-hover/card:bg-violet-300 dark:group-hover/card:shadow-[0_0_8px_#a78bfa]",
                  dogEarFold: "border-b-stone-300 group-hover/card:border-b-stone-400 dark:border-b-[#121422] dark:group-hover/card:border-b-[#201d2a]",
                }
              : isInput
              ? {
                  cardBorder: "border-stone-200/90 hover:border-amber-500/40 dark:border-white/10 dark:hover:border-amber-500/40",
                  cardBg: "bg-[#FAF8F5] hover:bg-[#F3EFE6] dark:bg-[#0c0d12] dark:hover:bg-[#141310]",
                  cardShadow: "shadow-md hover:shadow-xl dark:shadow-xl dark:hover:shadow-[0_20px_40px_-15px_rgba(245,158,11,0.2),0_0_24px_rgba(245,158,11,0.08)]",
                  sheenBg: "bg-gradient-to-b from-amber-500/10 via-amber-500/2 to-transparent dark:from-amber-500/10 dark:via-amber-500/2 dark:to-transparent",
                  badge: "bg-amber-100 text-amber-800 border-amber-200 group-hover/card:border-amber-400 dark:bg-amber-500/15 dark:text-amber-300 dark:border-amber-500/30 dark:group-hover/card:border-amber-400/60",
                  icon: "text-amber-700 group-hover/card:text-amber-800 dark:text-amber-400 dark:group-hover/card:text-amber-300",
                  dot: "bg-amber-600 group-hover/card:bg-amber-700 dark:bg-amber-400 dark:group-hover/card:bg-amber-300 dark:group-hover/card:shadow-[0_0_8px_#fbbf24]",
                  dogEarFold: "border-b-stone-300 group-hover/card:border-b-stone-400 dark:border-b-[#121422] dark:group-hover/card:border-b-[#261f14]",
                }
              : {
                  cardBorder: "border-stone-200/90 hover:border-emerald-500/50 dark:border-white/10 dark:hover:border-emerald-500/50",
                  cardBg: "bg-[#FAF8F5] hover:bg-[#F3EFE6] dark:bg-[#0c0d12] dark:hover:bg-[#101514]",
                  cardShadow: "shadow-md hover:shadow-xl dark:shadow-xl dark:hover:shadow-[0_20px_40px_-15px_rgba(16,185,129,0.35),0_0_24px_rgba(16,185,129,0.12)]",
                  sheenBg: "bg-gradient-to-b from-emerald-500/10 via-emerald-500/2 to-transparent dark:from-emerald-500/12 dark:via-emerald-500/3 dark:to-transparent",
                  badge: "bg-emerald-100 text-emerald-800 border-emerald-200 group-hover/card:border-emerald-400 dark:bg-emerald-500/15 dark:text-emerald-300 dark:border-emerald-500/30 dark:group-hover/card:border-emerald-400/60",
                  icon: "text-emerald-700 group-hover/card:text-emerald-800 dark:text-emerald-400 dark:group-hover/card:text-emerald-300",
                  dot: "bg-emerald-600 group-hover/card:bg-emerald-700 dark:bg-emerald-400 dark:group-hover/card:bg-emerald-300 dark:group-hover/card:shadow-[0_0_8px_#34d399]",
                  dogEarFold: "border-b-stone-300 group-hover/card:border-b-stone-400 dark:border-b-[#121422] dark:group-hover/card:border-b-[#10201d]",
                };

            return (
              <div
                key={doc.uniqueFlowKey}
                className={`relative w-[215px] sm:w-[235px] min-w-[215px] sm:min-w-[235px] h-[280px] sm:h-[300px] rounded-2xl border p-4 flex flex-col justify-between shrink-0 select-none cursor-default group/card transition-all duration-250 ease-spring-smooth hover:-translate-y-2.5 hover:scale-[1.015] ${themeStyles.cardBorder} ${themeStyles.cardBg} ${themeStyles.cardShadow}`}
              >
                {/* Ambient gradient top sheen that fades in on hover */}
                <div className={`absolute inset-0 rounded-2xl pointer-events-none transition-opacity duration-300 opacity-0 group-hover/card:opacity-100 ${themeStyles.sheenBg}`} />

                {/* Tactile Document Dog-Ear Fold at top right corner */}
                <div className="absolute top-0 right-0 w-7 h-7 overflow-hidden rounded-tr-2xl pointer-events-none z-10">
                  <div className="absolute top-0 right-0 w-0 h-0 border-t-[28px] border-t-[#FAF8F5] dark:border-t-[#07080a] border-l-[28px] border-l-transparent transition-colors duration-250" />
                  <div className={`absolute top-0 right-0 w-0 h-0 border-b-[26px] ${themeStyles.dogEarFold} border-r-[26px] border-r-transparent shadow-xs transition-colors duration-250`} />
                </div>

                {/* Document Header Bar */}
                <div className="flex items-center justify-between pr-4 relative z-10">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider transition-all duration-200 ${themeStyles.badge}`}
                  >
                    {isBoth ? (
                      <>
                        <ArrowLeftRight className="h-3 w-3" />
                        <span>IN & OUT (BOTH)</span>
                      </>
                    ) : isInput ? (
                      <>
                        <ArrowDownLeft className="h-3 w-3" />
                        <span>INPUT FILE</span>
                      </>
                    ) : (
                      <>
                        <ArrowUpRight className="h-3 w-3" />
                        <span>EXPORT FILE</span>
                      </>
                    )}
                  </span>

                  <IconComponent
                    className={`h-4 w-4 transition-transform duration-300 ease-spring-smooth group-hover/card:scale-125 group-hover/card:-rotate-6 ${themeStyles.icon}`}
                  />
                </div>

                {/* Format Stamp & Name */}
                <div className="space-y-0.5 pt-2 relative z-10">
                  <div className="font-mono font-black text-2xl sm:text-3xl tracking-tight text-stone-900 group-hover/card:text-stone-950 dark:text-white dark:group-hover/card:text-white transition-all duration-200">
                    {doc.ext}
                  </div>
                  <div className="text-xs sm:text-[13px] font-sans font-bold text-stone-800 group-hover/card:text-stone-950 dark:text-zinc-200 dark:group-hover/card:text-white transition-colors duration-200">
                    {doc.name}
                  </div>
                  <div className="text-[10.5px] text-stone-600 group-hover/card:text-stone-700 dark:text-zinc-400 dark:group-hover/card:text-zinc-300 font-sans line-clamp-1 transition-colors duration-200">
                    {doc.subtitle}
                  </div>
                </div>

                {/* Miniature Document Content Simulation Preview */}
                <div className="my-auto py-1 relative z-10">
                  {renderMiniPreview(doc.previewType, doc.ext)}
                </div>

                {/* Document Footer: Engine Details */}
                <div className="pt-2 border-t border-stone-200/90 group-hover/card:border-stone-300 dark:border-white/[0.08] dark:group-hover/card:border-white/15 flex items-center justify-between text-[10px] font-mono text-stone-600 dark:text-zinc-400 relative z-10 transition-colors duration-200">
                  <span className="truncate group-hover/card:text-stone-800 dark:group-hover/card:text-zinc-200 transition-colors duration-200">{doc.engine}</span>
                  <span
                    className={`h-1.5 w-1.5 rounded-full shrink-0 ml-1 transition-all duration-250 group-hover/card:scale-150 ${themeStyles.dot}`}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default EngineTrustBar;
