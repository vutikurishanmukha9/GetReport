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
  // Unique formats (no duplicates)
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
    type: "BOTH",
    ext: ".PARQUET",
    name: "Apache Parquet",
    subtitle: "Columnar analytics ingestion & export",
    engine: "Arrow IPC zero-copy engine",
    previewType: "sheet",
    icon: Database,
  },
  {
    id: "fmt-json",
    type: "BOTH",
    ext: ".JSON",
    name: "Structured Object",
    subtitle: "JSON & JSONL ingestion & export",
    engine: "Streaming simdjson validator",
    previewType: "code",
    icon: FileCode,
  },
  {
    id: "fmt-snowflake",
    type: "IN",
    ext: "SNOWFLAKE",
    name: "Warehouse Table",
    subtitle: "Direct warehouse table & query sync",
    engine: "Async streaming connector",
    previewType: "sheet",
    icon: Database,
  },
  {
    id: "fmt-rules",
    type: "OUT",
    ext: ".JSON (RULES)",
    name: "Cleaning Recipe",
    subtitle: "Reproducible pipeline cleaning rules",
    engine: "DAG deterministic execution engine",
    previewType: "code",
    icon: FileCode,
  },
  {
    id: "fmt-tsv",
    type: "BOTH",
    ext: ".TSV",
    name: "Tab-Separated",
    subtitle: "Bioinformatics & TSV format support",
    engine: "Fast tab-delimited processor",
    previewType: "sheet",
    icon: Table,
  },
];

const FLOWING_DOCUMENTS = [
  ...DOCUMENTS.map((d) => ({ ...d, uniqueFlowKey: `${d.id}-flow-a` })),
  ...DOCUMENTS.map((d) => ({ ...d, uniqueFlowKey: `${d.id}-flow-b` })),
];

export const EngineTrustBar = () => {

  const renderMiniPreview = (previewType: "sheet" | "report" | "code", ext: string) => {
    if (previewType === "sheet") {
      return (
        <div className="rounded-lg bg-muted/50 dark:bg-black/40 border border-border dark:border-white/5 p-2 space-y-1.5 font-mono text-[9px] select-none transition-all duration-250 group-hover/card:bg-muted dark:group-hover/card:bg-black/70 group-hover/card:border-border dark:group-hover/card:border-white/10 group-hover/card:shadow-inner">
          <div className="grid grid-cols-3 gap-1 text-muted-foreground border-b border-border dark:border-white/5 pb-1 font-semibold">
            <span>A: ID</span>
            <span>B: VALUE</span>
            <span>C: STAT</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-foreground/80">
            <span>#1042</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-medium">$48,200</span>
            <span className="text-muted-foreground">OK</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-foreground/80">
            <span>#1043</span>
            <span className="text-emerald-600 dark:text-emerald-400 font-medium">$32,150</span>
            <span className="text-muted-foreground">OK</span>
          </div>
        </div>
      );
    }
    if (previewType === "report") {
      return (
        <div className="rounded-lg bg-muted/50 dark:bg-black/40 border border-border dark:border-white/5 p-2 space-y-1.5 font-mono text-[9px] select-none transition-all duration-250 group-hover/card:bg-muted dark:group-hover/card:bg-black/70 group-hover/card:border-border dark:group-hover/card:border-white/10 group-hover/card:shadow-inner">
          <div className="flex items-center justify-between border-b border-border dark:border-white/5 pb-1">
            <span className="text-foreground font-bold">AUDIT REPORT</span>
            <span className="text-emerald-600 dark:text-emerald-400 text-[8px] px-1 rounded bg-emerald-500/10 font-bold">99.4%</span>
          </div>
          <div className="space-y-1 pt-0.5">
            <div className="h-1.5 bg-zinc-300 dark:bg-zinc-500/30 rounded w-full group-hover/card:bg-zinc-400 dark:group-hover/card:bg-zinc-500/40 transition-colors duration-250" />
            <div className="h-1.5 bg-violet-400/40 dark:bg-violet-500/30 rounded w-3/4 group-hover/card:bg-violet-400/60 dark:group-hover/card:bg-violet-500/40 transition-colors duration-250" />
            <div className="h-1.5 bg-emerald-400/40 dark:bg-emerald-500/30 rounded w-5/6 group-hover/card:bg-emerald-400/60 dark:group-hover/card:bg-emerald-500/40 transition-colors duration-250" />
          </div>
        </div>
      );
    }
    return (
      <div className="rounded-lg bg-muted/50 dark:bg-black/40 border border-border dark:border-white/5 p-2 font-mono text-[8.5px] text-muted-foreground space-y-0.5 select-none transition-all duration-250 group-hover/card:bg-muted dark:group-hover/card:bg-black/70 group-hover/card:border-border dark:group-hover/card:border-white/10 group-hover/card:shadow-inner">
        <div><span className="text-violet-600 dark:text-violet-400">&#123;</span> &quot;format&quot;: <span className="text-emerald-600 dark:text-emerald-300">&quot;{ext}&quot;</span>,</div>
        <div className="pl-2">&quot;valid_rows&quot;: <span className="text-amber-600 dark:text-amber-300">140000</span>,</div>
        <div className="pl-2">&quot;errors&quot;: <span className="text-emerald-600 dark:text-emerald-300">0</span> <span className="text-violet-600 dark:text-violet-400">&#125;</span></div>
      </div>
    );
  };

  return (
    <section className="border-b border-border bg-background py-6 sm:py-10 overflow-hidden text-foreground transition-colors duration-200">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto mb-4 sm:mb-8 space-y-2 sm:space-y-2.5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-muted border border-border text-muted-foreground text-[11px] font-mono font-semibold tracking-wide">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span>Universal Ingestion & Export Engine</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-foreground tracking-[-0.02em]">
            Supported File Formats: Input & Output
          </h2>
          <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl mx-auto font-sans font-normal leading-relaxed">
            Formats like Excel (.XLSX) and CSV (.CSV) are fully supported for both input ingestion and cleaned export, alongside specialized database inputs and executive PDF deliverables.
          </p>

          {/* Format Type Classification Legend & Speed Controls */}
          <div className="flex flex-wrap items-center justify-center gap-2 pt-1 font-mono text-[10.5px]">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-500/30 font-medium">
              <ArrowLeftRight className="h-3 w-3" />
              <span>Input & Output (Both)</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30 font-medium">
              <ArrowDownLeft className="h-3 w-3" />
              <span>Input Ingestion</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30 font-medium">
              <ArrowUpRight className="h-3 w-3" />
              <span>Export Deliverable</span>
            </span>
          </div>
        </div>

      </div>

      {/* Infinite Horizontal Moving Stream */}
      <div className="relative w-full overflow-hidden [mask-image:linear-gradient(to_right,transparent_0%,black_6%,black_94%,transparent_100%)] [-webkit-mask-image:linear-gradient(to_right,transparent_0%,black_6%,black_94%,transparent_100%)] group py-4 sm:py-5">
        <div 
          className="animate-marquee-flow flex items-center gap-4 sm:gap-5"
          style={{ "--marquee-speed": "9s" } as React.CSSProperties}
        >
          {FLOWING_DOCUMENTS.map((doc) => {
            const IconComponent = doc.icon;
            const isBoth = doc.type === "BOTH";
            const isInput = doc.type === "IN";

            const themeStyles = isBoth
              ? {
                  cardBorder: "border-border dark:border-white/10 hover:border-violet-500/50",
                  cardBg: "bg-card dark:bg-[#0c0d12] hover:bg-violet-500/[0.04] dark:hover:bg-[#15131a]",
                  cardShadow: "shadow-sm dark:shadow-xl hover:shadow-[0_20px_40px_-15px_rgba(139,92,246,0.2),0_0_24px_rgba(139,92,246,0.08)]",
                  sheenBg: "bg-gradient-to-b from-violet-500/12 via-violet-500/3 to-transparent",
                  badge: "bg-violet-500/15 text-violet-700 dark:text-violet-300 border border-violet-500/30 group-hover/card:border-violet-400/60",
                  icon: "text-violet-600 dark:text-violet-400 group-hover/card:text-violet-500",
                  dot: "bg-violet-500 group-hover/card:bg-violet-400 group-hover/card:shadow-[0_0_8px_#a78bfa]",
                  dogEarFold: "group-hover/card:border-b-zinc-200 dark:group-hover/card:border-b-[#201d2a]",
                }
              : isInput
              ? {
                  cardBorder: "border-border dark:border-white/10 hover:border-amber-500/40",
                  cardBg: "bg-card dark:bg-[#0c0d12] hover:bg-amber-500/[0.04] dark:hover:bg-[#141310]",
                  cardShadow: "shadow-sm dark:shadow-xl hover:shadow-[0_20px_40px_-15px_rgba(245,158,11,0.15),0_0_24px_rgba(245,158,11,0.06)]",
                  sheenBg: "bg-gradient-to-b from-amber-500/10 via-amber-500/2 to-transparent",
                  badge: "bg-amber-500/15 text-amber-700 dark:text-amber-300 border border-amber-500/30 group-hover/card:border-amber-400/60",
                  icon: "text-amber-600 dark:text-amber-400 group-hover/card:text-amber-500",
                  dot: "bg-amber-500 group-hover/card:bg-amber-400 group-hover/card:shadow-[0_0_8px_#fbbf24]",
                  dogEarFold: "group-hover/card:border-b-zinc-200 dark:group-hover/card:border-b-[#261f14]",
                }
              : {
                  cardBorder: "border-border dark:border-white/10 hover:border-emerald-500/50",
                  cardBg: "bg-card dark:bg-[#0c0d12] hover:bg-emerald-500/[0.04] dark:hover:bg-[#101514]",
                  cardShadow: "shadow-sm dark:shadow-xl hover:shadow-[0_20px_40px_-15px_rgba(16,185,129,0.2),0_0_24px_rgba(16,185,129,0.08)]",
                  sheenBg: "bg-gradient-to-b from-emerald-500/12 via-emerald-500/3 to-transparent",
                  badge: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border border-emerald-500/30 group-hover/card:border-emerald-400/60",
                  icon: "text-emerald-600 dark:text-emerald-400 group-hover/card:text-emerald-500",
                  dot: "bg-emerald-500 group-hover/card:bg-emerald-400 group-hover/card:shadow-[0_0_8px_#34d399]",
                  dogEarFold: "group-hover/card:border-b-zinc-200 dark:group-hover/card:border-b-[#10201d]",
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
                  <div className="absolute top-0 right-0 w-0 h-0 border-t-[28px] border-t-zinc-200 dark:border-t-white/10 group-hover/card:border-t-zinc-300 dark:group-hover/card:border-t-white/25 border-l-[28px] border-l-transparent transition-colors duration-250" />
                  <div className={`absolute top-0 right-0 w-0 h-0 border-b-[26px] border-b-zinc-100 dark:border-b-[#121422] ${themeStyles.dogEarFold} border-r-[26px] border-r-transparent shadow-xs transition-colors duration-250`} />
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
                  <div className="font-mono font-black text-2xl sm:text-3xl tracking-tight text-foreground transition-all duration-200 group-hover/card:text-foreground">
                    {doc.ext}
                  </div>
                  <div className="text-xs sm:text-[13px] font-sans font-bold text-foreground/90 group-hover/card:text-foreground transition-colors duration-200">
                    {doc.name}
                  </div>
                  <div className="text-[10.5px] text-muted-foreground font-sans line-clamp-1 group-hover/card:text-foreground/80 transition-colors duration-200">
                    {doc.subtitle}
                  </div>
                </div>

                {/* Miniature Document Content Simulation Preview */}
                <div className="my-auto py-1 relative z-10">
                  {renderMiniPreview(doc.previewType, doc.ext)}
                </div>

                {/* Document Footer: Engine Details */}
                <div className="pt-2 border-t border-border dark:border-white/[0.08] group-hover/card:border-border/80 dark:group-hover/card:border-white/15 flex items-center justify-between text-[10px] font-mono text-muted-foreground relative z-10 transition-colors duration-200">
                  <span className="truncate group-hover/card:text-foreground/90 transition-colors duration-200">{doc.engine}</span>
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
