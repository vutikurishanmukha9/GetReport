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

// Duplicate document array for seamless unbroken infinite horizontal flow
const FLOWING_DOCUMENTS = [...DOCUMENTS, ...DOCUMENTS];

export const EngineTrustBar = () => {
  const renderMiniPreview = (previewType: "sheet" | "report" | "code", ext: string) => {
    if (previewType === "sheet") {
      return (
        <div className="rounded-lg bg-black/40 border border-white/5 p-2 space-y-1.5 font-mono text-[9px] select-none">
          <div className="grid grid-cols-3 gap-1 text-zinc-400 border-b border-white/5 pb-1 font-semibold">
            <span>A: ID</span>
            <span>B: VALUE</span>
            <span>C: STAT</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-zinc-300">
            <span>#1042</span>
            <span className="text-emerald-400">$48,200</span>
            <span className="text-zinc-400">OK</span>
          </div>
          <div className="grid grid-cols-3 gap-1 text-zinc-300">
            <span>#1043</span>
            <span className="text-emerald-400">$32,150</span>
            <span className="text-zinc-400">OK</span>
          </div>
        </div>
      );
    }
    if (previewType === "report") {
      return (
        <div className="rounded-lg bg-black/40 border border-white/5 p-2 space-y-1.5 font-mono text-[9px] select-none">
          <div className="flex items-center justify-between border-b border-white/5 pb-1">
            <span className="text-zinc-300 font-bold">AUDIT REPORT</span>
            <span className="text-emerald-400 text-[8px] px-1 rounded bg-emerald-500/10 font-bold">99.4%</span>
          </div>
          <div className="space-y-1 pt-0.5">
            <div className="h-1.5 bg-sky-500/30 rounded w-full" />
            <div className="h-1.5 bg-violet-500/30 rounded w-3/4" />
            <div className="h-1.5 bg-emerald-500/30 rounded w-5/6" />
          </div>
        </div>
      );
    }
    return (
      <div className="rounded-lg bg-black/40 border border-white/5 p-2 font-mono text-[8.5px] text-zinc-400 space-y-0.5 select-none">
        <div><span className="text-violet-400">&#123;</span> &quot;format&quot;: <span className="text-emerald-300">&quot;{ext}&quot;</span>,</div>
        <div className="pl-2">&quot;valid_rows&quot;: <span className="text-amber-300">140000</span>,</div>
        <div className="pl-2">&quot;errors&quot;: <span className="text-emerald-300">0</span> <span className="text-violet-400">&#125;</span></div>
      </div>
    );
  };

  return (
    <section className="border-b border-white/[0.08] bg-[#07080a] py-6 sm:py-10 overflow-hidden text-zinc-200">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header with Clear Both-Support Mention */}
        <div className="text-center max-w-3xl mx-auto mb-4 sm:mb-8 space-y-2 sm:space-y-2.5">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.05] border border-white/10 text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>Universal Ingestion & Export Engine</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-white tracking-[-0.02em]">
            Supported File Formats: Input & Output
          </h2>
          <p className="text-xs sm:text-sm text-zinc-400 max-w-2xl mx-auto font-sans font-normal leading-relaxed">
            Formats like Excel (.XLSX) and CSV (.CSV) are fully supported for both input ingestion and cleaned export, alongside specialized database inputs and executive PDF deliverables.
          </p>

          {/* Format Type Classification Legend */}
          <div className="flex flex-wrap items-center justify-center gap-2 pt-1 font-mono text-[10.5px]">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-violet-500/15 text-violet-300 border border-violet-500/30 font-medium">
              <ArrowLeftRight className="h-3 w-3" />
              <span>Input & Output (Both)</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-medium">
              <ArrowDownLeft className="h-3 w-3" />
              <span>Input Ingestion</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-medium">
              <ArrowUpRight className="h-3 w-3" />
              <span>Export Deliverable</span>
            </span>
          </div>
        </div>

      </div>

      {/* Infinite Horizontal Moving Stream of Deduplicated Document Cards */}
      <div className="relative w-full overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_4%,black_96%,transparent)] group py-2">
        <div className="animate-marquee-flow flex items-center gap-4 sm:gap-5">
          {FLOWING_DOCUMENTS.map((doc, idx) => {
            const IconComponent = doc.icon;
            const isBoth = doc.type === "BOTH";
            const isInput = doc.type === "IN";

            return (
              <div
                key={`${doc.id}-${idx}`}
                className="relative w-[215px] sm:w-[235px] min-w-[215px] sm:min-w-[235px] h-[280px] sm:h-[300px] rounded-2xl border border-white/10 bg-[#0c0d14] p-4 flex flex-col justify-between shadow-xl transition-all duration-300 hover:border-white/30 hover:-translate-y-1.5 hover:shadow-2xl hover:shadow-black/70 group/card shrink-0 select-none cursor-default"
              >
                {/* Tactile Document Dog-Ear Fold at top right corner */}
                <div className="absolute top-0 right-0 w-7 h-7 overflow-hidden rounded-tr-2xl pointer-events-none">
                  <div className="absolute top-0 right-0 w-0 h-0 border-t-[28px] border-t-white/10 border-l-[28px] border-l-transparent" />
                  <div className="absolute top-0 right-0 w-0 h-0 border-b-[26px] border-b-[#121422] border-r-[26px] border-r-transparent shadow-xs" />
                </div>

                {/* Document Header Bar */}
                <div className="flex items-center justify-between pr-4">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider ${
                      isBoth
                        ? "bg-violet-500/20 text-violet-300 border border-violet-500/40"
                        : isInput
                        ? "bg-indigo-500/15 text-indigo-300 border border-indigo-500/30"
                        : "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
                    }`}
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
                    className={`h-4 w-4 ${
                      isBoth
                        ? "text-violet-400"
                        : isInput
                        ? "text-indigo-400"
                        : "text-emerald-400"
                    }`}
                  />
                </div>

                {/* Format Stamp & Name */}
                <div className="space-y-0.5 pt-2">
                  <div className="font-mono font-black text-2xl sm:text-3xl tracking-tight text-white group-hover/card:text-zinc-100 transition-colors">
                    {doc.ext}
                  </div>
                  <div className="text-xs sm:text-[13px] font-sans font-bold text-zinc-200">
                    {doc.name}
                  </div>
                  <div className="text-[10.5px] text-zinc-400 font-sans line-clamp-1">
                    {doc.subtitle}
                  </div>
                </div>

                {/* Miniature Document Content Simulation Preview */}
                <div className="my-auto py-1">
                  {renderMiniPreview(doc.previewType, doc.ext)}
                </div>

                {/* Document Footer: Engine Details */}
                <div className="pt-2 border-t border-white/[0.08] flex items-center justify-between text-[10px] font-mono text-zinc-400">
                  <span className="truncate">{doc.engine}</span>
                  <span
                    className={`h-1.5 w-1.5 rounded-full shrink-0 ml-1 ${
                      isBoth
                        ? "bg-violet-400"
                        : isInput
                        ? "bg-indigo-400"
                        : "bg-emerald-400"
                    }`}
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
