import { ShieldCheck, Cpu, FileText, CheckCircle2 } from "lucide-react";

export const EngineTrustBar = () => {
  const formats = [
    { ext: ".CSV", label: "Comma-Separated" },
    { ext: ".XLSX", label: "Excel Sheets" },
    { ext: ".PARQUET", label: "Apache Parquet" },
    { ext: ".TSV", label: "Tab-Separated" },
    { ext: ".JSONL", label: "JSON Lines" },
  ];

  return (
    <section className="border-y border-border/70 bg-muted/25 py-4 sm:py-5 overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4 lg:gap-6">
          
          {/* Formats Section */}
          <div className="flex flex-wrap items-center justify-center lg:justify-start gap-2 sm:gap-2.5">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-muted-foreground mr-1">
              Supported Formats:
            </span>
            {formats.map((item) => (
              <span
                key={item.ext}
                className="inline-flex items-center px-2.5 py-1 rounded-md border border-border/80 bg-background/90 text-foreground font-mono text-[11px] font-semibold tracking-wide shadow-2xs hover:border-primary/40 transition-colors"
                title={item.label}
              >
                {item.ext}
              </span>
            ))}
          </div>

          {/* Engine & Security Badges */}
          <div className="flex flex-wrap items-center justify-center lg:justify-end gap-3 sm:gap-4 font-mono text-[11px] text-muted-foreground">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-background/60 border border-border/60">
              <Cpu className="h-3.5 w-3.5 text-primary" />
              <span>Polars Engine</span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-background/60 border border-border/60">
              <FileText className="h-3.5 w-3.5 text-primary" />
              <span>Dual PDF Pipelines</span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-emerald-700">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
              <span className="font-semibold">Zero-Persistence In-Memory</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};
