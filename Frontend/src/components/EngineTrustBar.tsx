import { ShieldCheck, Cpu, FileText } from "lucide-react";

export const EngineTrustBar = () => {
  const formats = [
    { ext: ".CSV", label: "Comma-Separated" },
    { ext: ".XLSX", label: "Excel Sheets" },
    { ext: ".PARQUET", label: "Apache Parquet" },
    { ext: ".TSV", label: "Tab-Separated" },
    { ext: ".JSONL", label: "JSON Lines" },
  ];

  return (
    <section className="border-b border-slate-200/80 bg-white/70 backdrop-blur-md py-4 sm:py-5 overflow-hidden text-slate-700">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-4 lg:gap-6">
          
          {/* Formats Section */}
          <div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-2 sm:gap-2.5 w-full lg:w-auto">
            <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-slate-500 text-center sm:text-left">
              Supported Formats:
            </span>
            <div className="flex flex-wrap items-center justify-center gap-1.5 sm:gap-2">
              {formats.map((item) => (
                <span
                  key={item.ext}
                  className="inline-flex items-center px-2.5 py-1 rounded-md border border-slate-200 bg-white text-slate-800 font-mono text-[11px] font-semibold tracking-wide shadow-2xs hover:border-violet-400 hover:text-violet-700 transition-colors"
                  title={item.label}
                >
                  {item.ext}
                </span>
              ))}
            </div>
          </div>

          {/* Engine & Security Badges */}
          <div className="flex flex-wrap items-center justify-center lg:justify-end gap-2 sm:gap-4 font-mono text-[10.5px] sm:text-[11px] text-slate-600 w-full lg:w-auto">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-violet-50/80 border border-violet-200 text-violet-700 font-medium shadow-2xs">
              <Cpu className="h-3.5 w-3.5 text-violet-600" />
              <span>Instant Scan (&lt; 2s)</span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50/80 border border-indigo-200 text-indigo-700 font-medium shadow-2xs">
              <FileText className="h-3.5 w-3.5 text-indigo-600" />
              <span>Executive PDF Briefs</span>
            </div>
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50/80 border border-emerald-200 text-emerald-800 font-semibold shadow-2xs">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
              <span>100% Private (Never Stored)</span>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};
