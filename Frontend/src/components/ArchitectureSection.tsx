import { Cpu, ShieldCheck, Layers, Terminal } from "lucide-react";

export const ArchitectureSection = () => {
  const stages = [
    {
      step: "01",
      title: "Upload Your File",
      engine: "Excel, CSV, or data files",
      description: "Drag and drop your spreadsheet of any size. GetReport instantly scans your columns, verifies formats, and prepares your audit in seconds.",
      badge: "Under 2 Seconds",
      accentBg: "bg-stone-200 border border-stone-300 text-stone-800 dark:bg-white/[0.04] dark:border-white/10 dark:text-zinc-300",
      badgeColor: "bg-stone-200 border border-stone-300 text-stone-800 dark:bg-white/[0.04] dark:text-zinc-300 dark:border-white/10",
      icon: Cpu,
    },
    {
      step: "02",
      title: "Spot Errors & Outliers",
      engine: "Automated quality check",
      description: "Automatically detects blank cells, duplicate rows, broken formats, and unusual numbers that could distort your metrics.",
      badge: "Instant Quality Score",
      accentBg: "bg-amber-100 border border-amber-300 text-amber-900 dark:bg-amber-500/10 dark:border-amber-500/25 dark:text-amber-400",
      badgeColor: "bg-amber-100 border border-amber-300 text-amber-900 dark:bg-amber-500/10 dark:text-amber-300 dark:border-amber-500/25",
      icon: Layers,
    },
    {
      step: "03",
      title: "Approve Fixes With 1 Click",
      engine: "You stay in complete control",
      description: "Review suggested fixes in an easy list. Accept recommended corrections or keep your original values. Nothing changes without your confirmation.",
      badge: "100% Transparent",
      accentBg: "bg-purple-100 border border-purple-300 text-purple-900 dark:bg-purple-500/10 dark:border-purple-500/25 dark:text-purple-400",
      badgeColor: "bg-purple-100 border border-purple-300 text-purple-900 dark:bg-purple-500/10 dark:text-purple-300 dark:border-purple-500/25",
      icon: Terminal,
    },
    {
      step: "04",
      title: "Download & Present",
      engine: "Clean Excel & PDF summaries",
      description: "Export clean spreadsheets for your team, alongside beautifully formatted executive PDF reports ready for clients and leadership.",
      badge: "Boardroom Ready",
      accentBg: "bg-emerald-100 border border-emerald-300 text-emerald-900 dark:bg-emerald-500/10 dark:border-emerald-500/25 dark:text-emerald-400",
      badgeColor: "bg-emerald-100 border border-emerald-300 text-emerald-900 dark:bg-emerald-500/10 dark:text-emerald-300 dark:border-emerald-500/25",
      icon: ShieldCheck,
    },
  ];

  return (
    <section className="py-10 sm:py-14 bg-[#F5F2EB] border-b border-stone-300/80 dark:bg-[#07080a] dark:border-white/[0.08] relative overflow-hidden text-stone-900 dark:text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-2.5 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-stone-200/80 border border-stone-300 text-stone-700 dark:bg-white/[0.05] dark:border-white/10 dark:text-zinc-300 text-[11px] font-mono font-semibold tracking-wide">
            <span>Simple 4-Step Process</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-stone-900 dark:text-white tracking-[-0.02em] leading-tight">
            From messy spreadsheet to board-ready report in 4 steps
          </h2>
          <p className="text-xs sm:text-base text-stone-600 dark:text-zinc-400 max-w-xl mx-auto leading-relaxed font-sans font-normal">
            No coding, complex setups, or data science required. Upload your file, review clear recommendations, and export polished results in seconds.
          </p>
        </div>

        {/* 4-Stage Architectural Pipeline Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5">
          {stages.map((stage) => {
            const Icon = stage.icon;
            return (
              <div
                key={stage.step}
                className="rounded-[1.75rem] sm:rounded-3xl bg-[#FAF8F5] border border-stone-300/80 p-5 sm:p-6 shadow-md flex flex-col justify-between transition-all duration-200 group hover:border-stone-400 hover:-translate-y-0.5 dark:bg-[#0c0d12] dark:border-white/10 dark:shadow-xl dark:hover:border-white/20"
              >
                <div className="space-y-3.5 sm:space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-2xl sm:text-3xl font-black text-stone-400 group-hover:text-stone-700 dark:text-zinc-700 dark:group-hover:text-zinc-300 transition-colors">
                      {stage.step}
                    </span>
                    <div className={`h-10 w-10 sm:h-11 sm:w-11 rounded-xl sm:rounded-2xl flex items-center justify-center border shrink-0 ${stage.accentBg}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                  </div>

                  <div>
                    <h3 className="font-display font-bold text-base text-stone-900 dark:text-white">
                      {stage.title}
                    </h3>
                    <div className="text-[10.5px] sm:text-[11px] font-mono font-semibold mt-0.5 text-stone-600 dark:text-zinc-400">
                      {stage.engine}
                    </div>
                  </div>

                  <p className="text-xs text-stone-600 dark:text-zinc-400 leading-relaxed font-sans font-normal">
                    {stage.description}
                  </p>
                </div>

                <div className="pt-4 sm:pt-6 mt-3 sm:mt-4 border-t border-stone-300/80 dark:border-white/10">
                  <span className={`inline-flex items-center text-[10px] font-mono font-semibold px-2.5 py-1 rounded-full border ${stage.badgeColor}`}>
                    {stage.badge}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </section>
  );
};

export default ArchitectureSection;
