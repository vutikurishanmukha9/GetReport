import { Cpu, ShieldCheck, Layers, Terminal } from "lucide-react";

export const ArchitectureSection = () => {
  const stages = [
    {
      step: "01",
      title: "Upload Your File",
      engine: "Excel, CSV, or data files",
      description: "Drag and drop your spreadsheet of any size. GetReport instantly scans your columns, verifies formats, and prepares your audit in seconds.",
      badge: "Under 2 Seconds",
      cardBg: "bg-gradient-to-b from-blue-100 via-blue-50 to-sky-100/70",
      cardBorder: "border-blue-300 hover:border-blue-500 shadow-md",
      accentColor: "text-white bg-blue-600 border-blue-600 shadow-md shadow-blue-500/20",
      badgeColor: "bg-blue-600 text-white border-blue-700 font-bold shadow-2xs",
      icon: Cpu,
    },
    {
      step: "02",
      title: "Spot Errors & Outliers",
      engine: "Automated quality check",
      description: "Automatically detects blank cells, duplicate rows, broken formats, and unusual numbers that could distort your metrics.",
      badge: "Instant Quality Score",
      cardBg: "bg-gradient-to-b from-violet-100 via-purple-50 to-indigo-100/70",
      cardBorder: "border-violet-300 hover:border-violet-500 shadow-md",
      accentColor: "text-white bg-violet-600 border-violet-600 shadow-md shadow-violet-500/20",
      badgeColor: "bg-violet-600 text-white border-violet-700 font-bold shadow-2xs",
      icon: Layers,
    },
    {
      step: "03",
      title: "Approve Fixes With 1 Click",
      engine: "You stay in complete control",
      description: "Review suggested fixes in an easy list. Accept recommended corrections or keep your original values—nothing changes without your confirmation.",
      badge: "100% Transparent",
      cardBg: "bg-gradient-to-b from-amber-100 via-orange-50 to-yellow-100/70",
      cardBorder: "border-amber-300 hover:border-amber-500 shadow-md",
      accentColor: "text-white bg-amber-600 border-amber-600 shadow-md shadow-amber-500/20",
      badgeColor: "bg-amber-600 text-white border-amber-700 font-bold shadow-2xs",
      icon: Terminal,
    },
    {
      step: "04",
      title: "Download & Present",
      engine: "Clean Excel & PDF summaries",
      description: "Export clean spreadsheets for your team, alongside beautifully formatted executive PDF reports ready for clients and leadership.",
      badge: "Boardroom Ready",
      cardBg: "bg-gradient-to-b from-emerald-100 via-teal-50 to-green-100/70",
      cardBorder: "border-emerald-300 hover:border-emerald-500 shadow-md",
      accentColor: "text-white bg-emerald-600 border-emerald-600 shadow-md shadow-emerald-500/20",
      badgeColor: "bg-emerald-600 text-white border-emerald-700 font-bold shadow-2xs",
      icon: ShieldCheck,
    },
  ];

  return (
    <section className="py-10 sm:py-14 bg-gradient-to-b from-[#EAEFF5]/80 via-[#F1F5F9]/60 to-[#E8EDF5]/80 backdrop-blur-xs border-b border-slate-300/80 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Section Header */}
        <div className="max-w-3xl mx-auto text-center space-y-3 mb-6 sm:mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-200/80 border border-slate-300 text-slate-700 text-[11px] font-mono font-semibold tracking-wide">
            <span>Simple 4-Step Process</span>
          </div>
          <h2 className="text-2xl sm:text-3xl lg:text-4xl font-display font-black text-foreground tracking-[-0.02em] leading-tight">
            From messy spreadsheet to board-ready report in 4 steps
          </h2>
          <p className="text-xs sm:text-base text-muted-foreground max-w-xl mx-auto leading-relaxed font-sans">
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
                className={`rounded-[1.75rem] sm:rounded-3xl ${stage.cardBg} border-2 ${stage.cardBorder} p-5 sm:p-6 shadow-sm flex flex-col justify-between transition-all duration-200 group`}
              >
                <div className="space-y-3.5 sm:space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-2xl sm:text-3xl font-black text-slate-300 group-hover:text-slate-900 transition-colors">
                      {stage.step}
                    </span>
                    <div className={`h-10 w-10 sm:h-11 sm:w-11 rounded-xl sm:rounded-2xl flex items-center justify-center border shrink-0 ${stage.accentColor}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                  </div>

                  <div>
                    <h3 className="font-display font-bold text-base text-slate-900">
                      {stage.title}
                    </h3>
                    <div className="text-[10.5px] sm:text-[11px] font-mono font-semibold mt-0.5 text-slate-600">
                      {stage.engine}
                    </div>
                  </div>

                  <p className="text-xs text-slate-600 leading-relaxed font-sans">
                    {stage.description}
                  </p>
                </div>

                <div className="pt-4 sm:pt-6 mt-3 sm:mt-4 border-t border-slate-100">
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
