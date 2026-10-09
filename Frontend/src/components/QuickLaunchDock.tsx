import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

export const QuickLaunchDock = () => {
  return (
    <section className="py-10 sm:py-14 bg-[#07080a] border-b border-white/[0.08] relative overflow-hidden text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="rounded-[1.75rem] sm:rounded-[2.5rem] bg-[#0c0d14] text-white shadow-2xl p-7 sm:p-10 text-center space-y-4 sm:space-y-5 relative overflow-hidden border border-white/10">
          <div className="space-y-3 max-w-xl mx-auto relative z-10">
            <h2 className="text-2xl xs:text-3xl sm:text-4xl font-display font-black text-white tracking-[-0.02em] leading-tight">
              Ready to check and clean your data?
            </h2>
            <p className="text-xs sm:text-base text-zinc-400 leading-relaxed font-sans font-normal">
              No account or credit card required. Upload any Excel or CSV file to immediately find errors, fix duplicates, and get a clear quality score.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 relative z-10">
            <Link to="/workspace" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-7 rounded-full bg-white hover:bg-zinc-100 text-black font-sans font-bold text-sm tracking-tight shadow-[0_2px_16px_rgba(255,255,255,0.2),inset_0_1px_0_rgba(255,255,255,1)] border border-white transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2 cursor-pointer group"
              >
                <span>Start Checking Your Data Free</span>
                <ArrowRight className="h-4 w-4 text-black transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </Link>

            <Link to="/how-it-works" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-6 rounded-full border border-white/10 bg-white/[0.05] hover:bg-white/[0.1] text-zinc-200 font-sans font-semibold text-sm transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center cursor-pointer"
              >
                <span>See How It Works</span>
              </button>
            </Link>
          </div>

          {/* Micro trust indicators */}
          <div className="pt-5 sm:pt-6 border-t border-white/10 flex flex-wrap items-center justify-center gap-3 sm:gap-6 text-[10px] sm:text-[11px] font-mono text-zinc-400 relative z-10">
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>Works with any spreadsheet</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>100% Private (Never leaves your device)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>Free to use, no signup needed</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default QuickLaunchDock;
