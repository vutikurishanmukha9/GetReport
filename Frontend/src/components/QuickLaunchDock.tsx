import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const QuickLaunchDock = () => {
  return (
    <section className="py-10 sm:py-14 bg-gradient-to-b from-[#EEF2F6]/80 via-[#F1F5F9]/60 to-[#E8EDF5]/80 backdrop-blur-xs border-b border-slate-300/80 relative overflow-hidden">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="rounded-[1.75rem] sm:rounded-[2.5rem] bg-gradient-to-br from-violet-900 via-indigo-950 to-purple-900 text-white shadow-2xl p-6 sm:p-10 text-center space-y-5 sm:space-y-6 relative overflow-hidden border border-violet-400/30">
          <div className="absolute top-0 right-0 w-96 h-96 bg-violet-500/20 rounded-full blur-3xl pointer-events-none" />
          <div className="absolute -bottom-10 -left-10 w-80 h-80 bg-fuchsia-500/15 rounded-full blur-3xl pointer-events-none" />

          <div className="space-y-3 max-w-xl mx-auto relative z-10">
            <h2 className="text-2xl xs:text-3xl sm:text-4xl font-display font-black text-white tracking-[-0.02em] leading-tight">
              Ready to check and clean your data?
            </h2>
            <p className="text-xs sm:text-base text-violet-200/90 leading-relaxed font-normal">
              No account or credit card required. Upload any Excel or CSV file to immediately find errors, fix duplicates, and get a clear quality score.
            </p>
          </div>

          {/* Primary Action Button */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 relative z-10">
            <Link to="/workspace" className="w-full sm:w-auto">
              <button
                className="w-full sm:w-auto h-12 px-7 rounded-xl bg-white text-violet-950 hover:bg-violet-50 font-display font-bold text-sm tracking-tight shadow-[0_4px_16px_-2px_rgba(0,0,0,0.25),inset_0_1px_1px_rgba(255,255,255,0.9)] border border-white/60 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 gap-2.5 cursor-pointer flex items-center justify-center group"
              >
                <span>Start Checking Your Data Free</span>
                <ArrowRight className="h-4 w-4 text-violet-700 transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </Link>

            <Link to="/how-it-works" className="w-full sm:w-auto">
              <Button
                size="lg"
                variant="outline"
                className="w-full sm:w-auto h-12 px-6 rounded-xl border-white/30 bg-white/10 hover:bg-white/20 text-white font-display font-semibold text-sm shadow-2xs hover:border-violet-300 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center"
              >
                <span>See How It Works</span>
              </Button>
            </Link>
          </div>

          {/* Micro trust indicators */}
          <div className="pt-5 sm:pt-6 border-t border-white/15 flex flex-wrap items-center justify-center gap-3 sm:gap-6 text-[10px] sm:text-[11px] font-mono text-violet-200/80 relative z-10">
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shrink-0" />
              <span>Works with any spreadsheet</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shrink-0" />
              <span>100% Private (Never leaves your device)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-300 shrink-0" />
              <span>Free to use, no signup needed</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default QuickLaunchDock;
