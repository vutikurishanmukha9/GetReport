import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const QuickLaunchDock = () => {
  return (
    <section className="py-10 sm:py-14 bg-[#07080a] border-b border-white/[0.08] relative overflow-hidden text-zinc-100">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl">
        <div className="rounded-[1.75rem] sm:rounded-[2.5rem] bg-[#0c0d12] text-white shadow-2xl p-7 sm:p-10 text-center space-y-4 sm:space-y-5 relative overflow-hidden border border-white/10">
          <div className="space-y-3 max-w-xl mx-auto relative z-10">
            <h2 className="text-2xl xs:text-3xl sm:text-4xl font-display font-extrabold text-white tracking-[-0.03em] leading-tight">
              Ready to audit and certify your data?
            </h2>
            <p className="text-xs sm:text-base text-zinc-400 leading-relaxed font-sans font-normal">
              Zero setup or infrastructure required. Run streaming schema checks, inspect transformation lineage, and compile publication-ready PDF reports in seconds.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-4 relative z-10">
            <Button
              asChild
              variant="launch"
              size="xl"
              className="w-full sm:w-auto min-h-[44px] h-11 px-7 rounded-full shadow-sm group"
            >
              <Link to="/workspace">
                <span>Launch Ingestion Studio</span>
                <ArrowRight className="h-4 w-4 text-black transition-transform duration-200 group-hover:translate-x-1" />
              </Link>
            </Button>

            <Button
              asChild
              variant="outline"
              size="xl"
              className="w-full sm:w-auto min-h-[44px] h-11 px-6 rounded-full border-white/10 bg-white/[0.05] hover:bg-white/[0.1] text-zinc-200"
            >
              <Link to="/how-it-works">
                <span>View Architecture</span>
              </Link>
            </Button>
          </div>

          {/* Technical indicators */}
          <div className="pt-5 sm:pt-6 border-t border-white/10 flex flex-wrap items-center justify-center gap-3 sm:gap-6 text-[10px] sm:text-[11px] font-mono text-zinc-400 relative z-10">
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>Zero-Copy In-Memory Polars Core</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>100% Ephemeral (Zero persistent storage)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span>Cryptographic DAG Audit Trail</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default QuickLaunchDock;
