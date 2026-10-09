import { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { FileSpreadsheet, ArrowRight } from "lucide-react";

export const MobileFloatingBar = () => {
  const [isVisible, setIsVisible] = useState(false);
  const location = useLocation();

  useEffect(() => {
    // Only show on home page or non-workspace pages
    if (location.pathname === "/workspace") {
      setIsVisible(false);
      return;
    }

    const handleScroll = () => {
      const scrollY = window.scrollY || document.documentElement.scrollTop;
      // Reveal past the hero section (>250px)
      setIsVisible(scrollY > 250);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    return () => window.removeEventListener("scroll", handleScroll);
  }, [location.pathname]);

  if (location.pathname === "/workspace") return null;

  return (
    <div
      className={`fixed bottom-4 left-3 right-3 z-40 max-w-sm mx-auto md:hidden pointer-events-none transition-all duration-300 ease-out transform ${
        isVisible
          ? "translate-y-0 opacity-100 scale-100"
          : "translate-y-16 opacity-0 scale-95 pointer-events-none"
      }`}
    >
      {/* Double-Bezel Hardware Architecture in Obsidian Dark */}
      <div className="pointer-events-auto p-1 rounded-full bg-white/[0.04] backdrop-blur-2xl ring-1 ring-white/10 shadow-[0_20px_48px_-10px_rgba(0,0,0,0.8),0_4px_16px_rgba(0,0,0,0.4)]">
        <div className="rounded-full bg-[#0c0d14]/95 backdrop-blur-xl border border-white/10 px-3.5 py-1.5 flex items-center justify-between gap-3">
          
          {/* Left Pod: Brand Identity & Privacy Indicator */}
          <Link to="/" className="flex items-center gap-2.5 min-w-0 group cursor-pointer">
            <div className="h-9 w-9 rounded-xl bg-white/[0.06] border border-white/10 text-white flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform duration-200">
              <FileSpreadsheet className="h-4 w-4 text-white" />
            </div>
            <div className="text-left leading-tight min-w-0">
              <span className="text-xs font-display font-black text-white tracking-tight block truncate">
                GetReport
              </span>
              <span className="text-[10px] font-mono text-emerald-400 font-semibold flex items-center gap-1 mt-0.5">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                100% Private
              </span>
            </div>
          </Link>

          {/* Right Pod: Solid White Primary CTA Button with 44px min touch target */}
          <Link to="/workspace" className="shrink-0">
            <button
              type="button"
              className="min-h-[44px] h-10 px-4 rounded-full bg-white hover:bg-zinc-100 text-black font-sans font-bold text-xs tracking-tight shadow-[0_2px_12px_rgba(255,255,255,0.2),inset_0_1px_0_rgba(255,255,255,1)] border border-white flex items-center gap-1.5 active:scale-95 transition-all duration-150 cursor-pointer"
            >
              <span>Start Free</span>
              <ArrowRight className="h-3.5 w-3.5 text-black" />
            </button>
          </Link>

        </div>
      </div>
    </div>
  );
};

export default MobileFloatingBar;
