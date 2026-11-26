import { useState, useEffect } from "react";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { m, AnimatePresence } from "framer-motion";

const SAAS_HIGHLIGHTS = [
  "boardroom decisions.",
  "zero-error audits.",
  "high-stakes numbers.",
  "executive reporting.",
  "critical spreadsheets.",
];

export const HeroSection = () => {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setIndex((prev) => (prev + 1) % SAAS_HIGHLIGHTS.length);
    }, 2800);
    return () => clearInterval(timer);
  }, []);

  return (
    <section className="relative pt-28 lg:pt-30 pb-6 sm:pb-12 bg-hero-custom-gradient border-b border-stone-300/80 dark:border-white/10 overflow-hidden text-center text-stone-900 dark:text-zinc-100">
      {/* 1. Dual-Tone Architectural Blueprint Grid with Center Alignment & Ambient Illumination */}
      <div 
        className="absolute inset-0 pointer-events-none select-none overflow-hidden [mask-image:radial-gradient(ellipse_115%_85%_at_50%_36%,#000_45%,transparent_98%)]"
        aria-hidden="true"
      >
        {/* Ambient Focal Glow Underlay (Illuminates grid lines naturally at center) */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_75%_60%_at_50%_32%,rgba(255,255,255,0.7),transparent_75%)] dark:bg-[radial-gradient(ellipse_70%_55%_at_50%_30%,rgba(255,255,255,0.06),transparent_75%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_45%_at_50%_34%,rgba(128,0,0,0.035),transparent_70%)] dark:bg-[radial-gradient(ellipse_60%_45%_at_50%_32%,rgba(244,63,94,0.03),transparent_70%)]" />

        {/* Dual-Tone Architectural Vector Matrix (80px Major Structural Grid + 20px Micro-Subdivisions) */}
        <svg
          className="absolute inset-0 h-full w-full"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern
              id="hero-grid-blueprint"
              width="80"
              height="80"
              patternUnits="userSpaceOnUse"
              x="50%"
              y="-1"
            >
              {/* Secondary 20px Blueprint Micro-Subdivisions */}
              <path
                d="M 20.5 0 V 80 M 40.5 0 V 80 M 60.5 0 V 80 M 0 20.5 H 80 M 0 40.5 H 80 M 0 60.5 H 80"
                fill="none"
                strokeWidth="1"
                className="stroke-stone-900/[0.035] dark:stroke-white/[0.04]"
              />
              {/* Primary 80px Structural Ledger Grid Lines */}
              <path
                d="M.5 80V.5H80"
                fill="none"
                strokeWidth="1"
                className="stroke-stone-900/[0.10] dark:stroke-white/[0.11]"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" strokeWidth="0" fill="url(#hero-grid-blueprint)" />
        </svg>
      </div>

      {/* 2. Tactile Film Grain Overlay */}
      <div className="hero-grain-overlay" />

      <m.div
        className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl relative z-10 space-y-5 sm:space-y-7"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        {/* Display Headline with Smooth Text Motion - Linear / Vercel style */}
        <h1 className="text-3xl xs:text-4xl sm:text-5xl md:text-6xl lg:text-[4.5rem] font-display font-black tracking-[-0.04em] text-stone-900 dark:text-white leading-[1.1] sm:leading-[1.08] max-w-4xl mx-auto">
          <span className="block">The data audit platform</span>
          <span className="block">built for</span>
          <span className="inline-block relative overflow-hidden align-bottom min-h-[1.15em] min-w-[260px] xs:min-w-[300px] sm:min-w-[440px] px-1">
            <AnimatePresence mode="wait">
              <m.span
                key={SAAS_HIGHLIGHTS[index]}
                initial={{ y: 36, opacity: 0, filter: "blur(6px)" }}
                animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
                exit={{ y: -36, opacity: 0, filter: "blur(6px)" }}
                transition={{ duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
                className="inline-block text-transparent bg-clip-text bg-gradient-to-r from-stone-900 via-stone-800 to-stone-600 dark:from-white dark:via-zinc-100 dark:to-zinc-400"
              >
                {SAAS_HIGHLIGHTS[index]}
              </m.span>
            </AnimatePresence>
          </span>
        </h1>

        {/* Subheadline - Enterprise Technical Architecture Statement */}
        <p className="text-base sm:text-lg md:text-xl lg:text-[1.3rem] text-stone-600 dark:text-zinc-300 leading-relaxed max-w-3xl mx-auto font-sans font-normal [text-wrap:balance] pt-1">
          <span className="text-stone-950 dark:text-white font-semibold">
            Audit millions of rows in seconds.
          </span>{" "}
          <span>
            Detect silent formula corruption, hidden blank cells, and conflicting anomalies before they reach clients or leadership.
          </span>{" "}
          <span className="text-emerald-700 dark:text-emerald-400 font-semibold underline decoration-emerald-500/40 underline-offset-4">
            100% locally with zero cloud storage.
          </span>
        </p>

        {/* Action Button Row - Dual Pill CTAs with 48px+ touch targets */}
        <div className="space-y-3 pt-2">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 sm:gap-4 max-w-md sm:max-w-none mx-auto">
            <Link to="/workspace" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-8 rounded-full bg-[#800000] hover:bg-[#6b0000] text-white dark:bg-white dark:hover:bg-zinc-100 dark:text-black font-sans font-bold text-sm tracking-tight shadow-md border border-[#800000] dark:border-white transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2.5 cursor-pointer group"
              >
                <span>Start Free Audit</span>
                <ArrowRight className="h-4 w-4 text-white dark:text-black transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </Link>

            <Link to="/how-it-works" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-7 rounded-full bg-stone-200/70 hover:bg-stone-300 text-stone-800 dark:bg-white/[0.05] dark:hover:bg-white/[0.1] dark:text-zinc-200 dark:hover:text-white font-sans font-semibold text-sm border border-stone-300 dark:border-white/10 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center cursor-pointer"
              >
                <span>See How It Works</span>
              </button>
            </Link>
          </div>

          {/* Micro Trust Indicators with Deliberate Spacing & Refined Typography */}
          <div className="flex flex-wrap items-center justify-center gap-x-6 sm:gap-x-7 gap-y-2 pt-2.5 text-[13px] sm:text-sm font-sans font-medium text-stone-600 dark:text-zinc-400 tracking-[-0.01em]">
            <span className="whitespace-nowrap">No signup required</span>
            <span className="hidden sm:inline-block w-1 h-1 rounded-full bg-stone-300 dark:bg-zinc-700 shrink-0" aria-hidden="true" />
            <span className="whitespace-nowrap">Instant browser scan</span>
            <span className="hidden sm:inline-block w-1 h-1 rounded-full bg-stone-300 dark:bg-zinc-700 shrink-0" aria-hidden="true" />
            <span className="whitespace-nowrap">Works with Excel, CSV & Parquet</span>
          </div>
        </div>
      </m.div>
    </section>
  );
};

export default HeroSection;
