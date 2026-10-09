import { useState, useEffect } from "react";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";

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
    <section className="relative pt-28 lg:pt-30 pb-6 sm:pb-12 bg-hero-custom-gradient border-b border-white/10 overflow-hidden text-center text-zinc-100">
      {/* 1. Subtle Micro-Dot Technical Grid */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(255,255,255,0.08)_1px,transparent_1px)] [background-size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_30%,#000_65%,transparent_100%)] opacity-30 pointer-events-none" />

      {/* 2. Tactile Film Grain Overlay */}
      <div className="hero-grain-overlay" />

      <motion.div 
        className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-5xl relative z-10 space-y-5 sm:space-y-7"
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        {/* Display Headline with Smooth Text Motion - Linear / Vercel style */}
        <h1 className="text-3xl xs:text-4xl sm:text-5xl md:text-6xl lg:text-[4.5rem] font-display font-black tracking-[-0.04em] text-white leading-[1.1] sm:leading-[1.08] max-w-4xl mx-auto">
          <span className="block">The data audit platform</span>
          <span className="block">built for</span>
          <span className="inline-block relative overflow-hidden align-bottom min-h-[1.15em] min-w-[260px] xs:min-w-[300px] sm:min-w-[440px] px-1">
            <AnimatePresence mode="wait">
              <motion.span
                key={SAAS_HIGHLIGHTS[index]}
                initial={{ y: 36, opacity: 0, filter: "blur(6px)" }}
                animate={{ y: 0, opacity: 1, filter: "blur(0px)" }}
                exit={{ y: -36, opacity: 0, filter: "blur(6px)" }}
                transition={{ duration: 0.42, ease: [0.16, 1, 0.3, 1] }}
                className="inline-block text-transparent bg-clip-text bg-gradient-to-r from-white via-zinc-100 to-indigo-300"
              >
                {SAAS_HIGHLIGHTS[index]}
              </motion.span>
            </AnimatePresence>
          </span>
        </h1>

        {/* Subheadline - Clear SaaS value proposition without em dash */}
        <p className="text-base sm:text-lg md:text-xl text-zinc-400 leading-relaxed max-w-2xl mx-auto font-sans font-normal [text-wrap:pretty]">
          Audit millions of rows in seconds. Detect silent formula corruption, hidden blank cells, and conflicting anomalies before they reach clients or leadership. 100% locally with zero cloud storage.
        </p>

        {/* Action Button Row - Dual Pill CTAs with 48px+ touch targets */}
        <div className="space-y-3 pt-2">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3 sm:gap-4 max-w-md sm:max-w-none mx-auto">
            <Link to="/workspace" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-8 rounded-full bg-white hover:bg-zinc-100 text-black font-sans font-bold text-sm tracking-tight shadow-[0_2px_20px_rgba(255,255,255,0.22),inset_0_1px_0_rgba(255,255,255,1)] border border-white transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2.5 cursor-pointer group"
              >
                <span>Start Free Audit</span>
                <ArrowRight className="h-4 w-4 text-black transition-transform duration-200 group-hover:translate-x-1" />
              </button>
            </Link>

            <Link to="/how-it-works" className="w-full sm:w-auto">
              <button
                type="button"
                className="w-full sm:w-auto min-h-[48px] h-12 px-7 rounded-full bg-white/[0.05] hover:bg-white/[0.1] text-zinc-200 hover:text-white font-sans font-semibold text-sm border border-white/10 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center cursor-pointer"
              >
                <span>See How It Works</span>
              </button>
            </Link>
          </div>

          {/* Micro Trust Indicators */}
          <p className="text-[11px] sm:text-xs font-mono text-zinc-400">
            No signup required • Instant browser scan • Works with Excel, CSV & Parquet
          </p>
        </div>
      </motion.div>
    </section>
  );
};

export default HeroSection;
