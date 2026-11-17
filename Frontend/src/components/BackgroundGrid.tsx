import React from "react";

interface BackgroundGridProps {
  className?: string;
  patternId?: string;
  glow?: boolean;
}

export const BackgroundGrid: React.FC<BackgroundGridProps> = ({
  className = "",
  patternId = "page-grid-blueprint",
  glow = true,
}) => {
  return (
    <div
      className={`fixed inset-0 pointer-events-none select-none overflow-hidden z-0 ${className}`}
      aria-hidden="true"
    >
      {/* Ambient Focal Glow Underlay (Illuminates grid lines naturally from top) */}
      {glow && (
        <>
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_55%_at_50%_15%,rgba(255,255,255,0.7),transparent_75%)] dark:bg-[radial-gradient(ellipse_75%_55%_at_50%_15%,rgba(255,255,255,0.06),transparent_75%)]" />
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_45%_at_50%_20%,rgba(128,0,0,0.035),transparent_70%)] dark:bg-[radial-gradient(ellipse_60%_45%_at_50%_20%,rgba(244,63,94,0.03),transparent_70%)]" />
        </>
      )}

      {/* Dual-Tone Architectural Vector Matrix (80px Major Structural Grid + 20px Micro-Subdivisions) */}
      <svg
        className="absolute inset-0 h-full w-full"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id={patternId}
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
              className="stroke-stone-900/[0.045] dark:stroke-white/[0.05]"
            />
            {/* Primary 80px Structural Ledger Grid Lines */}
            <path
              d="M.5 80V.5H80"
              fill="none"
              strokeWidth="1"
              className="stroke-stone-900/[0.12] dark:stroke-white/[0.13]"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" strokeWidth="0" fill={`url(#${patternId})`} />
      </svg>

      {/* Tactile Film Grain Overlay */}
      <div className="hero-grain-overlay" />
    </div>
  );
};

export default BackgroundGrid;
