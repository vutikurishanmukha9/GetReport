import { useState, useRef, MouseEvent } from "react";
import { BarChart3, FileSpreadsheet, ArrowRight, CheckCircle2, ShieldAlert, Cpu } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

export const HeroSection = () => {
  // 3D tilt tracking for terminal card (transitions.dev & beUI pattern)
  const cardRef = useRef<HTMLDivElement>(null);
  const [tilt, setTilt] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);

  const handleMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    // Mild perspective rotation (max +/- 5 deg)
    const rotateX = ((y - centerY) / centerY) * -5;
    const rotateY = ((x - centerX) / centerX) * 5;
    setTilt({ x: rotateX, y: rotateY });
  };

  const handleMouseLeave = () => {
    setIsHovered(false);
    setTilt({ x: 0, y: 0 });
  };

  return (
    <section className="relative pt-24 pb-12 sm:pt-28 sm:pb-16 md:pt-32 md:pb-20 lg:pt-36 lg:pb-24 overflow-hidden">
      {/* Subtle tint background */}
      <div className="absolute inset-0 bg-gradient-to-b from-muted/30 to-transparent -z-10" />

      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center">
          
          {/* Left Column: Left-aligned content */}
          <div className="lg:col-span-7 text-left space-y-6 sm:space-y-8">
            
            {/* Tagline Badge with shimmer motion token */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full border border-border/80 bg-muted/40 text-muted-foreground text-xs font-mono uppercase tracking-wider t-badge-shimmer t-card-lift">
              <span className="h-2 w-2 rounded-full bg-primary t-pulse-dot inline-block" />
              <FileSpreadsheet className="h-3.5 w-3.5 text-primary" />
              <span>Auditable Report Generator</span>
            </div>

            {/* Apple-style Display Headline */}
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-display font-semibold tracking-[-0.03em] text-foreground leading-[1.05] max-w-xl uppercase">
              Turn raw data into
              <span className="block text-primary mt-1 font-display tracking-[-0.03em]">publication-ready reports</span>
            </h1>

            {/* Subheadline */}
            <p className="text-base sm:text-lg md:text-xl text-muted-foreground max-w-lg leading-relaxed font-sans">
              Upload your CSV or Excel files. Review quality issues inside the interactive ledger,
              and receive a comprehensive PDF report with statistical charts and RAG insights.
            </p>

            {/* CTA Buttons with spring physics */}
            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 pt-2">
              <Link to="/workspace" className="w-full sm:w-auto">
                <Button size="lg" className="w-full sm:w-auto h-12 px-6 rounded-xl shadow-premium t-card-lift t-spring-press font-semibold font-display text-sm">
                  Start Free Audit
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Button>
              </Link>
              <Link to="/how-it-works" className="w-full sm:w-auto">
                <Button size="lg" variant="outline" className="w-full sm:w-auto h-12 px-6 rounded-xl border-border bg-card hover:bg-muted/10 shadow-premium t-card-lift t-spring-press font-display text-sm">
                  How it Works
                </Button>
              </Link>
            </div>

            {/* Stats row with motion tokens */}
            <div className="grid grid-cols-3 gap-2 sm:gap-4 max-w-md pt-2">
              <div className="flex flex-col p-3 sm:p-4 rounded-xl border border-border bg-card shadow-sm t-card-lift t-spring-press text-center sm:text-left">
                <span className="text-base sm:text-lg md:text-xl font-display font-semibold tracking-tight text-foreground">&lt; 1 Min</span>
                <span className="text-[8px] sm:text-[9px] text-muted-foreground font-medium mt-0.5 sm:mt-1 font-mono uppercase tracking-wider truncate">generation</span>
              </div>
              <div className="flex flex-col p-3 sm:p-4 rounded-xl border border-border bg-card shadow-sm t-card-lift t-spring-press text-center sm:text-left">
                <span className="text-base sm:text-lg md:text-xl font-display font-semibold tracking-tight text-foreground">PDF</span>
                <span className="text-[8px] sm:text-[9px] text-muted-foreground font-medium mt-0.5 sm:mt-1 font-mono uppercase tracking-wider truncate">ready report</span>
              </div>
              <div className="flex flex-col p-3 sm:p-4 rounded-xl border border-border bg-card shadow-sm t-card-lift t-spring-press text-center sm:text-left">
                <span className="text-base sm:text-lg md:text-xl font-display font-semibold tracking-tight text-foreground">A-F</span>
                <span className="text-[8px] sm:text-[9px] text-muted-foreground font-medium mt-0.5 sm:mt-1 font-mono uppercase tracking-wider truncate">data grades</span>
              </div>
            </div>

          </div>

          {/* Right Column: 3D Interactive Terminal Card */}
          <div 
            className="lg:col-span-5 w-full max-w-sm sm:max-w-md lg:max-w-none mx-auto perspective-1000"
            onMouseMove={handleMouseMove}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={handleMouseLeave}
          >
            <div 
              ref={cardRef}
              style={{
                transform: isHovered 
                  ? `perspective(1000px) rotateX(${tilt.x}deg) rotateY(${tilt.y}deg) translateY(-4px)`
                  : 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)',
                transition: isHovered ? 'transform 80ms ease-out' : 'transform 400ms cubic-bezier(0.16, 1, 0.3, 1)'
              }}
              className="relative border border-border bg-card rounded-2xl p-4 sm:p-6 shadow-premium overflow-hidden min-h-[340px] sm:min-h-[365px] flex flex-col justify-between select-none"
            >
              {/* Header */}
              <div className="flex items-center justify-between border-b pb-4 mb-4 border-border/60">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-primary t-pulse-dot inline-block" />
                  <span className="text-[10px] font-mono tracking-wider uppercase text-muted-foreground font-semibold">Active processing logs</span>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded border border-border/40">dataset_v2.csv</span>
              </div>
              
              {/* Pipeline stages list */}
              <div className="space-y-3 flex-1 font-mono text-[11px] sm:text-xs">
                <div className="flex items-center justify-between transition-colors hover:text-foreground">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" /> 01_Polars_Ingest
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">402ms</span>
                </div>
                <div className="flex items-center justify-between transition-colors hover:text-foreground">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" /> 02_Integrity_Score
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">128ms</span>
                </div>
                <div className="flex items-center justify-between transition-colors hover:text-foreground">
                  <span className="text-muted-foreground flex items-center gap-2">
                    <ShieldAlert className="h-3.5 w-3.5 text-emerald-500 shrink-0" /> 03_VIF_Collinearity
                  </span>
                  <span className="text-muted-foreground/60 text-[9px]">215ms</span>
                </div>
                <div className="flex items-center justify-between bg-primary/5 p-1.5 -mx-1.5 rounded-lg border border-primary/15">
                  <span className="text-foreground font-semibold flex items-center gap-2">
                    <Cpu className="h-3.5 w-3.5 text-primary animate-spin" /> 04_RAG_Synthesis
                  </span>
                  <span className="text-primary font-bold text-[9px] uppercase tracking-wider">Running…</span>
                </div>
                <div className="flex items-center justify-between text-muted-foreground/40">
                  <span className="flex items-center gap-2">
                    <span className="h-3.5 w-3.5 rounded-full border border-current flex items-center justify-center text-[8px]">○</span> 05_PDF_Compilation
                  </span>
                  <span className="text-[9px]">Queued</span>
                </div>
              </div>

              {/* Mini-chart & Grade A badge */}
              <div className="mt-6 pt-4 border-t border-border/60 flex items-center justify-between gap-4">
                <div className="flex-1 space-y-1.5">
                  <span className="text-[9px] font-mono uppercase text-muted-foreground flex items-center gap-1">
                    <BarChart3 className="h-3 w-3 text-primary" /> Data Health Score
                  </span>
                  <div className="flex items-end gap-1.5 h-10 pt-1">
                    <div className="w-full bg-muted rounded-t h-4 transition-all duration-300 hover:h-6" />
                    <div className="w-full bg-muted rounded-t h-6 transition-all duration-300 hover:h-8" />
                    <div className="w-full bg-muted rounded-t h-8 transition-all duration-300 hover:h-9" />
                    <div className="w-full bg-primary/45 rounded-t h-10 transition-all duration-300 hover:bg-primary/60" />
                    <div className="w-full bg-primary rounded-t h-9 transition-all duration-300 hover:h-10" />
                  </div>
                </div>
                <div className="flex flex-col items-center justify-center h-14 w-14 rounded-xl border border-border bg-background shadow-xs shrink-0 select-none t-card-lift">
                  <span className="text-[8px] font-mono text-muted-foreground leading-none font-bold uppercase">GRADE</span>
                  <span className="text-xl font-display font-bold text-primary mt-0.5">A</span>
                </div>
              </div>
              
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};
