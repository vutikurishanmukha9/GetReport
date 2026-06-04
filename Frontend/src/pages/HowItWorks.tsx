import { UploadCloud, Search, FileDown, ArrowRight, Gauge, Brain, FileCheck, Server, Sparkles, Cpu, Code2 } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const steps = [
  {
    id: "01",
    icon: UploadCloud,
    title: "Ingestion & Inferred Schemas",
    description: "Drop your CSV or Excel ledger. Our Polars parser streams the file in memory, automatically resolving character encodings, field separators, and primary column data types.",
    details: ["Supports CSV, XLSX, XLS", "Auto-detects UTF-8 & ISO-8859", "Zero database storage (in-memory)"],
  },
  {
    id: "02",
    icon: Search,
    title: "Hygiene Scoring & Capping",
    description: "Our quality algorithms calculate completeness, stability, and validity. Duplicates are flagged, missing cells capped/filled, and type mismatches repaired in a transaction audit DAG.",
    details: ["A-F column confidence grades", "Interactive check overrides", "Lock-in schema definitions"],
  },
  {
    id: "03",
    icon: Brain,
    title: "Statistical Decisions Auditing",
    description: "Executes Pearson correlations, extreme outlier checks, skewness tests, Kurtosis shapes, and time-series conceptual drift. Every test run produces structured logs explaining why it ran.",
    details: ["Why-I-Did-X transparency logs", "VIF multicollinearity metrics", "drift shifts analysis"],
  },
  {
    id: "04",
    icon: Sparkles,
    title: "AI Synthesis & RAG Context",
    description: "A semantic layer extracts the business domain (e.g. Sales, Education) and passes data summaries to our RAG engine, generating deep summaries and custom feature recommendations.",
    details: ["One-hot & scaling suggestions", "Contextual anomaly indicators", "Confidential row protection"],
  },
  {
    id: "05",
    icon: FileDown,
    title: "Board-Ready PDF Compilation",
    description: "Compiles a formatted executive audit report via WeasyPrint, featuring high-DPI charts, complete stats tables, and recommendations. Cached locally for easy stakeholder sharing.",
    details: ["Matplotlib chart embeddings", "Executive-style report cover", "Local cache buffer download"],
  },
];

export const HowItWorks = () => {
  return (
    <div className="min-h-screen bg-background animate-in fade-in duration-500 pb-20">
      
      {/* Editorial Title Header */}
      <div className="border-b border-border/60 bg-background py-20">
        <div className="container mx-auto px-4 text-center space-y-4">
          <Badge variant="outline" className="font-mono text-xs uppercase tracking-wider text-primary">
            Audit Methodology
          </Badge>
          <h1 className="text-4xl sm:text-5xl font-display font-extrabold text-foreground tracking-tight uppercase">
            Transparent Data Pipeline.
          </h1>
          <p className="text-sm sm:text-base text-muted-foreground max-w-2xl mx-auto leading-relaxed">
            From raw spreadsheet fields to board-level audit reports. We expose every decision, every threshold, and every mathematical operation.
          </p>
        </div>
      </div>

      {/* Timeline List Section */}
      <div className="container mx-auto px-4 py-20">
        <div className="max-w-3xl mx-auto relative pl-6 sm:pl-8 border-l border-border/70 space-y-12">
          
          {steps.map((step) => (
            <div key={step.id} className="relative group">
              {/* Connected Circle Bullet */}
              <div className="absolute -left-[39px] sm:-left-[47px] top-1.5 h-6 w-6 sm:h-8 sm:w-8 rounded-full bg-background border border-border group-hover:border-primary flex items-center justify-center transition-colors duration-200 z-10">
                <step.icon className="h-3.5 w-3.5 sm:h-4 sm:w-4 text-muted-foreground group-hover:text-primary transition-colors duration-200" />
              </div>

              {/* Step content */}
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <span className="text-xl sm:text-2xl font-mono font-bold text-muted-foreground/30">{step.id}</span>
                  <h3 className="text-lg sm:text-xl font-display font-bold text-foreground uppercase tracking-tight">{step.title}</h3>
                </div>

                <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed max-w-2xl font-sans">
                  {step.description}
                </p>

                {/* Detail badges */}
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {step.details.map((detail, i) => (
                    <Badge 
                      key={i} 
                      variant="secondary" 
                      className="px-2 py-0.5 rounded bg-muted/60 text-muted-foreground border border-transparent font-mono text-[9px] hover:border-border transition-all"
                    >
                      {detail}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Under the Hood Pipeline block Diagram */}
      <div className="border-t border-border/60 bg-muted/35 py-20">
        <div className="container mx-auto px-4">
          <div className="text-center max-w-xl mx-auto mb-16 space-y-2">
            <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground">Under the Hood</h2>
            <p className="text-xs sm:text-sm text-muted-foreground">The architecture behind GetReport’s sub-second parsing and AI generation.</p>
          </div>

          {/* Connected Block diagram pipeline */}
          <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-5 gap-6 md:gap-4 items-center">
            
            {/* Stage 1 */}
            <Card className="border border-border/80 bg-card p-5 text-center shadow-premium rounded-xl flex flex-col justify-between h-40">
              <div className="text-[10px] font-mono text-primary font-bold uppercase tracking-wider">01 Client App</div>
              <div className="h-10 w-10 bg-primary/10 text-primary rounded-xl mx-auto flex items-center justify-center my-2 shadow-inner">
                <Code2 className="h-5 w-5" />
              </div>
              <div className="space-y-0.5">
                <span className="block text-xs font-semibold text-foreground">React / Vite UI</span>
                <span className="block text-[10px] text-muted-foreground">Stream files chunk-by-chunk</span>
              </div>
            </Card>

            <div className="hidden md:flex justify-center text-muted-foreground/40 shrink-0">
              <ArrowRight className="h-5 w-5" />
            </div>

            {/* Stage 2 */}
            <Card className="border border-border/80 bg-card p-5 text-center shadow-premium rounded-xl flex flex-col justify-between h-40">
              <div className="text-[10px] font-mono text-primary font-bold uppercase tracking-wider">02 API Gateway</div>
              <div className="h-10 w-10 bg-primary/10 text-primary rounded-xl mx-auto flex items-center justify-center my-2 shadow-inner">
                <Server className="h-5 w-5" />
              </div>
              <div className="space-y-0.5">
                <span className="block text-xs font-semibold text-foreground">FastAPI Router</span>
                <span className="block text-[10px] text-muted-foreground">Secure routes & validation</span>
              </div>
            </Card>

            <div className="hidden md:flex justify-center text-muted-foreground/40 shrink-0">
              <ArrowRight className="h-5 w-5" />
            </div>

            {/* Stage 3 */}
            <Card className="border border-border/80 bg-card p-5 text-center shadow-premium rounded-xl flex flex-col justify-between h-40">
              <div className="text-[10px] font-mono text-primary font-bold uppercase tracking-wider">03 Processing</div>
              <div className="h-10 w-10 bg-primary/10 text-primary rounded-xl mx-auto flex items-center justify-center my-2 shadow-inner">
                <Cpu className="h-5 w-5" />
              </div>
              <div className="space-y-0.5">
                <span className="block text-xs font-semibold text-foreground">Polars Engine</span>
                <span className="block text-[10px] text-muted-foreground">Rust-powered multi-threaded math</span>
              </div>
            </Card>

          </div>
        </div>
      </div>

      {/* CTA Footer */}
      <div className="container mx-auto px-4 pt-24 text-center space-y-6">
        <h2 className="text-2xl sm:text-3xl font-display font-bold text-foreground uppercase tracking-tight">Execute your audit</h2>
        <div className="pt-2">
          <Link to="/">
            <Button size="lg" className="h-11 rounded-xl shadow-premium hover:-translate-y-0.5 active:scale-95 transition-all">
              <span>Ingest File</span>
              <ArrowRight className="ml-1.5 h-4 w-4" />
            </Button>
          </Link>
        </div>
      </div>

    </div>
  );
};

export default HowItWorks;
