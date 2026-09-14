/* Hallmark · component: features · genre: modern-minimal · theme: Quiet
 * macrostructure: Bento Grid knobs: tiles=5, spans=mosaic, border=hairline
 * contrast: pass
 */

import { useState } from "react";
import { 
  SlidersHorizontal, 
  BarChart3, 
  FileText, 
  ShieldCheck, 
  MessageSquareCode,
  CheckCircle2,
  Clock
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export const FeaturesSection = () => {
  // Interactive Ledger state demonstrating core GetReport feature
  const [ledger, setLedger] = useState([
    { id: "1", anomaly: "age (null values)", action: "fill_mean", approved: true },
    { id: "2", anomaly: "revenue (negative values)", action: "drop_row", approved: false },
    { id: "3", anomaly: "join_date (string)", action: "to_datetime", approved: true },
  ]);

  const toggleLedgerItem = (id: string) => {
    setLedger((prev) =>
      prev.map((item) =>
        item.id === id ? { ...item, approved: !item.approved } : item
      )
    );
  };

  const approvedCount = ledger.filter((i) => i.approved).length;
  const healthScore = Math.min(100, Math.round(((11 + approvedCount) / 14) * 100));
  const validityScore = approvedCount === 3 ? "100%" : "89%";

  return (
    <section id="features" className="pt-6 sm:pt-8 md:pt-10 pb-8 sm:pb-10 md:pb-12 bg-muted/20 border-t">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Left-aligned clean Section Header */}
        <div className="max-w-2xl text-left mb-6 sm:mb-8 space-y-2.5">
          <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-foreground">
            Built for visual data transparency
          </h2>
          <p className="text-base sm:text-lg text-muted-foreground">
            GetReport automates complex profiling pipelines without sacrificing auditable user oversight.
          </p>
        </div>

        {/* Bento Grid (F1 Bento Layout) - 5 Tiles in mixed spans */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          
          {/* Tile 1: Auto Data Cleaning (lg:col-span-2 lg:row-span-2) */}
          <Card className="lg:col-span-2 lg:row-span-2 border bg-card flex flex-col justify-between t-card-lift">
            <CardContent className="p-6 sm:p-8 flex flex-col h-full justify-between gap-6">
              <div className="space-y-3 text-left">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/5 text-primary t-spring-press">
                  <SlidersHorizontal className="h-5 w-5" />
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-lg sm:text-xl font-bold">Interactive Data Cleaning</h3>
                  <span className="inline-flex items-center text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-full border border-border/60">
                    Interactive preview
                  </span>
                </div>
                <p className="text-sm sm:text-base text-muted-foreground max-w-md">
                  Inspect identified anomalies, duplicates, and type mismatches in real-time. Approve or reject suggested transformations within the active Issue Ledger.
                </p>
              </div>

              {/* Data Health Dashboard Mockup to fill the gap */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4 mt-auto">
                <div className="bg-emerald-50/50 border border-emerald-100 rounded-xl p-4 flex flex-col justify-between space-y-3 transition-all hover:shadow-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold text-emerald-800 uppercase tracking-wider font-mono">Dataset Health Score</span>
                    <span className="text-xl font-display font-extrabold text-emerald-700 transition-all duration-300">{healthScore}%</span>
                  </div>
                  <div className="w-full bg-emerald-100 rounded-full h-2 overflow-hidden">
                    <div 
                      className="bg-emerald-500 h-full rounded-full transition-all duration-500" 
                      style={{ width: `${healthScore}%` }}
                    />
                  </div>
                  <span className="text-[11px] text-emerald-800/80 font-sans">{11 + approvedCount} of 14 quality tests passed</span>
                </div>
                
                <div className="bg-card border rounded-xl p-4 flex flex-col justify-center space-y-2.5 shadow-xs">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-muted-foreground flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"/>Completeness</span>
                    <span className="text-foreground font-semibold">100%</span>
                  </div>
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-muted-foreground flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"/>Uniqueness</span>
                    <span className="text-foreground font-semibold">100%</span>
                  </div>
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-muted-foreground flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full ${approvedCount === 3 ? "bg-emerald-500" : "bg-amber-500"}`}/>
                      Validity
                    </span>
                    <span className="text-foreground font-semibold transition-all">{validityScore}</span>
                  </div>
                </div>
              </div>

              {/* Pure CSS Mockup of the Issue Ledger with Interactive Row Toggles */}
              <div className="border bg-muted/20 rounded-xl p-4 font-mono text-[11px] sm:text-xs text-muted-foreground space-y-2">
                <div className="flex items-center justify-between text-xs text-foreground font-semibold border-b pb-2">
                  <span>Detected Anomaly</span>
                  <span>Cleaning Action</span>
                  <span>Status (Click to toggle)</span>
                </div>
                {ledger.map((item, idx) => (
                  <div
                    key={item.id}
                    onClick={() => toggleLedgerItem(item.id)}
                    className={`flex items-center justify-between py-1.5 px-1.5 rounded-md cursor-pointer transition-colors hover:bg-muted/50 ${
                      idx < ledger.length - 1 ? "border-b border-dashed" : ""
                    }`}
                    title="Click to toggle status"
                  >
                    <span className="truncate max-w-[130px] sm:max-w-[160px]">{item.anomaly}</span>
                    <span className="text-primary font-semibold">{item.action}</span>
                    <button
                      type="button"
                      className={`text-[11px] font-semibold px-2 py-0.5 rounded transition-all flex items-center gap-1 cursor-pointer ${
                        item.approved
                          ? "text-emerald-700 bg-emerald-500/15 border border-emerald-500/30"
                          : "text-amber-700 bg-amber-500/15 border border-amber-500/30 hover:bg-emerald-500/20"
                      }`}
                    >
                      {item.approved ? (
                        <>
                          <CheckCircle2 className="h-3 w-3 text-emerald-600" />
                          <span>Approved</span>
                        </>
                      ) : (
                        <>
                          <Clock className="h-3 w-3 text-amber-600 shrink-0" />
                          <span>Pending (Click)</span>
                        </>
                      )}
                    </button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Tile 2: Smart Chart Visualization (lg:col-span-1) */}
          <Card className="border bg-card transition-all duration-200 hover:-translate-y-1 hover:border-primary/20 hover:shadow-xs">
            <CardContent className="p-6 flex flex-col justify-between h-full gap-4">
              <div className="space-y-2 text-left">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary">
                  <BarChart3 className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">Automatic Charts</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Infers semantic relationships to generate high-resolution distribution graphs and correlations.
                </p>
              </div>
              
              {/* CSS Mini-Chart Illustration with Tooltip values */}
              <div className="space-y-1.5">
                <div className="flex items-end gap-1.5 h-16 pt-2 px-2 border-b">
                  <div className="w-full bg-primary/10 rounded-t h-6 transition-all hover:bg-primary/30 cursor-pointer" title="0-20% range: 14% density" />
                  <div className="w-full bg-primary/20 rounded-t h-10 transition-all hover:bg-primary/40 cursor-pointer" title="20-40% range: 28% density" />
                  <div className="w-full bg-primary/40 rounded-t h-16 transition-all hover:bg-primary/60 cursor-pointer" title="40-60% range: 52% density" />
                  <div className="w-full bg-primary/60 rounded-t h-12 transition-all hover:bg-primary/75 cursor-pointer" title="60-80% range: 36% density" />
                  <div className="w-full bg-primary/80 rounded-t h-8 transition-all hover:bg-primary cursor-pointer" title="80-100% range: 22% density" />
                </div>
                <div className="flex justify-between text-[9px] font-mono text-muted-foreground/60 px-1">
                  <span>Low</span>
                  <span>Density Distribution</span>
                  <span>High</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tile 3: RAG Insights (lg:col-span-1) */}
          <Card className="border bg-card t-card-lift">
            <CardContent className="p-6 flex flex-col justify-between h-full gap-4">
              <div className="space-y-2 text-left">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary t-spring-press">
                  <MessageSquareCode className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">RAG Insights Chat</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Ask natural language questions about your datasets directly using our vector-indexed chat.
                </p>
              </div>

              {/* RAG Chat mockup */}
              <div className="bg-muted/30 rounded-lg p-2.5 text-[10px] space-y-1 border">
                <div className="text-muted-foreground font-semibold">Q: Find the outlier.</div>
                <div className="text-primary font-mono bg-background p-1.5 rounded shadow-xs leading-normal">
                  Row #42 shows a 12x higher revenue ($150k) than the cluster mean.
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tile 4: Secure Ingestion (lg:col-span-1) */}
          <Card className="border bg-card t-card-lift">
            <CardContent className="p-6 flex flex-col justify-between h-full gap-4">
              <div className="space-y-2 text-left">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary t-spring-press">
                  <ShieldCheck className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">Secure Processing</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Uses strict magic-number signature checks and query sanitization limits to secure ingestion.
                </p>
              </div>
              
              <div className="flex flex-col gap-1.5 font-mono text-[10px] text-emerald-600 bg-emerald-50/50 p-2.5 rounded-lg border border-emerald-500/20">
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3 shrink-0 text-emerald-600" />
                  <span>Magic-byte signature verified</span>
                </div>
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3 shrink-0 text-emerald-600" />
                  <span>Payload size verified (&lt;50MB)</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tile 5: Dual-Engine PDF Pipeline (lg:col-span-2) */}
          <Card className="lg:col-span-2 border bg-card t-card-lift">
            <CardContent className="p-6 sm:p-8 flex flex-col sm:flex-row gap-6 items-start sm:items-center justify-between">
              <div className="space-y-2 text-left max-w-sm">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary t-spring-press">
                  <FileText className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">Dual-Engine PDF Output</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Generate lightweight reports locally with ReportLab, or compile high-fidelity CSS-cached print sheets with WeasyPrint in production.
                </p>
              </div>

              <div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-wider shrink-0 w-full sm:w-auto">
                <div className="flex-1 sm:flex-initial text-center border p-3 rounded-lg bg-muted/20">
                  <span className="block font-bold text-foreground">ReportLab</span>
                  <span className="text-[9px] text-muted-foreground">Local Engine</span>
                </div>
                <span className="text-muted-foreground">→</span>
                <div className="flex-1 sm:flex-initial text-center border border-primary/20 p-3 rounded-lg bg-primary/5">
                  <span className="block font-bold text-primary">WeasyPrint</span>
                  <span className="text-[9px] text-muted-foreground">Prod Engine</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tile 6: ML Readiness Assessment (lg:col-span-3) */}
          <Card className="lg:col-span-3 border bg-card t-card-lift">
            <CardContent className="p-6 sm:p-8 flex flex-col sm:flex-row gap-6 items-start sm:items-center justify-between">
              <div className="space-y-2 text-left max-w-xl">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary t-spring-press">
                  <CheckCircle2 className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">Machine Learning Readiness Assessment</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Determine if your data is ready for model training. Evaluates overall dataset health, flags extreme imbalance, constant features, high nullness rates, and suggests target-specific preprocessing actions.
                </p>
              </div>

              <div className="flex items-center gap-3 font-mono text-[10px] w-full sm:w-auto shrink-0">
                <div className="flex-1 sm:flex-initial text-center border p-3 rounded-lg bg-emerald-500/10 border-emerald-500/20 text-emerald-700">
                  <span className="block font-bold text-xs">85% Score</span>
                  <span className="text-[8px] uppercase tracking-wider">Needs Cleaning</span>
                </div>
              </div>
            </CardContent>
          </Card>

        </div>

      </div>
    </section>
  );
};
