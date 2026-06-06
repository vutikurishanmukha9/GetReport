/* Hallmark · component: features · genre: modern-minimal · theme: Quiet
 * macrostructure: Bento Grid knobs: tiles=5, spans=mosaic, border=hairline
 * contrast: pass
 */

import { 
  Wand2, 
  BarChart3, 
  FileText, 
  ShieldCheck, 
  MessageSquareCode,
  TrendingUp,
  CheckCircle2
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

export const FeaturesSection = () => {
  return (
    <section id="features" className="py-16 sm:py-20 md:py-24 bg-muted/20 border-t">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 max-w-7xl">
        
        {/* Left-aligned clean Section Header (no eyebrows, no left-right two-col headers) */}
        <div className="max-w-2xl text-left mb-12 sm:mb-16 space-y-3">
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
          <Card className="lg:col-span-2 lg:row-span-2 border bg-card flex flex-col justify-between transition-all duration-200 hover:-translate-y-1 hover:border-primary/20 hover:shadow-xs">
            <CardContent className="p-6 sm:p-8 flex flex-col h-full justify-between gap-6">
              <div className="space-y-3 text-left">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/5 text-primary">
                  <Wand2 className="h-5 w-5" />
                </div>
                <h3 className="text-lg sm:text-xl font-bold">Interactive Data Cleaning</h3>
                <p className="text-sm sm:text-base text-muted-foreground max-w-md">
                  Inspect identified anomalies, duplicates, and type mismatches in real-time. Approve or reject suggested transformations within the active Issue Ledger.
                </p>
              </div>

              {/* Pure CSS Mockup of the Issue Ledger */}
              <div className="border bg-muted/20 rounded-xl p-4 font-mono text-[11px] sm:text-xs text-muted-foreground space-y-2">
                <div className="flex items-center justify-between text-xs text-foreground font-semibold border-b pb-2">
                  <span>Detected Anomaly</span>
                  <span>Cleaning Action</span>
                  <span>Status</span>
                </div>
                <div className="flex items-center justify-between border-b border-dashed pb-1.5 pt-1.5">
                  <span className="truncate max-w-[120px]">age (null values)</span>
                  <span className="text-primary">fill_mean</span>
                  <span className="text-emerald-600 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded">Approved</span>
                </div>
                <div className="flex items-center justify-between border-b border-dashed pb-1.5 pt-1.5">
                  <span className="truncate max-w-[120px]">revenue (negative values)</span>
                  <span className="text-primary">drop_row</span>
                  <span className="text-amber-600 font-semibold bg-amber-500/10 px-1.5 py-0.5 rounded">Pending</span>
                </div>
                <div className="flex items-center justify-between pt-1.5">
                  <span className="truncate max-w-[120px]">join_date (string)</span>
                  <span className="text-primary">to_datetime</span>
                  <span className="text-emerald-600 font-semibold bg-emerald-500/10 px-1.5 py-0.5 rounded">Approved</span>
                </div>
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
              
              {/* CSS Mini-Chart Illustration */}
              <div className="flex items-end gap-1.5 h-16 pt-2 px-2 border-b">
                <div className="w-full bg-primary/10 rounded-t h-6" />
                <div className="w-full bg-primary/20 rounded-t h-10" />
                <div className="w-full bg-primary/40 rounded-t h-16" />
                <div className="w-full bg-primary/60 rounded-t h-12" />
                <div className="w-full bg-primary/80 rounded-t h-8" />
              </div>
            </CardContent>
          </Card>

          {/* Tile 3: RAG Insights (lg:col-span-1) */}
          <Card className="border bg-card transition-all duration-200 hover:-translate-y-1 hover:border-primary/20 hover:shadow-xs">
            <CardContent className="p-6 flex flex-col justify-between h-full gap-4">
              <div className="space-y-2 text-left">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary">
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
          <Card className="border bg-card transition-all duration-200 hover:-translate-y-1 hover:border-primary/20 hover:shadow-xs">
            <CardContent className="p-6 flex flex-col justify-between h-full gap-4">
              <div className="space-y-2 text-left">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary">
                  <ShieldCheck className="h-5 w-5" />
                </div>
                <h3 className="text-base sm:text-lg font-bold">Secure Processing</h3>
                <p className="text-xs sm:text-sm text-muted-foreground">
                  Uses strict magic-number signature checks and query sanitization limits to secure ingestion.
                </p>
              </div>
              
              <div className="flex flex-col gap-1.5 font-mono text-[10px] text-emerald-600 bg-emerald-50/50 p-2.5 rounded-lg border border-emerald-500/20">
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3 shrink-0" /> Signature signature checked
                </div>
                <div className="flex items-center gap-1">
                  <CheckCircle2 className="h-3 w-3 shrink-0" /> Payload size verified (&lt;50MB)
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tile 5: Dual-Engine PDF Pipeline (lg:col-span-2) */}
          <Card className="lg:col-span-2 border bg-card transition-all duration-200 hover:-translate-y-1 hover:border-primary/20 hover:shadow-xs">
            <CardContent className="p-6 sm:p-8 flex flex-col sm:flex-row gap-6 items-start sm:items-center justify-between">
              <div className="space-y-2 text-left max-w-sm">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5 text-primary">
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

        </div>

      </div>
    </section>
  );
};
