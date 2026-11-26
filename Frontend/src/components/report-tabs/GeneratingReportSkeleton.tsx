import { BarChart3, Cpu, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import type { DatasetInfo } from "@/types/api";

export function GeneratingReportSkeleton({
  status,
  progress,
  info,
  filename,
}: {
  status: string;
  progress: number;
  info: DatasetInfo;
  filename: string;
}) {
  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-400">
      <Card className="border border-border/80 bg-card shadow-premium rounded-3xl overflow-hidden">
        <CardContent className="p-8 space-y-5">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-3.5 bg-primary/10 rounded-2xl relative">
                <Loader2 className="h-7 w-7 text-primary animate-spin" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-display font-bold text-foreground">{status}</h3>
                  <Badge
                    variant="outline"
                    className="font-mono text-xs font-bold text-primary bg-primary/5 border-primary/20"
                  >
                    {Math.round(progress)}%
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Compiling audit across{" "}
                  <span className="font-mono text-foreground font-semibold">{info.rows.toLocaleString()}</span> rows and{" "}
                  <span className="font-mono text-foreground font-semibold">{info.columns.length}</span> columns for{" "}
                  <span className="font-mono text-foreground">{filename}</span>
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-1.5 pt-2">
            <div className="h-2.5 w-full bg-secondary rounded-full overflow-hidden border border-border/40">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out rounded-full"
                style={{ width: `${Math.max(10, Math.min(100, progress))}%` }}
              />
            </div>
            <div className="flex justify-between items-center text-[10px] font-mono text-muted-foreground">
              <span>Initializing computation</span>
              <span>Generating Visual Charts &amp; PDF</span>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="md:col-span-1 border border-border/60 bg-card/60 p-5 space-y-3 rounded-2xl">
          <div className="h-4 w-28 bg-muted animate-pulse rounded-md" />
          <div className="h-16 w-full bg-muted/70 animate-pulse rounded-xl" />
          <div className="space-y-2 pt-2">
            <div className="h-3 w-full bg-muted/50 animate-pulse rounded" />
            <div className="h-3 w-3/4 bg-muted/50 animate-pulse rounded" />
          </div>
        </Card>

        <Card className="md:col-span-2 border border-border/60 bg-card/60 p-5 space-y-3 rounded-2xl">
          <div className="flex items-center justify-between">
            <div className="h-4 w-36 bg-muted animate-pulse rounded-md" />
            <div className="h-4 w-16 bg-muted/60 animate-pulse rounded-md" />
          </div>
          <div className="h-40 w-full bg-muted/40 animate-pulse rounded-xl flex items-center justify-center">
            <BarChart3 className="h-10 w-10 text-muted-foreground/30 animate-pulse" />
          </div>
        </Card>
      </div>

      <Card className="border border-border/60 bg-card/60 p-6 space-y-3 rounded-2xl">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-primary/40 animate-pulse" />
          <div className="h-4 w-48 bg-muted animate-pulse rounded-md" />
        </div>
        <div className="space-y-2">
          <div className="h-3.5 w-full bg-muted/60 animate-pulse rounded" />
          <div className="h-3.5 w-5/6 bg-muted/60 animate-pulse rounded" />
          <div className="h-3.5 w-4/6 bg-muted/40 animate-pulse rounded" />
        </div>
      </Card>
    </div>
  );
}
