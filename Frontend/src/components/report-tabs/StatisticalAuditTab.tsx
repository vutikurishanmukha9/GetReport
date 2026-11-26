import { ArrowRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { AnalysisResult } from "@/types/api";

export function StatisticalAuditTab({ analysis }: { analysis: AnalysisResult }) {
  const hasAdvancedStats = Boolean(analysis.advanced_stats && Object.keys(analysis.advanced_stats).length > 0);
  const allNominal = hasAdvancedStats
    ? Object.values(analysis.advanced_stats!).every(
        (s) => Math.abs(s.skewness) <= 1 && Math.abs(s.kurtosis) <= 3,
      )
    : false;

  return (
    <div className="grid gap-6 md:grid-cols-2">
      {hasAdvancedStats && (
        <Card className="border border-border bg-card shadow-premium rounded-3xl">
          <CardHeader className="border-b border-border pb-3">
            <CardTitle className="text-base font-display font-bold text-foreground">
              Distribution Shape (Moments)
            </CardTitle>
            <CardDescription className="text-xs font-mono text-muted-foreground">
              Skewness (Asymmetry) &amp; Kurtosis (Tail Weight)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[340px] overflow-y-auto text-sm pt-4">
            {Object.entries(analysis.advanced_stats!).map(([col, stats]) => {
              const isSkewed = Math.abs(stats.skewness) > 1;
              const isHeavy = Math.abs(stats.kurtosis) > 3;
              if (!isSkewed && !isHeavy) return null;

              return (
                <div
                  key={col}
                  className="flex justify-between items-center py-2.5 border-b border-border last:border-0 font-mono text-xs"
                >
                  <span className="font-semibold text-foreground truncate max-w-[160px]">{col}</span>
                  <div className="flex gap-1.5">
                    {isSkewed && (
                      <Badge
                        variant="secondary"
                        className="bg-amber-500/10 text-amber-400 border border-amber-500/30 rounded-lg text-[10px]"
                      >
                        skew: {stats.skewness.toFixed(2)}
                      </Badge>
                    )}
                    {isHeavy && (
                      <Badge
                        variant="secondary"
                        className="bg-primary/5 text-primary border border-primary/20 rounded-lg text-[10px]"
                      >
                        kurt: {stats.kurtosis.toFixed(2)}
                      </Badge>
                    )}
                  </div>
                </div>
              );
            })}
            {allNominal && (
              <p className="text-muted-foreground italic text-xs font-mono py-6 text-center">
                All numerical variables display symmetric distributions within nominal bounds.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {analysis.multicollinearity && (
        <Card className="border border-border bg-card shadow-premium rounded-3xl">
          <CardHeader className="border-b border-border pb-3">
            <CardTitle className="text-base font-display font-bold text-foreground">
              Multicollinearity &amp; Redundancy
            </CardTitle>
            <CardDescription className="text-xs font-mono text-muted-foreground">
              Correlated Feature Pairs (VIF Proxy)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[340px] overflow-y-auto text-sm pt-4">
            {analysis.multicollinearity.length > 0 ? (
              analysis.multicollinearity.map((item) => (
                <div
                  key={`mc-${item.features.join("-")}`}
                  className="flex flex-col py-2.5 border-b border-border last:border-0"
                >
                  <div className="flex justify-between font-mono text-xs text-foreground font-semibold">
                    <span className="truncate max-w-[130px]">{item.features[0]}</span>
                    <ArrowRight className="h-3.5 w-3.5 text-muted-foreground mx-2 self-center shrink-0" />
                    <span className="truncate max-w-[130px]">{item.features[1]}</span>
                  </div>
                  <div className="flex justify-between mt-1 text-[11px] font-mono text-muted-foreground">
                    <span>
                      Pearson r: <strong>{item.correlation.toFixed(2)}</strong>
                    </span>
                    <Badge
                      variant="destructive"
                      className="h-4.5 text-[9px] px-2 rounded-full bg-rose-950/40 text-rose-400 border border-rose-500/30 uppercase font-bold"
                    >
                      High Collinearity
                    </Badge>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-muted-foreground italic text-xs font-mono py-6 text-center">
                No collinear feature redundancies detected.
              </p>
            )}
          </CardContent>
        </Card>
      )}

      {analysis.time_series_analysis?.has_time_series && (
        <Card className="md:col-span-2 border border-border bg-card shadow-premium rounded-3xl">
          <CardHeader className="border-b border-border pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-display font-bold text-foreground">
                Time-Series Drift &amp; Temporal Trends
              </CardTitle>
              <Badge variant="outline" className="text-[10px] font-mono bg-muted/20 border-border">
                Time axis: {analysis.time_series_analysis.time_column}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-4">
            <div className="grid gap-3 sm:grid-cols-2 font-mono text-xs">
              {analysis.time_series_analysis.analyses &&
                Object.entries(analysis.time_series_analysis.analyses).map(([col, data]) => (
                  <div key={col} className="p-3 bg-muted/10 border border-border rounded-xl space-y-1.5">
                    <span className="font-bold text-foreground block truncate">{col}</span>
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-muted-foreground">Direction:</span>
                      <Badge
                        variant={data.trend?.direction === "upward" ? "default" : "secondary"}
                        className="text-[10px] px-2 h-4.5 rounded-full"
                      >
                        {data.trend?.direction || "neutral"} ({data.trend?.strength || "stable"})
                      </Badge>
                    </div>
                  </div>
                ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
