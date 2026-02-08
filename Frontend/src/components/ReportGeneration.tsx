import { useState, useEffect } from "react";
import { CheckCircle2, Download, RefreshCw, Loader2, FileText, ChevronRight, AlertTriangle, ArrowRight } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import type { AppStep } from "@/pages/Index";
import type { AnalysisResult, Charts, InsightResult, DatasetInfo } from "@/types/api";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

interface ReportGenerationProps {
  step: AppStep;
  taskId: string | null;
  filename: string;
  info: DatasetInfo;
  analysis: AnalysisResult;
  charts: Charts;
  insights: InsightResult;
  onComplete: () => void;
  onReset: () => void;
}

export const ReportGeneration = ({
  step,
  taskId,
  filename,
  info,
  analysis,
  charts,
  insights,
  onComplete,
  onReset
}: ReportGenerationProps) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing report engine...");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    if (step === "generating" && !isGenerating && !downloadUrl) {
      if (!taskId) {
        console.error("No Task ID found for generation.");
        return;
      }
      generateReport(taskId);
    }
  }, [step, taskId]);

  const generateReport = async (tid: string) => {
    setIsGenerating(true);
    setProgress(10);
    setStatus("Compiling statistical analysis on server...");

    try {
      // 1. Trigger Server-Side Generation (Secure)
      await api.generatePersistentReport(tid);
      setStatus("Downloading PDF...");
      setProgress(80);

      // 2. Fetch the generated Blob
      const blob = await api.downloadReportBlob(tid);

      setStatus("Report ready for download!");
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);

      onComplete();

      toast({
        title: "Report Generated Successfully!",
        description: "Your secure PDF report is ready.",
      });

    } catch (error) {
      console.error("Report generation failed:", error);
      setStatus("Failed to generate report.");
      toast({
        title: "Generation Failed",
        description: "Could not generate PDF report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadFile = () => {
    if (downloadUrl) {
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.setAttribute('download', `Report_${filename}_${new Date().getTime()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    }
  };

  if (step === "complete") {
    return (
      <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in zoom-in duration-500">

        {/* ─── Statistical Deep Dive (New) ─── */}
        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-6 duration-700 delay-300">
          <h3 className="text-xl font-semibold border-b pb-2">Statistical Deep Dive (Blunt Verdict)</h3>

          <div className="grid gap-4 md:grid-cols-2">
            {/* Skewness & Kurtosis */}
            {analysis.advanced_stats && Object.keys(analysis.advanced_stats).length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Distribution Shape</CardTitle>
                  <CardDescription>Skewness & Kurtosis (Rule #5)</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 max-h-[300px] overflow-y-auto text-sm">
                  {Object.entries(analysis.advanced_stats).map(([col, stats]) => {
                    const isSkewed = Math.abs(stats.skewness) > 1;
                    const isHeavy = Math.abs(stats.kurtosis) > 3;
                    if (!isSkewed && !isHeavy) return null;

                    return (
                      <div key={col} className="flex justify-between items-center py-1 border-b last:border-0">
                        <span className="font-medium">{col}</span>
                        <div className="flex gap-2">
                          {isSkewed && (
                            <Badge variant="secondary" className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100">
                              Skew: {stats.skewness.toFixed(2)}
                            </Badge>
                          )}
                          {isHeavy && (
                            <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-100">
                              Kurt: {stats.kurtosis.toFixed(2)}
                            </Badge>
                          )}
                        </div>
                      </div>
                    );
                  })}
                  {Object.values(analysis.advanced_stats).every(s => Math.abs(s.skewness) <= 1 && Math.abs(s.kurtosis) <= 3) && (
                    <p className="text-muted-foreground italic">All numeric columns appear normally distributed.</p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Multicollinearity */}
            {analysis.multicollinearity && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Multicollinearity (VIF Proxy)</CardTitle>
                  <CardDescription>Highly Correlated Features (Rule #12)</CardDescription>
                </CardHeader>
                <CardContent className="space-y-2 max-h-[300px] overflow-y-auto text-sm">
                  {analysis.multicollinearity.length > 0 ? (
                    analysis.multicollinearity.map((item, i) => (
                      <div key={i} className="flex flex-col py-2 border-b last:border-0">
                        <div className="flex justify-between font-medium">
                          <span>{item.features[0]}</span>
                          <ArrowRight className="h-4 w-4 text-muted-foreground mx-2" />
                          <span>{item.features[1]}</span>
                        </div>
                        <div className="flex justify-between mt-1 text-xs text-muted-foreground">
                          <span>Correlation: <strong>{item.correlation.toFixed(2)}</strong></span>
                          <Badge variant="destructive" className="h-5">High Redundancy</Badge>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="text-muted-foreground italic">No redundant features detected.</p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Time-Series Analysis (Enhanced Tier 1) */}
            {analysis.time_series_analysis?.has_time_series && (
              <Card className="md:col-span-2 border-l-4 border-blue-500">
                <CardHeader>
                  <CardTitle className="text-base flex justify-between items-center">
                    <span>Time-Series Analysis (Trend & Seasonality)</span>
                    <Badge variant="outline">
                      Time Column: {analysis.time_series_analysis.time_column}
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    Detected trends and seasonal patterns in your time-series data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-2">
                    {analysis.time_series_analysis.analyses && Object.entries(analysis.time_series_analysis.analyses).map(([col, data]: [string, any]) => (
                      <div key={col} className="p-3 bg-muted/50 rounded-lg">
                        <p className="font-medium text-sm mb-2">{col}</p>
                        <div className="space-y-2 text-xs">
                          {/* Trend */}
                          {data.trend?.detected ? (
                            <div className="flex justify-between items-center">
                              <span>Trend:</span>
                              <Badge
                                variant={data.trend.direction === "upward" ? "default" : data.trend.direction === "downward" ? "destructive" : "secondary"}
                                className="text-xs"
                              >
                                {data.trend.direction} ({data.trend.strength})
                              </Badge>
                            </div>
                          ) : (
                            <div className="flex justify-between items-center text-muted-foreground">
                              <span>Trend:</span>
                              <span>Not detected</span>
                            </div>
                          )}
                          {/* Seasonality */}
                          {data.seasonality?.detected ? (
                            <div className="flex justify-between items-center">
                              <span>Seasonality:</span>
                              <Badge variant="outline" className="text-xs">
                                {data.seasonality.primary_pattern}
                              </Badge>
                            </div>
                          ) : (
                            <div className="flex justify-between items-center text-muted-foreground">
                              <span>Seasonality:</span>
                              <span>Not detected</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Legacy Time-Series (backward compatibility) */}
            {analysis.time_series_analysis?.is_sorted !== undefined && (
              <Card className="md:col-span-2 border-l-4 border-blue-500">
                <CardHeader>
                  <CardTitle className="text-base flex justify-between items-center">
                    <span>Time-Series Integrity (Rule #13)</span>
                    <Badge variant={analysis.time_series_analysis.is_sorted ? "outline" : "destructive"}>
                      {analysis.time_series_analysis.is_sorted ? "Ordered Chronologically" : "Not Sorted By Time"}
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    Primary Time Column: <span className="font-mono text-primary">{analysis.time_series_analysis.primary_time_col}</span>
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    {analysis.time_series_analysis.drift_detected?.length > 0 ? (
                      <div className="space-y-2">
                        <p className="font-medium text-amber-700">Warning: Conceptual Drift Detected (&gt;30% Shift):</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {analysis.time_series_analysis.drift_detected.map((d: any, i: number) => (
                            <div key={i} className="bg-amber-50 p-2 rounded border border-amber-200 flex justify-between items-center">
                              <span className="font-semibold">{d.column}</span>
                              <div className="text-xs text-right">
                                <span className="block text-amber-800 font-bold">Swap: {d.shift_pct}%</span>
                                <span className="text-muted-foreground">{d.mean_p1} → {d.mean_p2}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 text-green-600">
                        <CheckCircle2 className="h-4 w-4" />
                        <span>No significant mean drift detected between first and second half of time window.</span>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center h-20 w-20 rounded-full bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 mb-4">
            <CheckCircle2 className="h-10 w-10" />
          </div>
          <h2 className="text-3xl font-bold tracking-tight">Report Ready!</h2>
          <p className="text-muted-foreground text-lg max-w-lg mx-auto">
            Your detailed analysis for <strong>{filename}</strong> has been successfully generated.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card className="border-2 border-primary/10 hover:border-primary/30 transition-colors">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Download className="h-5 w-5 text-primary" />
                Download PDF
              </CardTitle>
              <CardDescription>Get the full professional PDF report.</CardDescription>
            </CardHeader>
            <CardContent>
              <Button size="lg" className="w-full" onClick={downloadFile}>
                Download Report
                <ChevronRight className="ml-2 h-4 w-4" />
              </Button>
            </CardContent>
          </Card>

          <Card className="hover:bg-muted/50 transition-colors">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <RefreshCw className="h-5 w-5 text-muted-foreground" />
                Analyze Another
              </CardTitle>
              <CardDescription>Start fresh with a new dataset.</CardDescription>
            </CardHeader>
            <CardContent>
              <Button size="lg" variant="outline" className="w-full" onClick={onReset}>
                Start New Analysis
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Debug / Fallback info if download fails */}
        {!downloadUrl && !isGenerating && (
          <div className="p-4 rounded-lg bg-orange-50 dark:bg-orange-950/20 text-orange-600 text-sm flex items-center justify-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            <span>If the download didn't prepare correctly, try clicking "Generate Report" again.</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="max-w-xl mx-auto mt-12 text-center space-y-8 animate-in fade-in slide-in-from-bottom-5 duration-500">
      <div className="relative">
        <div className="absolute inset-0 flex items-center justify-center">
          <Loader2 className="h-32 w-32 animate-spin text-primary/10" />
        </div>
        <div className="relative z-10 bg-background/80 backdrop-blur-sm rounded-full p-8 inline-block">
          <FileText className="h-16 w-16 text-primary animate-pulse" />
        </div>
      </div>

      <div className="space-y-4">
        <h3 className="text-2xl font-semibold">{status}</h3>
        <p className="text-muted-foreground">
          Our AI is analyzing {info.rows.toLocaleString()} rows and finding insights...
        </p>

        <div className="w-full max-w-md mx-auto space-y-2">
          <div className="w-full max-w-md mx-auto space-y-2">
            {/* Indeterminate loader for honest feedback */}
            <div className="h-2 w-full overflow-hidden rounded-full bg-secondary">
              <div className="h-full w-full flex-1 bg-primary animate-indeterminate-progress" style={{ transformOrigin: "0% 50%" }}></div>
            </div>
            <p className="text-xs text-muted-foreground text-center">
              Rendering high-quality PDF...
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
