import { useState, useEffect } from "react";
import { CheckCircle2, Download, RefreshCw, Loader2, FileText, ChevronRight, AlertTriangle } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { AppStep } from "@/pages/Index";
import type { AnalysisResult, Charts, InsightResult } from "@/types/api";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

interface ReportGenerationProps {
  step: AppStep;
  filename: string;
  analysis: AnalysisResult;
  charts: Charts;
  insights: InsightResult;
  onComplete: () => void;
  onReset: () => void;
}

export const ReportGeneration = ({
  step,
  filename,
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
    // Only trigger if we are in 'generating' step and haven't started yet
    if (step === "generating" && !isGenerating && !downloadUrl) {
      generateReport();
    }
  }, [step]);

  const generateReport = async () => {
    setIsGenerating(true);
    setProgress(10);
    setStatus("Compiling statistical analysis...");

    try {
      // Simulate progress steps for UX while waiting for the single API call
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          // Increment faster at first, then slow down
          return prev + (prev < 50 ? 5 : 2);
        });
      }, 200);

      const blob = await api.generateReport(filename, analysis, charts, insights.insights_text);

      clearInterval(progressInterval);
      setProgress(100);
      setStatus("Report ready for download!");

      // Create object URL for download
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);

      onComplete();

      toast({
        title: "Report Generated Successfully!",
        description: "Your PDF report is ready to download.",
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
          Our AI is analyzing {analysis.metadata.total_rows} rows and finding insights...
        </p>

        <div className="w-full max-w-md mx-auto space-y-2">
          <Progress value={progress} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Analysis</span>
            <span>{Math.round(progress)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};
