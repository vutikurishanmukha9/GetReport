import { useState, useEffect } from "react";
import { 
  CheckCircle2, 
  Download, 
  RotateCcw, 
  Sparkles,
  FileText,
  BarChart3,
  TrendingUp,
  PieChart
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { AppStep, DatasetInfo } from "@/pages/Index";

interface ReportGenerationProps {
  step: AppStep;
  datasetInfo: DatasetInfo;
  onReset: () => void;
}

const generationSteps = [
  { id: 1, label: "Cleaning data", icon: Sparkles },
  { id: 2, label: "Analyzing patterns", icon: TrendingUp },
  { id: 3, label: "Generating charts", icon: BarChart3 },
  { id: 4, label: "Writing insights", icon: FileText },
];

export const ReportGeneration = ({ step, datasetInfo, onReset }: ReportGenerationProps) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (step === "generating") {
      const stepDuration = 750;
      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(progressInterval);
            return 100;
          }
          return prev + 2;
        });
      }, 60);

      const stepInterval = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= generationSteps.length - 1) {
            clearInterval(stepInterval);
            return prev;
          }
          return prev + 1;
        });
      }, stepDuration);

      return () => {
        clearInterval(progressInterval);
        clearInterval(stepInterval);
      };
    }
  }, [step]);

  const handleDownload = () => {
    // Simulate PDF download
    const link = document.createElement("a");
    link.href = "#";
    link.download = `${datasetInfo.fileName.replace(/\.[^/.]+$/, "")}_report.pdf`;
    // In real implementation, this would trigger actual PDF generation
    alert("In a real implementation, this would download your PDF report!");
  };

  if (step === "generating") {
    return (
      <div className="max-w-2xl mx-auto">
        <Card>
          <CardHeader className="text-center pb-4 sm:pb-6">
            <div className="mx-auto mb-4 flex h-14 w-14 sm:h-16 sm:w-16 items-center justify-center rounded-full bg-primary/10">
              <div className="h-6 w-6 sm:h-8 sm:w-8 animate-spin rounded-full border-3 border-primary border-t-transparent" />
            </div>
            <CardTitle className="text-xl sm:text-2xl">Generating Your Report</CardTitle>
            <p className="text-sm sm:text-base text-muted-foreground mt-2">
              Analyzing {datasetInfo.fileName}
            </p>
          </CardHeader>
          <CardContent className="space-y-6 sm:space-y-8">
            {/* Progress Bar */}
            <div className="space-y-2">
              <Progress value={progress} className="h-2 sm:h-3" />
              <p className="text-xs sm:text-sm text-center text-muted-foreground">
                {Math.round(progress)}% complete
              </p>
            </div>

            {/* Steps List */}
            <div className="space-y-3 sm:space-y-4">
              {generationSteps.map((genStep, index) => {
                const isCompleted = index < currentStep;
                const isCurrent = index === currentStep;
                const StepIcon = genStep.icon;

                return (
                  <div
                    key={genStep.id}
                    className={`flex items-center gap-3 sm:gap-4 p-3 sm:p-4 rounded-lg transition-all duration-300 ${
                      isCompleted
                        ? "bg-primary/10"
                        : isCurrent
                        ? "bg-muted"
                        : "opacity-50"
                    }`}
                  >
                    <div
                      className={`flex h-8 w-8 sm:h-10 sm:w-10 shrink-0 items-center justify-center rounded-full transition-colors ${
                        isCompleted
                          ? "bg-primary text-primary-foreground"
                          : isCurrent
                          ? "bg-primary/20 text-primary"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {isCompleted ? (
                        <CheckCircle2 className="h-4 w-4 sm:h-5 sm:w-5" />
                      ) : (
                        <StepIcon className="h-4 w-4 sm:h-5 sm:w-5" />
                      )}
                    </div>
                    <span className={`text-sm sm:text-base font-medium ${
                      isCompleted || isCurrent ? "" : "text-muted-foreground"
                    }`}>
                      {genStep.label}
                    </span>
                    {isCurrent && (
                      <div className="ml-auto">
                        <div className="h-2 w-2 sm:h-2.5 sm:w-2.5 rounded-full bg-primary animate-pulse" />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Completed state
  return (
    <div className="max-w-4xl mx-auto space-y-6 sm:space-y-8">
      {/* Success Header */}
      <Card className="border-primary/20 bg-primary/5">
        <CardContent className="p-6 sm:p-8 md:p-10">
          <div className="flex flex-col items-center text-center">
            <div className="mb-4 sm:mb-6 flex h-14 w-14 sm:h-16 sm:w-16 md:h-20 md:w-20 items-center justify-center rounded-full bg-primary">
              <CheckCircle2 className="h-7 w-7 sm:h-8 sm:w-8 md:h-10 md:w-10 text-primary-foreground" />
            </div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-2 sm:mb-3">
              Report Ready!
            </h1>
            <p className="text-sm sm:text-base md:text-lg text-muted-foreground max-w-md mb-6 sm:mb-8">
              Your comprehensive analytical report has been generated from{" "}
              <span className="font-medium text-foreground">{datasetInfo.fileName}</span>
            </p>
            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 w-full sm:w-auto">
              <Button 
                size="lg" 
                onClick={handleDownload}
                className="gap-2 text-base w-full sm:w-auto"
              >
                <Download className="h-5 w-5" />
                Download PDF Report
              </Button>
              <Button 
                variant="outline" 
                size="lg" 
                onClick={onReset}
                className="gap-2 w-full sm:w-auto"
              >
                <RotateCcw className="h-5 w-5" />
                Analyze Another File
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Report Summary Preview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 sm:p-6 flex flex-col items-center text-center">
            <FileText className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-2 sm:mb-3" />
            <span className="text-xl sm:text-2xl font-bold">12</span>
            <span className="text-xs sm:text-sm text-muted-foreground">Pages</span>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 sm:p-6 flex flex-col items-center text-center">
            <BarChart3 className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-2 sm:mb-3" />
            <span className="text-xl sm:text-2xl font-bold">8</span>
            <span className="text-xs sm:text-sm text-muted-foreground">Charts</span>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 sm:p-6 flex flex-col items-center text-center">
            <Sparkles className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-2 sm:mb-3" />
            <span className="text-xl sm:text-2xl font-bold">15</span>
            <span className="text-xs sm:text-sm text-muted-foreground">Insights</span>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 sm:p-6 flex flex-col items-center text-center">
            <PieChart className="h-6 w-6 sm:h-8 sm:w-8 text-primary mb-2 sm:mb-3" />
            <span className="text-xl sm:text-2xl font-bold">6</span>
            <span className="text-xs sm:text-sm text-muted-foreground">Categories</span>
          </CardContent>
        </Card>
      </div>

      {/* Sample Insights Preview */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg sm:text-xl flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Key Insights Preview
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-3 sm:p-4 rounded-lg bg-muted">
            <p className="text-sm sm:text-base">
              📈 <strong>Sales Trend:</strong> Revenue increased by approximately 40% from Q2 to Q3, 
              reaching the highest level in September.
            </p>
          </div>
          <div className="p-3 sm:p-4 rounded-lg bg-muted">
            <p className="text-sm sm:text-base">
              🏆 <strong>Top Performer:</strong> Widget C generated the highest revenue per unit, 
              outperforming other products by 25%.
            </p>
          </div>
          <div className="p-3 sm:p-4 rounded-lg bg-muted">
            <p className="text-sm sm:text-base">
              🌍 <strong>Regional Analysis:</strong> The North region contributed 35% of total sales, 
              making it the strongest market.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
