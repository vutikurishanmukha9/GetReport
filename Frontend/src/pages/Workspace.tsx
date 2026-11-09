import { useState, lazy, Suspense, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { FileUpload } from "@/components/FileUpload";
import { DataPreview } from "@/components/DataPreview";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import type { ApiResponse } from "@/types/api";
import { ChatInterface } from "@/components/ChatInterface";
import { ProcessPipeline } from "@/components/ProcessPipeline";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import { Loader2 } from "lucide-react";
import { ErrorBoundary } from "@/components/ErrorBoundary";

// Lazy-load heavy ReportGeneration component carrying matplotlib/base64 gallery
const ReportGeneration = lazy(() => 
  import("@/components/ReportGeneration").then(module => ({ default: module.ReportGeneration }))
);

export type AppStep = "upload" | "preview" | "generating" | "complete";

export const Workspace = () => {
  const [searchParams] = useSearchParams();
  const sampleParam = searchParams.get("sample") || undefined;
  const [step, setStep] = useState<AppStep>("upload");
  const [apiData, setApiData] = useState<ApiResponse | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  // Scroll to top on every workspace step transition (upload -> preview -> generating -> complete)
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }, [step]);

  // Pipeline status tracking for generating/complete phases
  const {
    status: pipelineStatus,
    progress: pipelineProgress,
    message: pipelineMessage,
  } = useTaskStatus(
    step === "generating" && taskId ? taskId : undefined
  );

  const handleFileUploaded = useCallback((data: ApiResponse, taskId: string) => {
    setApiData(data);
    setTaskId(taskId);
    setStep("preview");
  }, []);

  const handleGenerateReport = useCallback(() => {
    setStep("generating");
  }, []);

  const handleReportComplete = useCallback(() => {
    setStep("complete");
  }, []);

  const handleReset = useCallback(() => {
    setStep("upload");
    setApiData(null);
    setTaskId(null);
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground transition-colors duration-200 relative">
      {/* Technical Background Grid & Film Grain */}
      <div className="absolute inset-0 bg-[radial-gradient(rgba(0,0,0,0.04)_1px,transparent_1px)] dark:bg-[radial-gradient(rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px] pointer-events-none opacity-50 dark:opacity-40" />
      <div className="hero-grain-overlay" />
      <Header onReset={handleReset} showReset={step !== "upload"} />

      <main className="flex-1 pt-20 sm:pt-24 relative z-10">
        {step === "upload" && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-16 max-w-6xl space-y-8 animate-in fade-in duration-300">
            <div className="text-center space-y-3">
              <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-slate-100 dark:bg-white/[0.04] border border-slate-200 dark:border-white/10 text-slate-600 dark:text-zinc-300 text-[11px] font-mono font-semibold tracking-wide shadow-2xs">
                <span>Ingestion & In-Memory Pipeline</span>
              </div>
              <h1 className="text-3xl sm:text-4xl lg:text-5xl font-display font-black text-slate-950 dark:text-white tracking-[-0.03em] leading-tight">
                Data Ingestion Workspace
              </h1>
              <p className="text-sm sm:text-base text-slate-600 dark:text-muted-foreground max-w-md mx-auto leading-relaxed">
                Ingest your CSV or Excel ledger to begin the automated schema inference, data hygiene audit, and AI synthesis.
              </p>
            </div>
            <FileUpload onFileUploaded={handleFileUploaded} initialSampleId={sampleParam} />
          </div>
        )}

        {step === "preview" && apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <DataPreview
              info={apiData.info}
              cleaningReport={apiData.cleaning_report}
              analysis={apiData.analysis}
              onGenerateReport={handleGenerateReport}
              onBack={handleReset}
            />
          </div>
        )}

        {(step === "generating" || step === "complete") && apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12 space-y-12">
            {/* Process Pipeline: visible during report generation */}
            {step === "generating" && taskId && (
              <div className="max-w-6xl mx-auto">
                <ProcessPipeline
                  taskStatus={pipelineStatus}
                  message={pipelineMessage}
                  progress={pipelineProgress}
                  isActive={true}
                  minCompletedStage={4}
                />
              </div>
            )}

            <ErrorBoundary fallbackTitle="Data Report & Analytics Error">
              <Suspense fallback={
                <div className="flex flex-col items-center justify-center p-12 space-y-4">
                  <Loader2 className="h-8 w-8 text-primary animate-spin" />
                  <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">Loading report module...</span>
                </div>
              }>
                <ReportGeneration
                  step={step}
                  taskId={taskId}
                  filename={apiData.filename}
                  info={apiData.info}
                  analysis={apiData.analysis}
                  charts={apiData.charts}
                  insights={apiData.insights}
                  onComplete={handleReportComplete}
                  onReset={handleReset}
                />
              </Suspense>
            </ErrorBoundary>

            {/* Show Chat Interface ONLY when analysis is complete */}
            {step === "complete" && taskId && (
              <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
                <ChatInterface taskId={taskId} />
              </div>
            )}
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default Workspace;
