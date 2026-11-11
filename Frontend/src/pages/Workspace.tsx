import { useState, Suspense, useCallback, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { FileUpload } from "@/components/FileUpload";
import { DataPreview } from "@/components/DataPreview";
import { Header } from "@/components/Header";
import { Footer } from "@/components/Footer";
import type { ApiResponse, AppStep } from "@/types/api";
import { ChatInterface } from "@/components/ChatInterface";
import { ProcessPipeline } from "@/components/ProcessPipeline";
import { ReportGeneration } from "@/components/ReportGeneration";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import { Loader2, AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ErrorBoundary } from "@/components/ErrorBoundary";

export type { AppStep };

export const Workspace = () => {
  const [searchParams] = useSearchParams();
  const sampleParam = searchParams.get("sample") || undefined;
  const [step, setStep] = useState<AppStep>("upload");
  const [apiData, setApiData] = useState<ApiResponse | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);

  // Restore active workspace session from sessionStorage on mount
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem("getreport_active_session");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed.apiData && parsed.taskId && parsed.step) {
          setApiData(parsed.apiData);
          setTaskId(parsed.taskId);
          setStep(parsed.step);
        }
      }
    } catch (e) {
      console.warn("Failed to restore active workspace session:", e);
    }
  }, []);

  // Persist session to sessionStorage on step/data updates
  useEffect(() => {
    try {
      if (apiData && taskId && step !== "upload") {
        sessionStorage.setItem(
          "getreport_active_session",
          JSON.stringify({ apiData, taskId, step })
        );
      } else if (step === "upload") {
        sessionStorage.removeItem("getreport_active_session");
      }
    } catch (e) {
      console.warn("Failed to persist active workspace session:", e);
    }
  }, [apiData, taskId, step]);

  // Reset to upload if URL sampleParam changes
  useEffect(() => {
    if (sampleParam && step !== "upload") {
      setStep("upload");
    }
  }, [sampleParam, step]);

  // Scroll to top on every workspace step transition (upload -> preview -> generating -> complete)
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }, [step]);

  // Pipeline status tracking for generating phase
  const {
    status: pipelineStatus,
    progress: pipelineProgress,
    message: pipelineMessage,
  } = useTaskStatus(
    step === "generating" && taskId ? taskId : undefined
  );

  const handleFileUploaded = useCallback((data: ApiResponse, newTaskId: string) => {
    setApiData(data);
    setTaskId(newTaskId);
    setStep("preview");
  }, []);

  const handleGenerateReport = useCallback(() => {
    setStep("generating");
  }, []);

  const handleReportComplete = useCallback(() => {
    setStep("complete");
    if (taskId && apiData) {
      try {
        const historyJson = localStorage.getItem("getreport_audit_history");
        const existing: Array<{
          task_id: string;
          filename: string;
          created_at: string;
          rows?: number;
          columns?: number;
          grade?: string;
          score?: number;
          status: string;
          isDemo?: boolean;
        }> = historyJson ? JSON.parse(historyJson) : [];

        const newItem = {
          task_id: taskId,
          filename: apiData.filename || "Uploaded_Dataset.csv",
          created_at: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }),
          rows: apiData.info?.rows || 0,
          columns: apiData.info?.columns?.length || 0,
          grade: apiData.analysis?.confidence_scores?.dataset_grade || "A",
          score: apiData.analysis?.confidence_scores?.dataset_confidence ?? 90,
          status: "COMPLETED",
          isDemo: false,
        };

        const filtered = existing.filter((item) => item.task_id !== taskId);
        const updated = [newItem, ...filtered].slice(0, 20);
        localStorage.setItem("getreport_audit_history", JSON.stringify(updated));
      } catch (err) {
        console.warn("Failed to persist audit history in localStorage:", err);
      }
    }
  }, [taskId, apiData]);

  // Safety net: transition to complete if pipeline reaches COMPLETED status
  useEffect(() => {
    if (step === "generating" && pipelineStatus === "COMPLETED") {
      const timer = setTimeout(() => {
        setStep((current) => (current === "generating" ? "complete" : current));
      }, 2500);
      return () => clearTimeout(timer);
    }
  }, [step, pipelineStatus]);

  const handleReset = useCallback(() => {
    setStep("upload");
    setApiData(null);
    setTaskId(null);
    try {
      sessionStorage.removeItem("getreport_active_session");
    } catch (e) {
      console.warn("Could not clear active workspace session:", e);
    }
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

        {/* Empty state fallback if session expired or data missing */}
        {step !== "upload" && !apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-16 max-w-xl text-center space-y-5 animate-in fade-in duration-300">
            <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-600 dark:text-amber-400 mx-auto">
              <AlertCircle className="h-6 w-6" />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-xl font-display font-bold text-foreground">No Dataset Loaded</h2>
              <p className="text-sm text-muted-foreground">
                Your dataset session could not be retrieved. Please upload a dataset to begin the audit.
              </p>
            </div>
            <Button onClick={handleReset} variant="outline" className="gap-2 text-xs font-semibold rounded-xl">
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Return to Upload</span>
            </Button>
          </div>
        )}

        {step === "preview" && apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <ErrorBoundary fallbackTitle="Data Preview & Validation Error">
              <DataPreview
                info={apiData.info}
                cleaningReport={apiData.cleaning_report}
                analysis={apiData.analysis}
                onGenerateReport={handleGenerateReport}
                onBack={handleReset}
              />
            </ErrorBoundary>
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
                <ErrorBoundary fallbackTitle="AI Chat Assistant Error">
                  <ChatInterface taskId={taskId} />
                </ErrorBoundary>
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
