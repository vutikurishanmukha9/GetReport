import { useState, useCallback, useEffect } from "react";
import { Upload, FileSpreadsheet, X, Shield, Lock } from "lucide-react";
import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";
import type { ApiResponse, InspectionResult, CleaningRulesMap } from "@/types/api";
import { api } from "@/services/api";
import { DataHealthCheck } from "./DataHealthCheck";
import { IssueLedger } from "./IssueLedger";
import { ProcessPipeline } from "./ProcessPipeline";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import type { JobTaskResult } from "@/services/api";

function isInspectionResult(res: JobTaskResult | null | undefined): res is InspectionResult {
  return Boolean(res && ("quality_report" in res || "issue_ledger" in res));
}

function isApiResponse(res: JobTaskResult | null | undefined): res is ApiResponse {
  return Boolean(res && ("analysis" in res || "info" in res));
}

const VALID_EXTENSIONS = [
  ".csv", ".xlsx", ".xls", ".parquet", ".json", 
  ".jsonl", ".ndjson", ".tsv", ".feather", ".arrow", ".gz"
] as const;

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface FileUploadProps {
  onFileUploaded: (data: ApiResponse, taskId: string) => void;
}

export const FileUpload = ({ onFileUploaded }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [stagedFiles, setStagedFiles] = useState<File[]>([]);
  
  // Join options for multi-dataset upload
  const [joinKey, setJoinKey] = useState<string>("id");
  const [joinType, setJoinType] = useState<"inner" | "left" | "outer">("inner");

  // New States for Interactive Cleaning
  const [taskId, setTaskId] = useState<string | null>(null);
  const [inspectionData, setInspectionData] = useState<InspectionResult | null>(null);

  // Track what we are waiting for
  const [expectedPhase, setExpectedPhase] = useState<'INSPECTION' | 'ANALYSIS' | null>(null);

  // Real-Time Status Hook
  const { status: taskStatus, progress: taskProgress, message: taskMessage, result: taskResult, error: taskError } = useTaskStatus(taskId || undefined);

  const { toast } = useToast();

  // React to WebSocket Status Updates
  useEffect(() => {
    if (!taskId || !expectedPhase || !taskStatus) return;

    const normalizedStatus = taskStatus.toUpperCase();

    // CASE 1: Inspection Ready (WAITING_FOR_USER or COMPLETED with quality_report)
    if (expectedPhase === 'INSPECTION') {
      if ((normalizedStatus === 'WAITING_FOR_USER' || normalizedStatus === 'COMPLETED' || normalizedStatus === 'SUCCESS') && isInspectionResult(taskResult)) {
        setInspectionData(taskResult);
        setIsProcessing(false);
        setExpectedPhase(null); // Stop waiting
        toast({ title: "Data Inspection Complete", description: "Please review the detected issues found." });
        return;
      }

      // If inspection was bypassed or analysis returned directly
      if ((normalizedStatus === 'COMPLETED' || normalizedStatus === 'SUCCESS') && isApiResponse(taskResult)) {
        setIsProcessing(false);
        setInspectionData(null);
        setExpectedPhase(null);
        onFileUploaded(taskResult, taskId);
        toast({ title: "Analysis Complete!", description: `Successfully analyzed ${taskResult.info?.rows || 0} rows.` });
        return;
      }
    }

    // CASE 2: Analysis Complete
    if (expectedPhase === 'ANALYSIS') {
      if (normalizedStatus === 'COMPLETED' || normalizedStatus === 'SUCCESS' || normalizedStatus === 'DONE') {
        if (isApiResponse(taskResult)) {
          setIsProcessing(false);
          setInspectionData(null);
          setExpectedPhase(null);
          onFileUploaded(taskResult, taskId);
          toast({ title: "Analysis Complete!", description: `Successfully analyzed ${taskResult.info?.rows || 0} rows.` });
          return;
        }
      }
    }

    // CASE 3: Failure
    if (normalizedStatus === 'FAILED' || normalizedStatus === 'ERROR') {
      setIsProcessing(false);
      setExpectedPhase(null);
      toast({ title: "Processing Failed", description: taskError || "An error occurred.", variant: "destructive" });
    }

  }, [taskId, expectedPhase, taskStatus, taskResult, taskError, onFileUploaded, toast]);

  const validateFile = useCallback((file: File): boolean => {
    const hasValidExtension = VALID_EXTENSIONS.some(ext =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExtension) {
      toast({
        title: "Invalid file format",
        description: `Format not supported for '${file.name}'. Supported: CSV, TSV, Excel, Parquet, JSON.`,
        variant: "destructive",
      });
      return false;
    }

    if (file.size > 50 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: `'${file.name}' exceeds the 50MB size limit.`,
        variant: "destructive",
      });
      return false;
    }
    return true;
  }, [toast]);

  const addFilesToStaging = useCallback((files: File[]) => {
    const valid = files.filter(validateFile);
    if (valid.length === 0) return;

    setStagedFiles(prev => {
      const existingNames = new Set(prev.map(f => f.name));
      const uniqueNew = valid.filter(f => !existingNames.has(f.name));
      if (uniqueNew.length < valid.length) {
        toast({ title: "Duplicate file skipped", description: "Some files are already in staging." });
      }
      return [...prev, ...uniqueNew];
    });

    toast({
      title: "File Staged",
      description: `${valid.length} file(s) added to staging. Click 'Start Analysis' when ready!`,
    });
  }, [validateFile, toast]);

  const removeFileFromStaging = (index: number) => {
    setStagedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearStaging = () => {
    setStagedFiles([]);
    setInspectionData(null);
    setExpectedPhase(null);
    setTaskId(null);
  };

  const startAnalysisPipeline = async () => {
    if (stagedFiles.length === 0) return;

    setIsProcessing(true);
    setInspectionData(null);
    let success = false;

    try {
      if (stagedFiles.length === 1) {
        const fileToProcess = stagedFiles[0];
        toast({ title: "Uploading...", description: `Sending '${fileToProcess.name}' to server...` });
        const { task_id } = await api.uploadFile(fileToProcess);
        setTaskId(task_id);
      } else {
        toast({ title: "Joining Datasets...", description: `Merging ${stagedFiles.length} datasets on '${joinKey}' (${joinType} join)...` });
        const { task_id } = await api.uploadJoinedFiles(stagedFiles, joinKey, joinType);
        setTaskId(task_id);
      }
      
      setExpectedPhase('INSPECTION');
      success = true;

    } catch (error: unknown) {
      console.error("Error initiating upload:", error);
      let errorMessage = "Could not start upload.";
      if (error instanceof Error) errorMessage = error.message;
      toast({ title: "Upload Failed", description: errorMessage, variant: "destructive" });
    } finally {
      if (!success) {
        setIsProcessing(false);
      }
    }
  };

  const handleCleaningRules = async (rules: CleaningRulesMap) => {
    if (!taskId) return;
    setIsProcessing(true);
    let success = false;

    try {
      await api.startAnalysis(taskId, rules);
      setExpectedPhase('ANALYSIS'); // Start waiting for analysis
      success = true;
    } catch (error: unknown) {
      console.error("Failed to start analysis:", error);
      let errorMsg = "Failed to apply rules.";
      if (error instanceof Error) errorMsg = error.message;
      toast({ title: "Error", description: errorMsg, variant: "destructive" });
    } finally {
      if (!success) {
        setIsProcessing(false);
      }
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length > 0) {
      addFilesToStaging(droppedFiles);
    }
  }, [addFilesToStaging]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files ? Array.from(e.target.files) : [];
    if (selected.length > 0) {
      addFilesToStaging(selected);
    }
    e.target.value = ""; // Reset input
  };

  // ─── RENDER: HEALTH CHECK + ISSUE LEDGER UI ─────────────────────────────────
  if (inspectionData && !isProcessing) {
    const issueLedgerData = inspectionData.issue_ledger;

    const refreshIssues = async () => {
      if (!taskId) return;
      try {
        const updatedLedger = await api.getIssues(taskId);
        setInspectionData(prev => prev ? { ...prev, issue_ledger: updatedLedger } : prev);
      } catch (e) {
        console.error("Failed to refresh issues:", e);
      }
    };

    return (
      <div className="space-y-8 max-w-7xl mx-auto">
        {issueLedgerData && issueLedgerData.issues && issueLedgerData.issues.length > 0 && (
          <IssueLedger
            taskId={taskId!}
            data={issueLedgerData}
            onRefresh={refreshIssues}
            onProceed={() => handleCleaningRules({})}
          />
        )}

        <DataHealthCheck
          report={inspectionData.quality_report}
          onContinue={handleCleaningRules}
          isProcessing={isProcessing}
        />
      </div>
    );
  }

  // ─── RENDER: UPLOAD UI ────────────────────────────────────────────────────
  return (
    <motion.div
      className="max-w-3xl mx-auto space-y-6"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
    >
      {/* Process Pipeline — shown when processing is active */}
      <AnimatePresence>
        {isProcessing && taskId && (
          <ProcessPipeline
            taskStatus={taskStatus}
            message={taskMessage}
            progress={taskProgress}
            isActive={isProcessing}
          />
        )}
      </AnimatePresence>

      <Card className="border border-border bg-card shadow-premium rounded-2xl overflow-hidden transition-all duration-300 hover:border-border/80 hover:shadow-xl">
        <CardContent className="p-4 sm:p-6 space-y-6">
          {/* Dropzone Container */}
          <div
            className={`
              relative p-6 sm:p-8 md:p-10 text-center transition-all duration-200 min-h-[260px] flex flex-col items-center justify-center
              border-2 border-dashed border-border/60 rounded-xl
              ${isDragging ? "bg-primary/5 border-primary/45" : "bg-muted/10"}
              ${isProcessing ? "opacity-75 pointer-events-none" : ""}
            `}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {/* Upload Icon */}
            <motion.div
              className={`
                mx-auto mb-4 flex h-14 w-14 sm:h-16 sm:w-16 items-center justify-center 
                rounded-full transition-all duration-200 shadow-premium border border-border
                ${isDragging ? "bg-primary/10 border-primary/30 text-primary" : "bg-white text-muted-foreground"}
              `}
              animate={{ scale: isDragging ? 1.08 : 1 }}
              transition={{ type: "spring", stiffness: 260 }}
            >
              <AnimatePresence mode="wait">
                {isProcessing ? (
                  <motion.div
                    key="processing"
                    initial={{ opacity: 0, rotate: 0 }}
                    animate={{ opacity: 1, rotate: 360 }}
                    exit={{ opacity: 0 }}
                    transition={{ rotate: { repeat: Infinity, duration: 1.2, ease: "linear" } }}
                    className="h-6 w-6 sm:h-7 sm:w-7 rounded-full border-2 border-primary border-t-transparent"
                  />
                ) : (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Upload className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Dropzone Text & File Selector */}
            <div className="space-y-2">
              <h3 className="text-lg sm:text-xl font-display font-bold text-foreground tracking-tight">
                {isDragging ? "Drop datasets here" : "Drag & drop files or browse"}
              </h3>
              <p className="text-xs sm:text-sm text-muted-foreground max-w-md font-sans">
                Upload CSV, Excel (.xlsx, .xls), Parquet, TSV, or JSON files. Add as many files as you need before starting analysis.
              </p>
            </div>

            {/* Browse Button */}
            {!isProcessing && (
              <div className="mt-5">
                <label className="inline-block">
                  <input
                    type="file"
                    multiple
                    accept=".csv,.xlsx,.xls,.parquet,.json,.jsonl,.tsv"
                    onChange={handleFileSelect}
                    className="sr-only"
                  />
                  <Button asChild variant="outline" size="default" className="cursor-pointer border-border bg-white text-foreground hover:bg-muted/30 transition-all duration-150 rounded-xl px-5 py-2.5 shadow-sm">
                    <span className="text-xs font-semibold flex items-center gap-2">
                      <Upload className="h-3.5 w-3.5 text-primary" />
                      Browse Files to Stage
                    </span>
                  </Button>
                </label>
              </div>
            )}

            {/* Supported formats */}
            <div className="flex flex-wrap items-center justify-center gap-2 mt-6 text-xs">
              {['.csv', '.xlsx', '.parquet', '.json', '.tsv'].map((fmt) => (
                <span key={fmt} className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-white text-foreground/80 font-mono text-[10px] font-medium rounded-full border border-border/80 shadow-2xs">
                  <FileSpreadsheet className="h-3 w-3 text-primary" />
                  {fmt}
                </span>
              ))}
            </div>
          </div>

          {/* Staged Files Queue Manager */}
          {stagedFiles.length > 0 && (
            <motion.div 
              className="space-y-4 pt-2"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flex items-center justify-between border-b border-border/60 pb-2">
                <div className="flex items-center gap-2">
                  <h4 className="text-xs sm:text-sm font-display font-bold text-foreground uppercase tracking-wide">
                    Staged Datasets
                  </h4>
                  <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary font-mono text-xs font-bold">
                    {stagedFiles.length}
                  </span>
                </div>
                {!isProcessing && (
                  <button 
                    onClick={clearStaging}
                    className="text-xs text-muted-foreground hover:text-destructive transition-colors font-medium"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {/* File Cards List */}
              <div className="space-y-2.5 max-h-[220px] overflow-y-auto pr-1">
                {stagedFiles.map((file, idx) => {
                  const extIndex = file.name.lastIndexOf(".");
                  const ext = extIndex !== -1 ? file.name.slice(extIndex).toLowerCase() : "";
                  return (
                    <motion.div
                      key={`${file.name}-${file.size}-${file.lastModified}`}
                      className="flex items-center justify-between p-3 rounded-xl bg-white border border-border/80 shadow-2xs hover:border-primary/30 t-card-lift"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 10 }}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="p-2 rounded-lg bg-primary/5 text-primary shrink-0">
                          <FileSpreadsheet className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-display font-semibold text-xs sm:text-sm text-foreground truncate max-w-[240px] sm:max-w-[340px]">
                              {file.name}
                            </span>
                            <span className="px-1.5 py-0.2 rounded bg-muted text-[10px] font-mono text-muted-foreground uppercase">
                              {ext}
                            </span>
                          </div>
                          <span className="text-[11px] text-muted-foreground font-mono">
                            {formatFileSize(file.size)}
                          </span>
                        </div>
                      </div>

                      {!isProcessing && (
                        <button
                          onClick={() => removeFileFromStaging(idx)}
                          className="p-1.5 rounded-lg text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors t-spring-press cursor-pointer"
                          title="Remove file from staging"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </motion.div>
                  );
                })}
              </div>

              {/* Join Configuration (Shown when multiple files are staged) */}
              {stagedFiles.length > 1 && !isProcessing && (
                <div className="p-3.5 rounded-xl bg-primary/5 border border-primary/20 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-display font-bold text-primary uppercase tracking-wider">
                      Multi-Dataset Join Configuration
                    </span>
                    <span className="text-[11px] text-muted-foreground">Polars Rust Engine</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div>
                      <label htmlFor="primary-join-key" className="block text-muted-foreground font-medium mb-1">Primary Join Key Column</label>
                      <input
                        id="primary-join-key"
                        type="text"
                        value={joinKey}
                        onChange={(e) => setJoinKey(e.target.value)}
                        placeholder="e.g. id, user_id, date"
                        className="w-full px-3 py-1.5 rounded-lg border border-border bg-white text-foreground focus:outline-none focus:border-primary font-mono text-xs"
                      />
                    </div>
                    <div>
                      <label htmlFor="join-strategy-select" className="block text-muted-foreground font-medium mb-1">Join Strategy</label>
                      <select
                        id="join-strategy-select"
                        aria-label="Join Strategy"
                        value={joinType}
                        // SAFETY: Select element options strictly constrain values to 'inner' | 'left' | 'outer'
                        onChange={(e) => setJoinType(e.target.value as "inner" | "left" | "outer")}
                        className="w-full px-3 py-1.5 rounded-lg border border-border bg-white text-foreground focus:outline-none focus:border-primary font-sans text-xs"
                      >
                        <option value="inner">Inner Join (Matched Rows)</option>
                        <option value="left">Left Join (Keep Base Dataset)</option>
                        <option value="outer">Outer Join (All Rows)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* Manual "Start Analysis" Burgundy CTA Button */}
              {!isProcessing && (
                <div className="pt-2">
                  <Button
                    size="lg"
                    onClick={startAnalysisPipeline}
                    className="w-full rounded-xl shadow-premium bg-primary text-primary-foreground hover:bg-primary/90 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 py-3.5 text-sm font-display font-semibold flex items-center justify-center gap-2"
                  >
                    <FileSpreadsheet className="h-4 w-4" />
                    <span>
                      {stagedFiles.length > 1
                        ? `Start Joined Analysis (${stagedFiles.length} Datasets on '${joinKey}')`
                        : "Start Analysis & Audit (1 Dataset)"}
                    </span>
                  </Button>
                </div>
              )}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Privacy & Data Handling Disclosure */}
      <motion.div
        className="p-5 rounded-2xl border border-border/80 bg-white shadow-premium space-y-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-primary/10 text-primary shrink-0">
            <Shield className="h-4 w-4" />
          </div>
          <h3 className="text-xs sm:text-sm font-display font-bold text-foreground uppercase tracking-wide">
            Your Data, Your Control
          </h3>
        </div>

        <ul className="space-y-2 text-[11px] sm:text-xs text-muted-foreground leading-relaxed font-sans">
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">Session-scoped processing</strong> — Your file is processed for this audit session and is not used for training.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">Controlled retention</strong> — Download the outputs you need before the session expires.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">No Third-Party Sharing</strong> — Raw row-level data is never sent to any external AI provider. Only aggregated statistical summaries are used for insight generation.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">End-to-End Encryption</strong> — All uploads are transmitted over HTTPS / TLS 1.3. Server-side memory is isolated per request.</span>
          </li>
        </ul>

        <div className="pt-2 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Enterprise privacy standards built-in</span>
          <Link to="/privacy" className="text-primary hover:underline font-medium">
            Read Security Architecture &rarr;
          </Link>
        </div>
      </motion.div>
    </motion.div>
  );
};
