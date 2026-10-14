import { useState, useCallback, useEffect, useRef } from "react";
import {
  Upload, FileSpreadsheet, Shield, Lock, Trash2, Database,
  ArrowRightLeft, Link2, Sparkles, AlertCircle
} from "lucide-react";
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
import { SnowflakeConnectModal } from "./SnowflakeConnectModal";
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
  const [joinMode, setJoinMode] = useState<"common" | "asymmetric">("common");
  const [fileHeaders, setFileHeaders] = useState<Record<string, string[]>>({});
  const fileHeadersRef = useRef(fileHeaders);
  fileHeadersRef.current = fileHeaders;
  const [keyMapping, setKeyMapping] = useState<Record<string, string>>({});

  // New States for Interactive Cleaning
  const [taskId, setTaskId] = useState<string | null>(null);
  const [inspectionData, setInspectionData] = useState<InspectionResult | null>(null);
  const [isSnowflakeModalOpen, setIsSnowflakeModalOpen] = useState(false);

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

  // Helper to sniff headers from text files client-side without full server roundtrip
  const sniffHeaders = async (file: File): Promise<string[]> => {
    try {
      const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
      if (ext === ".csv" || ext === ".tsv" || ext === ".txt") {
        const slice = file.slice(0, 32768);
        const text = await slice.text();
        const firstLine = text.split(/\r?\n/)[0];
        if (firstLine) {
          const sep = ext === ".tsv" || firstLine.includes("\t") ? "\t" : (firstLine.includes(";") ? ";" : ",");
          const cols = firstLine
            .split(sep)
            .map(c => c.trim().replace(/^["']|["']$/g, ""))
            .filter(Boolean);
          if (cols.length > 0) return cols;
        }
      }
    } catch (err) {
      console.debug("Client-side header sniff failed for", file.name, err);
    }
    return [];
  };

  // Inspect headers whenever staged files change
  useEffect(() => {
    if (stagedFiles.length === 0) {
      setFileHeaders({});
      setKeyMapping({});
      return;
    }

    let active = true;

    const inspectAll = async () => {
      const needed = stagedFiles.filter(f => !fileHeadersRef.current[f.name]);
      if (needed.length === 0) return;

      const loadedMap: Record<string, string[]> = {};
      const needsBackendInspect: File[] = [];

      for (const f of needed) {
        const cols = await sniffHeaders(f);
        if (cols.length > 0) {
          loadedMap[f.name] = cols;
        } else {
          needsBackendInspect.push(f);
        }
      }

      if (needsBackendInspect.length > 0) {
        try {
          const res = await api.inspectHeaders(needsBackendInspect);
          for (const item of res.files) {
            if (item.columns && item.columns.length > 0) {
              loadedMap[item.filename] = item.columns;
            }
          }
        } catch (err) {
          console.debug("Backend inspectHeaders fallback failed:", err);
        }
      }

      if (active) {
        setFileHeaders(prev => ({ ...prev, ...loadedMap }));
      }
    };

    inspectAll();

    return () => {
      active = false;
    };
  }, [stagedFiles]);

  // Auto-detect join key and foreign-key relationships when file headers are loaded
  useEffect(() => {
    if (stagedFiles.length < 2) return;

    const f1 = stagedFiles[0];
    const f2 = stagedFiles[1];
    const cols1 = fileHeaders[f1.name] || [];
    const cols2 = fileHeaders[f2.name] || [];

    if (cols1.length === 0 || cols2.length === 0) return;

    // 1. Check for exact common column
    const common = cols1.filter(c => cols2.includes(c));
    if (common.length > 0) {
      const preferred = common.find(c => /(_id|^id|key|code)$/i.test(c)) || common[0];
      setJoinKey(preferred);
      setJoinMode("common");
      setKeyMapping({ [f1.name]: preferred, [f2.name]: preferred });
      return;
    }

    // 2. Asymmetric / Foreign Key Detection (e.g. deliveries.match_id = matches.id)
    setJoinMode("asymmetric");

    const name1Clean = f1.name.toLowerCase().replace(/[^a-z0-9]/g, "");
    const name2Clean = f2.name.toLowerCase().replace(/[^a-z0-9]/g, "");

    let left = cols1.find(c => /(_id|^id)$/i.test(c)) || cols1[0];
    let right = cols2.find(c => /(_id|^id)$/i.test(c)) || cols2[0];

    // Check if f1 has foreign key referencing f2 (e.g. deliveries has match_id, matches has id)
    const fk1 = cols1.find(c => {
      const cLow = c.toLowerCase();
      return cLow.endsWith("_id") && name2Clean.includes(cLow.replace("_id", ""));
    });
    if (fk1 && cols2.includes("id")) {
      left = fk1;
      right = "id";
    } else {
      // Check if f2 has foreign key referencing f1
      const fk2 = cols2.find(c => {
        const cLow = c.toLowerCase();
        return cLow.endsWith("_id") && name1Clean.includes(cLow.replace("_id", ""));
      });
      if (fk2 && cols1.includes("id")) {
        left = "id";
        right = fk2;
      }
    }

    setKeyMapping({
      [f1.name]: left,
      [f2.name]: right,
    });
  }, [stagedFiles, fileHeaders]);

  const removeFileFromStaging = (index: number) => {
    const target = stagedFiles[index];
    setStagedFiles(prev => prev.filter((_, i) => i !== index));
    if (target) {
      setFileHeaders(prev => {
        const updated = { ...prev };
        delete updated[target.name];
        return updated;
      });
      setKeyMapping(prev => {
        const updated = { ...prev };
        delete updated[target.name];
        return updated;
      });
    }
  };

  const clearStaging = () => {
    setStagedFiles([]);
    setFileHeaders({});
    setKeyMapping({});
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
        if (joinMode === "asymmetric" && stagedFiles.length === 2) {
          const leftFile = stagedFiles[0];
          const rightFile = stagedFiles[1];
          const leftKey = keyMapping[leftFile.name] || joinKey;
          const rightKey = keyMapping[rightFile.name] || joinKey;

          toast({
            title: "Joining Datasets...",
            description: `Merging '${leftFile.name}' (${leftKey}) and '${rightFile.name}' (${rightKey}) via ${joinType} join...`
          });
          const { task_id } = await api.uploadJoinedFiles(
            stagedFiles,
            undefined,
            joinType,
            leftKey,
            rightKey,
            { [leftFile.name]: leftKey, [rightFile.name]: rightKey }
          );
          setTaskId(task_id);
        } else {
          toast({
            title: "Joining Datasets...",
            description: `Merging ${stagedFiles.length} datasets on '${joinKey}' (${joinType} join)...`
          });
          const { task_id } = await api.uploadJoinedFiles(stagedFiles, joinKey, joinType);
          setTaskId(task_id);
        }
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
      {/* Process Pipeline: shown when processing is active */}
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
                rounded-full transition-all duration-200 shadow-premium border
                ${isDragging ? "bg-primary/20 border-primary/50 text-primary" : "bg-[#121420] border-white/10 text-zinc-300"}
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

            {/* Browse Button & Snowflake Connector */}
            {!isProcessing && (
              <div className="mt-5 flex flex-wrap items-center justify-center gap-3">
                <label className="inline-block">
                  <input
                    type="file"
                    multiple
                    accept=".csv,.xlsx,.xls,.parquet,.json,.jsonl,.tsv"
                    onChange={handleFileSelect}
                    className="sr-only"
                  />
                  <Button asChild variant="outline" size="default" className="cursor-pointer border-white/10 bg-[#121420] text-zinc-100 hover:bg-white/10 transition-colors duration-150 rounded-xl px-5 py-2.5 shadow-sm min-h-[44px]">
                    <span className="text-xs font-semibold flex items-center gap-2">
                      <Upload className="h-3.5 w-3.5 text-primary" />
                      Browse Files to Stage
                    </span>
                  </Button>
                </label>

                <Button
                  type="button"
                  variant="outline"
                  size="default"
                  onClick={() => setIsSnowflakeModalOpen(true)}
                  className="cursor-pointer border-white/10 bg-[#121420] text-zinc-100 hover:bg-white/10 hover:border-white/20 transition-colors duration-150 rounded-xl px-5 py-2.5 shadow-sm min-h-[44px]"
                >
                  <span className="text-xs font-semibold flex items-center gap-2">
                    <Database className="h-3.5 w-3.5 text-blue-400" />
                    <span>Connect Snowflake Warehouse</span>
                    <span className="px-1.5 py-0.5 text-[9px] font-mono font-semibold uppercase tracking-widest text-zinc-400 bg-white/[0.06] border border-white/10 rounded leading-none">
                      Beta
                    </span>
                  </span>
                </Button>
              </div>
            )}

            {/* Supported formats */}
            <div className="flex flex-wrap items-center justify-center gap-2 mt-6 text-xs">
              {['.csv', '.xlsx', '.parquet', '.json', '.tsv'].map((fmt) => (
                <span key={fmt} className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-[#121420] text-zinc-300 font-mono text-[10px] font-medium rounded-full border border-white/10 shadow-2xs">
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
                  <h4 className="text-xs sm:text-sm font-display font-bold text-foreground">
                    Staged Datasets
                  </h4>
                  <span className="px-2 py-0.5 rounded-full bg-primary/10 text-primary font-mono text-xs font-bold">
                    {stagedFiles.length}
                  </span>
                </div>
                {!isProcessing && (
                  <Button 
                    variant="outline"
                    size="sm"
                    onClick={clearStaging}
                    className="h-7 px-2.5 rounded-lg text-xs font-medium text-muted-foreground hover:text-destructive hover:bg-destructive/10 hover:border-destructive/30 border-border/80 gap-1.5 cursor-pointer transition-colors"
                  >
                    <Trash2 className="h-3 w-3" />
                    <span>Clear All</span>
                  </Button>
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
                      className="flex items-center justify-between p-3 rounded-xl bg-[#121420] border border-white/10 shadow-2xs hover:border-white/20 t-card-lift"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 10 }}
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="p-2 rounded-lg bg-primary/10 text-primary shrink-0">
                          <FileSpreadsheet className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-display font-semibold text-xs sm:text-sm text-white truncate max-w-[140px] sm:max-w-[320px]">
                              {file.name}
                            </span>
                            <span className="px-1.5 py-0.2 rounded bg-white/[0.06] text-[10px] font-mono text-zinc-300 uppercase">
                              {ext}
                            </span>
                          </div>
                          <span className="text-[11px] text-zinc-400 font-mono">
                            {formatFileSize(file.size)}
                          </span>
                        </div>
                      </div>

                      {!isProcessing && (
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => removeFileFromStaging(idx)}
                          className="h-7 w-7 rounded-lg text-rose-400 hover:text-white hover:bg-rose-600 border border-transparent hover:border-rose-600 p-0"
                          title="Delete this file from staging"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </motion.div>
                  );
                })}
              </div>

              {/* Join Configuration (Shown when multiple files are staged) */}
              {stagedFiles.length > 1 && !isProcessing && (() => {
                const f1 = stagedFiles[0];
                const f2 = stagedFiles[1];
                const cols1 = fileHeaders[f1?.name] || [];
                const cols2 = fileHeaders[f2?.name] || [];
                const commonCols = stagedFiles.length > 1 && stagedFiles.every(f => (fileHeaders[f.name]?.length || 0) > 0)
                  ? (fileHeaders[stagedFiles[0].name] || []).filter(c =>
                      stagedFiles.every(f => (fileHeaders[f.name] || []).includes(c))
                    )
                  : [];

                const missingInFiles = joinMode === "common" && joinKey.trim()
                  ? stagedFiles
                      .filter(f => {
                        const headers = fileHeaders[f.name];
                        return headers && headers.length > 0 && !headers.includes(joinKey.trim());
                      })
                      .map(f => f.name)
                  : [];

                return (
                  <div className="p-4 rounded-2xl bg-gradient-to-b from-white/[0.04] to-white/[0.01] border border-white/10 shadow-premium space-y-3.5">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] pb-3">
                      <div className="flex items-center gap-2">
                        <div className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                          <ArrowRightLeft className="h-4 w-4" />
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-display font-bold text-white tracking-wide">
                              Multi-Dataset Join Configuration
                            </span>
                            <span className="px-1.5 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 text-[10px] font-mono border border-indigo-500/30">
                              Polars Engine
                            </span>
                          </div>
                          <p className="text-[11px] text-zinc-400">
                            {stagedFiles.length === 2 && joinMode === "asymmetric"
                              ? "Relational cross-dataset key mapping"
                              : `Matching rows across ${stagedFiles.length} staged datasets`}
                          </p>
                        </div>
                      </div>

                      {stagedFiles.length === 2 && (
                        <div className="inline-flex rounded-lg p-0.5 bg-black/40 border border-white/10">
                          <button
                            type="button"
                            onClick={() => setJoinMode("common")}
                            className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all cursor-pointer ${
                              joinMode === "common"
                                ? "bg-white/10 text-white shadow-sm"
                                : "text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            Single Key
                          </button>
                          <button
                            type="button"
                            onClick={() => setJoinMode("asymmetric")}
                            className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all flex items-center gap-1 cursor-pointer ${
                              joinMode === "asymmetric"
                                ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 shadow-sm"
                                : "text-zinc-400 hover:text-zinc-200"
                            }`}
                          >
                            <Sparkles className="h-3 w-3 text-indigo-400" />
                            Key Mapping
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Mode 1: Asymmetric Key Mapping (e.g. deliveries.match_id = matches.id) */}
                    {joinMode === "asymmetric" && stagedFiles.length === 2 ? (
                      <div className="space-y-3">
                        <div className="grid grid-cols-1 md:grid-cols-11 gap-2.5 items-center">
                          {/* Left Dataset Key */}
                          <div className="md:col-span-5 p-3 rounded-xl bg-black/30 border border-white/10 space-y-1.5">
                            <div className="flex items-center justify-between text-[11px]">
                              <span className="text-zinc-200 font-semibold truncate max-w-[170px]" title={f1.name}>
                                {f1.name}
                              </span>
                              <span className="text-[10px] text-zinc-500 font-mono">Left Dataset</span>
                            </div>
                            <label htmlFor="left-join-select" className="text-[10px] text-zinc-400 font-medium block">
                              Join Key Column
                            </label>
                            {cols1.length > 0 ? (
                              <select
                                id="left-join-select"
                                value={keyMapping[f1.name] || ""}
                                onChange={(e) =>
                                  setKeyMapping(prev => ({ ...prev, [f1.name]: e.target.value }))
                                }
                                className="w-full px-2.5 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs cursor-pointer"
                              >
                                {cols1.map(col => (
                                  <option key={col} value={col}>
                                    {col}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                id="left-join-select"
                                type="text"
                                value={keyMapping[f1.name] || ""}
                                onChange={(e) =>
                                  setKeyMapping(prev => ({ ...prev, [f1.name]: e.target.value }))
                                }
                                placeholder="e.g. match_id"
                                className="w-full px-2.5 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs"
                              />
                            )}
                          </div>

                          {/* Linking Center Indicator */}
                          <div className="md:col-span-1 flex flex-col items-center justify-center py-1">
                            <div className="h-7 w-7 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
                              <Link2 className="h-3.5 w-3.5" />
                            </div>
                          </div>

                          {/* Right Dataset Key */}
                          <div className="md:col-span-5 p-3 rounded-xl bg-black/30 border border-white/10 space-y-1.5">
                            <div className="flex items-center justify-between text-[11px]">
                              <span className="text-zinc-200 font-semibold truncate max-w-[170px]" title={f2.name}>
                                {f2.name}
                              </span>
                              <span className="text-[10px] text-zinc-500 font-mono">Right Dataset</span>
                            </div>
                            <label htmlFor="right-join-select" className="text-[10px] text-zinc-400 font-medium block">
                              Matching Key Column
                            </label>
                            {cols2.length > 0 ? (
                              <select
                                id="right-join-select"
                                value={keyMapping[f2.name] || ""}
                                onChange={(e) =>
                                  setKeyMapping(prev => ({ ...prev, [f2.name]: e.target.value }))
                                }
                                className="w-full px-2.5 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs cursor-pointer"
                              >
                                {cols2.map(col => (
                                  <option key={col} value={col}>
                                    {col}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                id="right-join-select"
                                type="text"
                                value={keyMapping[f2.name] || ""}
                                onChange={(e) =>
                                  setKeyMapping(prev => ({ ...prev, [f2.name]: e.target.value }))
                                }
                                placeholder="e.g. id"
                                className="w-full px-2.5 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs"
                              />
                            )}
                          </div>
                        </div>

                        {/* Join Strategy & Summary Banner */}
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
                          <div>
                            <label htmlFor="join-strategy-select" className="block text-zinc-400 font-medium mb-1 text-xs">
                              Join Strategy
                            </label>
                            <select
                              id="join-strategy-select"
                              aria-label="Join Strategy"
                              value={joinType}
                              onChange={(e) => setJoinType(e.target.value as "inner" | "left" | "outer")}
                              className="w-full px-3 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-sans text-xs cursor-pointer"
                            >
                              <option value="inner">Inner Join (Matched Rows Only)</option>
                              <option value="left">Left Join (Keep All in {f1.name})</option>
                              <option value="outer">Full Outer Join (Keep All Rows)</option>
                            </select>
                          </div>
                          <div className="flex items-end">
                            <div className="w-full p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[11px] text-indigo-300 flex items-center gap-1.5">
                              <Sparkles className="h-3.5 w-3.5 text-indigo-400 shrink-0" />
                              <span className="truncate">
                                Linking <strong className="text-white font-mono">{f1.name}.{keyMapping[f1.name] || '?'}</strong> = <strong className="text-white font-mono">{f2.name}.{keyMapping[f2.name] || '?'}</strong>
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* Mode 2: Single Key */
                      <div className="space-y-2">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                          <div>
                            <label htmlFor="primary-join-key" className="block text-zinc-400 font-medium mb-1">
                              Primary Join Key Column
                            </label>
                            {commonCols.length > 0 ? (
                              <select
                                id="primary-join-key"
                                value={joinKey}
                                onChange={(e) => setJoinKey(e.target.value)}
                                className="w-full px-3 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs cursor-pointer"
                              >
                                {commonCols.map(col => (
                                  <option key={col} value={col}>
                                    {col} (Found in all {stagedFiles.length} files)
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <input
                                id="primary-join-key"
                                type="text"
                                value={joinKey}
                                onChange={(e) => setJoinKey(e.target.value)}
                                placeholder="e.g. id, match_id, user_id"
                                className="w-full px-3 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-mono text-xs"
                              />
                            )}
                          </div>
                          <div>
                            <label htmlFor="join-strategy-select-common" className="block text-zinc-400 font-medium mb-1">
                              Join Strategy
                            </label>
                            <select
                              id="join-strategy-select-common"
                              aria-label="Join Strategy"
                              value={joinType}
                              onChange={(e) => setJoinType(e.target.value as "inner" | "left" | "outer")}
                              className="w-full px-3 py-1.5 rounded-lg border border-white/10 bg-[#0d0f17] text-white focus:outline-none focus:border-indigo-500 font-sans text-xs cursor-pointer"
                            >
                              <option value="inner">Inner Join (Matched Rows)</option>
                              <option value="left">Left Join (Keep Base Dataset)</option>
                              <option value="outer">Outer Join (All Rows)</option>
                            </select>
                          </div>
                        </div>

                        {/* Missing key warning banner */}
                        {missingInFiles.length > 0 && (
                          <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/25 text-amber-300 text-xs flex items-center justify-between gap-2">
                            <div className="flex items-center gap-1.5">
                              <AlertCircle className="h-4 w-4 shrink-0 text-amber-400" />
                              <span>
                                Key <strong className="text-white font-mono">'{joinKey}'</strong> is not present in <strong>{missingInFiles.join(", ")}</strong>.
                              </span>
                            </div>
                            {stagedFiles.length === 2 && (
                              <button
                                type="button"
                                onClick={() => setJoinMode("asymmetric")}
                                className="text-xs font-semibold text-amber-200 hover:text-white underline cursor-pointer shrink-0"
                              >
                                Switch to Key Mapping
                              </button>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}

              {/* Manual "Start Analysis" CTA Button */}
              {!isProcessing && (() => {
                const f1 = stagedFiles[0];
                const f2 = stagedFiles[1];
                const isAsym = joinMode === "asymmetric" && stagedFiles.length === 2;
                const missingInFiles = joinMode === "common" && joinKey.trim()
                  ? stagedFiles
                      .filter(f => {
                        const headers = fileHeaders[f.name];
                        return headers && headers.length > 0 && !headers.includes(joinKey.trim());
                      })
                      .map(f => f.name)
                  : [];

                const isDisabled = stagedFiles.length > 1 && (
                  isAsym
                    ? !keyMapping[f1?.name] || !keyMapping[f2?.name]
                    : !joinKey.trim() || missingInFiles.length > 0
                );

                const buttonLabel = stagedFiles.length > 1
                  ? isAsym && f1 && f2
                    ? `Start Joined Analysis (${f1.name}.${keyMapping[f1.name] || '?'} = ${f2.name}.${keyMapping[f2.name] || '?'})`
                    : `Start Joined Analysis (${stagedFiles.length} Datasets on '${joinKey}')`
                  : "Start Analysis & Audit (1 Dataset)";

                return (
                  <div className="pt-2">
                    <button
                      onClick={startAnalysisPipeline}
                      disabled={isDisabled}
                      className={`w-full min-h-[48px] h-12 rounded-full font-sans font-bold text-sm tracking-tight transition-all duration-150 flex items-center justify-center gap-2 cursor-pointer ${
                        isDisabled
                          ? "bg-zinc-800 text-zinc-500 border border-zinc-700/50 cursor-not-allowed"
                          : "bg-white hover:bg-zinc-100 text-black shadow-[0_2px_16px_rgba(255,255,255,0.2),inset_0_1px_0_rgba(255,255,255,1)] border border-white hover:-translate-y-0.5 active:scale-95"
                      }`}
                    >
                      <FileSpreadsheet className={`h-4 w-4 ${isDisabled ? "text-zinc-500" : "text-black"}`} />
                      <span className="truncate px-2">{buttonLabel}</span>
                    </button>
                  </div>
                );
              })()}
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Privacy & Data Handling Disclosure */}
      <motion.div
        className="p-5 rounded-2xl border border-white/10 bg-[#0c0d14] shadow-premium space-y-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-primary/10 text-primary shrink-0">
            <Shield className="h-4 w-4" />
          </div>
          <h3 className="text-xs sm:text-sm font-display font-bold text-foreground">
            Your Data, Your Control
          </h3>
        </div>

        <ul className="space-y-2 text-[11px] sm:text-xs text-muted-foreground leading-relaxed font-sans">
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">Session-scoped processing:</strong> Your file is processed for this audit session and is not used for training.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">Controlled retention:</strong> Download the outputs you need before the session expires.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">No Third-Party Sharing:</strong> Raw row-level data is never sent to any external AI provider. Only aggregated statistical summaries are used for insight generation.</span>
          </li>
          <li className="flex items-start gap-2">
            <Lock className="h-3.5 w-3.5 text-primary/70 mt-0.5 shrink-0" />
            <span><strong className="text-foreground">End-to-End Encryption:</strong> All uploads are transmitted over HTTPS / TLS 1.3. Server-side memory is isolated per request.</span>
          </li>
        </ul>

        <div className="pt-2 border-t border-white/10 flex items-center justify-between text-[11px] text-muted-foreground">
          <span>Enterprise privacy standards built-in</span>
          <Link to="/privacy" className="text-primary hover:underline font-medium">
            Read Security Architecture &rarr;
          </Link>
        </div>
      </motion.div>

      {/* Snowflake Data Warehouse Integration Modal */}
      <SnowflakeConnectModal
        isOpen={isSnowflakeModalOpen}
        onClose={() => setIsSnowflakeModalOpen(false)}
        onImportStarted={(importedTaskId) => {
          setTaskId(importedTaskId);
          setIsProcessing(true);
          setExpectedPhase('INSPECTION');
          toast({
            title: "Snowflake Ingestion Active",
            description: "Streaming Snowflake records into the inspection audit engine...",
          });
        }}
      />
    </motion.div>
  );
};
