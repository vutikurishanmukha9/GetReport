import { useState, useCallback, useEffect } from "react";
import { Upload, FileSpreadsheet, AlertCircle, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";
import type { ApiResponse, InspectionResult, CleaningRulesMap } from "@/types/api";
import { api } from "@/services/api";
import { DataHealthCheck } from "./DataHealthCheck";
import { IssueLedger } from "./IssueLedger";
import { ProcessPipeline } from "./ProcessPipeline";

interface FileUploadProps {
  onFileUploaded: (data: ApiResponse, taskId: string) => void;
}

import { useTaskStatus } from "@/hooks/useTaskStatus"; // Add import

export const FileUpload = ({ onFileUploaded }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

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

    // CASE 1: Inspection Ready
    if (expectedPhase === 'INSPECTION') {
      if (normalizedStatus === 'WAITING_FOR_USER' && taskResult && taskResult.stage === 'INSPECTION') {
        setInspectionData(taskResult as InspectionResult);
        setIsProcessing(false);
        setExpectedPhase(null); // Stop waiting
        toast({ title: "Data Inspection Complete", description: "Please review the issues found." });
      }
    }

    // CASE 2: Analysis Complete
    if (expectedPhase === 'ANALYSIS') {
      if (normalizedStatus === 'COMPLETED') {
        if (taskResult && 'analysis' in taskResult) {
          setIsProcessing(false);
          setInspectionData(null);
          setExpectedPhase(null);
          onFileUploaded(taskResult as ApiResponse, taskId);
          toast({ title: "Analysis Complete!", description: `Successfully analyzed ${taskResult.info.rows} rows.` });
        }
      }
    }

    // CASE 3: Failure
    if (normalizedStatus === 'FAILED') {
      setIsProcessing(false);
      setExpectedPhase(null);
      toast({ title: "Processing Failed", description: taskError || "An error occurred.", variant: "destructive" });
    }

  }, [taskId, expectedPhase, taskStatus, taskResult, taskError, onFileUploaded]);

  const validateFile = (file: File): boolean => {
    const validExtensions = [
      ".csv", ".xlsx", ".xls", ".parquet", ".json", 
      ".jsonl", ".ndjson", ".tsv", ".feather", ".arrow", ".gz"
    ];

    const hasValidExtension = validExtensions.some(ext =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExtension) {
      toast({
        title: "Invalid file format",
        description: "Supported formats: CSV, TSV, Excel (.xlsx, .xls), Parquet, JSON, JSONL, Feather, GZ.",
        variant: "destructive",
      });
      return false;
    }

    if (file.size > 50 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload a file smaller than 50MB",
        variant: "destructive",
      });
      return false;
    }
    return true;
  };

  const processFile = async (file: File) => {
    setIsProcessing(true);
    setSelectedFile(file);
    setInspectionData(null);

    try {
      toast({ title: "Uploading...", description: "Sending file to server..." });

      const { task_id } = await api.uploadFile(file);
      setTaskId(task_id);
      setExpectedPhase('INSPECTION'); // Start waiting for inspection

    } catch (error: any) {
      console.error("Error initiating upload:", error);
      let errorMessage = "Could not start upload.";
      if (error.response?.data?.detail) errorMessage = error.response.data.detail;
      toast({ title: "Upload Failed", description: errorMessage, variant: "destructive" });
      setIsProcessing(false);
      setSelectedFile(null);
    }
  };

  const handleCleaningRules = async (rules: CleaningRulesMap) => {
    if (!taskId) return;
    setIsProcessing(true);

    try {
      await api.startAnalysis(taskId, rules);
      setExpectedPhase('ANALYSIS'); // Start waiting for analysis
    } catch (error: any) {
      console.error("Failed to start analysis:", error);
      const errorMsg = error.response?.data?.message || "Failed to apply rules.";
      toast({ title: "Error", description: errorMsg, variant: "destructive" });
      setIsProcessing(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && validateFile(file)) {
      processFile(file);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      processFile(file);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setInspectionData(null);
    setExpectedPhase(null);
    setTaskId(null); // Disconnect WebSocket
  };

  // ─── RENDER: HEALTH CHECK + ISSUE LEDGER UI ─────────────────────────────────
  if (inspectionData && !isProcessing) {
    const issueLedgerData = inspectionData.issue_ledger;

    // Refresh function to re-fetch the issue ledger with updated statuses
    const refreshIssues = async () => {
      if (!taskId) return;
      try {
        // Fetch the updated issue ledger directly from the issues endpoint
        const updatedLedger = await api.getIssues(taskId);
        // Merge the updated ledger into the current inspectionData
        setInspectionData(prev => prev ? { ...prev, issue_ledger: updatedLedger } : prev);
      } catch (e) {
        console.error("Failed to refresh issues:", e);
      }
    };

    return (
      <div className="space-y-8 max-w-7xl mx-auto">
        {/* Issue Ledger Section */}
        {issueLedgerData && issueLedgerData.issues && issueLedgerData.issues.length > 0 && (
          <IssueLedger
            taskId={taskId!}
            data={issueLedgerData}
            onRefresh={refreshIssues}
            onProceed={() => handleCleaningRules({})}
          />
        )}

        {/* DataHealthCheck for column-level controls */}
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
      className="max-w-2xl mx-auto"
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
        <CardContent className="p-3">
          <div
            className={`
              relative p-6 sm:p-8 md:p-12 text-center transition-all duration-200 min-h-[360px] flex flex-col items-center justify-center
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
                mx-auto mb-4 sm:mb-6 flex h-14 w-14 sm:h-16 sm:w-16 items-center justify-center 
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
                ) : selectedFile ? (
                  <motion.div
                    key="file"
                    initial={{ opacity: 0, scale: 0.7 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.7 }}
                  >
                    <FileSpreadsheet className="h-6 w-6 sm:h-7 sm:w-7 text-primary" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Upload className="h-6 w-6 sm:h-7 sm:w-7" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Text Content */}
            <AnimatePresence mode="wait">
              {selectedFile ? (
                <motion.div
                  key="selected"
                  className="space-y-3"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                >
                  <div className="flex items-center justify-center gap-2 text-sm sm:text-base">
                    <span className="font-display font-medium text-foreground truncate max-w-[200px] sm:max-w-[300px]">
                      {selectedFile.name}
                    </span>
                    {!isProcessing && (
                      <button
                        onClick={clearFile}
                        className="p-1 hover:bg-muted rounded-full transition-colors active:scale-90"
                      >
                        <X className="h-4 w-4 text-muted-foreground hover:text-foreground" />
                      </button>
                    )}
                  </div>
                  {isProcessing && (
                    <p className="text-xs sm:text-sm text-muted-foreground font-mono">processing file…</p>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                >
                  <h3 className="text-lg sm:text-xl font-display font-bold mb-2 text-foreground tracking-tight">
                    {isDragging ? "Drop your file here" : "Upload your dataset"}
                  </h3>
                  <p className="text-sm text-muted-foreground mb-4 sm:mb-6 px-4 font-sans leading-relaxed">
                    Drag and drop your CSV or Excel file, or click to browse
                  </p>

                  {/* Browse Button */}
                  <label className="inline-block">
                    <input
                      type="file"
                      accept=".csv,.xlsx,.xls"
                      onChange={handleFileSelect}
                      className="sr-only"
                    />
                    <Button asChild variant="default" size="lg" className="cursor-pointer shadow-premium transition-all duration-150 hover:-translate-y-0.5 active:scale-95">
                      <span className="text-sm font-medium">
                        <Upload className="h-4 w-4 mr-2" />
                        Browse Files
                      </span>
                    </Button>
                  </label>

                  {/* Supported formats */}
                  <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 mt-5 sm:mt-8 text-xs font-mono">
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-white text-muted-foreground rounded-full border border-border shadow-xs">
                      <FileSpreadsheet className="h-3 w-3" />
                      csv
                    </span>
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-white text-muted-foreground rounded-full border border-border shadow-xs">
                      <FileSpreadsheet className="h-3 w-3" />
                      xlsx
                    </span>
                    <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-white text-muted-foreground rounded-full border border-border shadow-xs">
                      <FileSpreadsheet className="h-3 w-3" />
                      xls
                    </span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </CardContent>
      </Card>

      {/* Info Note */}
      <motion.div
        className="flex items-start gap-2.5 sm:gap-3 mt-4 sm:mt-6 p-3 sm:p-4 rounded-xl border border-border bg-white text-xs sm:text-sm text-muted-foreground leading-relaxed shadow-xs"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        <AlertCircle className="h-4 w-4 sm:h-5 sm:w-5 shrink-0 text-primary mt-0.5" />
        <p>
          Your data is processed securely. Files are not stored permanently and are
          automatically deleted after report generation.
        </p>
      </motion.div>
    </motion.div>
  );
};
