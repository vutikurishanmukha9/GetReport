import { useState, useCallback } from "react";
import { Upload, FileSpreadsheet, AlertCircle, X } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { motion, AnimatePresence } from "framer-motion";
import type { ApiResponse, InspectionResult, CleaningRulesMap } from "@/types/api";
import { api } from "@/services/api";
import { DataHealthCheck } from "./DataHealthCheck";

interface FileUploadProps {
  onFileUploaded: (data: ApiResponse) => void;
}

export const FileUpload = ({ onFileUploaded }: FileUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // New States for Interactive Cleaning
  const [taskId, setTaskId] = useState<string | null>(null);
  const [inspectionData, setInspectionData] = useState<InspectionResult | null>(null);
  const [activePoll, setActivePoll] = useState<NodeJS.Timeout | null>(null);

  const { toast } = useToast();

  const validateFile = (file: File): boolean => {
    // ... (validation logic same as before)
    const validTypes = [
      "text/csv",
      "application/vnd.ms-excel",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ];
    const validExtensions = [".csv", ".xlsx", ".xls"];

    const hasValidExtension = validExtensions.some(ext =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!validTypes.includes(file.type) && !hasValidExtension) {
      toast({
        title: "Invalid file type",
        description: "Please upload a CSV or Excel file (.csv, .xlsx, .xls)",
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

  const startPolling = (taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await api.getTaskStatus(taskId);

        // CASE 1: Inspection Ready (State: WAITING_FOR_USER)
        // Note: My backend implementation sets status="WAITING_FOR_USER"
        // Let's verify if I set it explicitly or if I need to rely on message/result structure?
        // I set: title_task_manager.update_status(task_id, "WAITING_FOR_USER", result_payload)
        // So status should be "WAITING_FOR_USER"

        if (status.status === 'WAITING_FOR_USER' && status.result && status.result.stage === 'INSPECTION') {
          clearInterval(pollInterval);
          setInspectionData(status.result as InspectionResult);
          setIsProcessing(false); // Stop spinner, show UI
          toast({
            title: "Data Inspection Complete",
            description: "Please review the issues found.",
          });
          return;
        }

        // CASE 2: Analysis Complete
        if (status.status === 'COMPLETED' && status.result && !status.result.stage) { // If stage is missing, it's final result? Or specific flag?
          // Actually final result has 'info' / 'analysis' keys, while inspection has 'quality_report'.
          // Let's check keys to be safe.
          if ('analysis' in status.result) {
            clearInterval(pollInterval);
            setIsProcessing(false);
            setInspectionData(null); // Clear inspection UI
            onFileUploaded(status.result as ApiResponse);
            toast({
              title: "Analysis Complete!",
              description: `Successfully analyzed ${status.result.info.rows} rows.`,
            });
          }
        }

        // CASE 3: Failure
        else if (status.status === 'FAILED') {
          clearInterval(pollInterval);
          setIsProcessing(false);
          throw new Error(status.error || "Analysis failed");
        }

        // CASE 4: Still Processing
        else {
          // Keep waiting...
        }
      } catch (err: any) {
        clearInterval(pollInterval);
        setIsProcessing(false);
        console.error("Polling error:", err);
        toast({
          title: "Error",
          description: "Connection lost during polling.",
          variant: "destructive",
        });
      }
    }, 1500);
    setActivePoll(pollInterval);
  };

  const processFile = async (file: File) => {
    setIsProcessing(true);
    setSelectedFile(file);
    setInspectionData(null);

    try {
      toast({
        title: "Uploading...",
        description: "Sending file to server...",
      });

      const { task_id } = await api.uploadFile(file);
      setTaskId(task_id);
      startPolling(task_id);

    } catch (error: any) {
      console.error("Error initiating upload:", error);
      let errorMessage = "Could not start upload.";
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      toast({
        title: "Upload Failed",
        description: errorMessage,
        variant: "destructive",
      });
      setIsProcessing(false);
      setSelectedFile(null);
    }
  };

  const handleCleaningRules = async (rules: CleaningRulesMap) => {
    if (!taskId) return;
    setIsProcessing(true); // Restart spinner

    try {
      await api.startAnalysis(taskId, rules);
      // Resume polling for final result
      startPolling(taskId);
    } catch (error) {
      console.error("Failed to start analysis:", error);
      toast({
        title: "Error",
        description: "Failed to apply rules.",
        variant: "destructive"
      });
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
    if (activePoll) clearInterval(activePoll);
  };

  // ─── RENDER: HEALTH CHECK UI ──────────────────────────────────────────────
  if (inspectionData && !isProcessing) {
    return (
      <DataHealthCheck
        report={inspectionData.quality_report}
        onContinue={handleCleaningRules}
        isProcessing={isProcessing}
      />
    );
  }

  // ─── RENDER: UPLOAD UI ────────────────────────────────────────────────────
  return (
    <motion.div
      className="max-w-2xl mx-auto"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <Card className="border-2 border-dashed transition-colors duration-200 hover:border-primary/50">
        <CardContent className="p-0">
          <div
            className={`
              relative p-6 sm:p-8 md:p-12 text-center transition-all duration-200
              ${isDragging ? "bg-primary/5 border-primary" : ""}
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
                mx-auto mb-4 sm:mb-6 flex h-14 w-14 sm:h-16 sm:w-16 md:h-20 md:w-20 items-center justify-center 
                rounded-full transition-all duration-200
                ${isDragging ? "bg-primary/20" : "bg-muted"}
              `}
              animate={{ scale: isDragging ? 1.1 : 1 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <AnimatePresence mode="wait">
                {isProcessing ? (
                  <motion.div
                    key="processing"
                    initial={{ opacity: 0, rotate: 0 }}
                    animate={{ opacity: 1, rotate: 360 }}
                    exit={{ opacity: 0 }}
                    transition={{ rotate: { repeat: Infinity, duration: 1, ease: "linear" } }}
                    className="h-6 w-6 sm:h-7 sm:w-7 md:h-8 md:w-8 rounded-full border-2 border-primary border-t-transparent"
                  />
                ) : selectedFile ? (
                  <motion.div
                    key="file"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.5 }}
                  >
                    <FileSpreadsheet className="h-6 w-6 sm:h-7 sm:w-7 md:h-8 md:w-8 text-primary" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="upload"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  >
                    <Upload className="h-6 w-6 sm:h-7 sm:w-7 md:h-8 md:w-8 text-muted-foreground" />
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Text Content */}
            <AnimatePresence mode="wait">
              {selectedFile ? (
                <motion.div
                  key="selected"
                  className="space-y-2"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <div className="flex items-center justify-center gap-2 text-sm sm:text-base">
                    <span className="font-medium truncate max-w-[200px] sm:max-w-[300px]">
                      {selectedFile.name}
                    </span>
                    {!isProcessing && (
                      <button
                        onClick={clearFile}
                        className="p-1 hover:bg-muted rounded-full transition-colors"
                      >
                        <X className="h-4 w-4 text-muted-foreground" />
                      </button>
                    )}
                  </div>
                  {isProcessing && (
                    <p className="text-sm text-muted-foreground">Processing your file...</p>
                  )}
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  <h3 className="text-base sm:text-lg md:text-xl font-semibold mb-2">
                    {isDragging ? "Drop your file here" : "Upload your dataset"}
                  </h3>
                  <p className="text-sm sm:text-base text-muted-foreground mb-4 sm:mb-6 px-4">
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
                    <Button asChild variant="default" size="lg" className="cursor-pointer">
                      <span className="text-sm sm:text-base">
                        <Upload className="h-4 w-4 mr-2" />
                        Browse Files
                      </span>
                    </Button>
                  </label>

                  {/* Supported formats */}
                  <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3 mt-4 sm:mt-6 text-xs sm:text-sm text-muted-foreground">
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-muted rounded">
                      <FileSpreadsheet className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                      CSV
                    </span>
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-muted rounded">
                      <FileSpreadsheet className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                      XLSX
                    </span>
                    <span className="inline-flex items-center gap-1 px-2 py-1 bg-muted rounded">
                      <FileSpreadsheet className="h-3 w-3 sm:h-3.5 sm:w-3.5" />
                      XLS
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
        className="flex items-start gap-2 sm:gap-3 mt-4 sm:mt-6 p-3 sm:p-4 rounded-lg bg-muted/50 text-xs sm:text-sm text-muted-foreground"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <AlertCircle className="h-4 w-4 sm:h-5 sm:w-5 shrink-0 mt-0.5" />
        <p>
          Your data is processed securely. Files are not stored permanently and are
          automatically deleted after report generation.
        </p>
      </motion.div>
    </motion.div>
  );
};
