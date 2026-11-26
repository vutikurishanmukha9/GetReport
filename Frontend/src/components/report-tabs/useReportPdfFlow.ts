import { useEffect, useRef, useState } from "react";
import { api } from "@/services/api";
import type { AppStep } from "@/pages/Workspace";
import type { ExportFormat } from "@/types/api";
import type { HiPlotPayload } from "../HiPlotParallelCoordinates";

interface ToastFn {
  (opts: { title: string; description: string; variant?: "default" | "destructive" }): void;
}

interface ReportPipelineCallbacks {
  isCancelled: () => boolean;
  onStatus: (status: string) => void;
  onProgress: (value: number | ((prev: number) => number)) => void;
  onDownloadUrl: (url: string) => void;
  onComplete: () => void;
  onError: (errMsg: string) => void;
}

function delayMs(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function pollReportReady(
  taskId: string,
  attempt: number,
  callbacks: ReportPipelineCallbacks,
): Promise<void> {
  if (callbacks.isCancelled()) return Promise.resolve();
  if (attempt >= 60) {
    return Promise.reject(new Error("Timed out waiting for PDF compilation."));
  }
  return delayMs(1500)
    .then(() => {
      if (callbacks.isCancelled()) return null;
      return api.getReportStatus(taskId);
    })
    .then((repStatus) => {
      if (!repStatus || callbacks.isCancelled()) return undefined;
      if (repStatus.status === "ready") return undefined;
      if (repStatus.status === "failed") {
        return Promise.reject(new Error("Server report compilation failed."));
      }
      callbacks.onProgress((prev) => Math.min(92, Math.max(prev, 30 + attempt * 2)));
      return pollReportReady(taskId, attempt + 1, callbacks);
    });
}

function runReportCompilationPipeline(
  taskId: string,
  callbacks: ReportPipelineCallbacks,
): void {
  callbacks.onStatus("Checking compiled report cache…");
  callbacks.onProgress(15);

  api
    .getReportStatus(taskId)
    .then((initialStatus) => {
      if (callbacks.isCancelled()) return null;
      if (initialStatus.status === "ready") return true;
      callbacks.onStatus("Compiling publication PDF report…");
      callbacks.onProgress(30);
      return api
        .generatePersistentReport(taskId)
        .then(() => pollReportReady(taskId, 0, callbacks))
        .then(() => true);
    })
    .then((shouldDownload) => {
      if (!shouldDownload || callbacks.isCancelled()) return null;
      callbacks.onStatus("Downloading PDF…");
      callbacks.onProgress(96);
      return api.downloadReportBlob(taskId);
    })
    .then((blob) => {
      if (!blob || callbacks.isCancelled()) return;
      const url = window.URL.createObjectURL(blob);
      callbacks.onDownloadUrl(url);
      callbacks.onProgress(100);
      callbacks.onStatus("Report ready!");
      callbacks.onComplete();
    })
    .catch((err) => {
      if (callbacks.isCancelled()) return;
      console.error("Report generation failed:", err);
      const errMsg = err instanceof Error ? err.message : "Could not generate PDF report.";
      callbacks.onError(errMsg);
    });
}

export function useReportPdfFlow({
  step,
  taskId,
  filename,
  onComplete,
  onReset,
  toast,
}: {
  step: AppStep;
  taskId: string | null;
  filename: string;
  onComplete: () => void;
  onReset: () => void;
  toast: ToastFn;
}) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing report engine…");
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const startedGenerationRef = useRef<string | null>(null);
  const [remoteHiplotPayload, setRemoteHiplotPayload] = useState<HiPlotPayload | null>(null);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isDeletingExtract, setIsDeletingExtract] = useState(false);

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        window.URL.revokeObjectURL(downloadUrl);
      }
    };
  }, [downloadUrl]);

  useEffect(() => {
    if (step !== "complete" || !taskId) return;
    let cancelled = false;
    api
      .getHiPlotData(taskId, 2000)
      .then((res) => {
        if (cancelled || !res || !Array.isArray(res.dimensions) || res.dimensions.length < 2) return;
        setRemoteHiplotPayload({
          total_rows: res.total_rows,
          sampled_rows: res.sampled_rows,
          is_sampled: Boolean(res.is_sampled ?? res.total_rows > res.sampled_rows),
          dimensions: res.dimensions,
          datapoints: res.datapoints,
        });
      })
      .catch(() => {
        // Silently keep preview fallback if analytics endpoint unavailable
      });
    return () => {
      cancelled = true;
    };
  }, [step, taskId]);

  useEffect(() => {
    if (step !== "generating" || !taskId) return;
    if (startedGenerationRef.current === taskId) return;
    startedGenerationRef.current = taskId;

    let cancelled = false;
    runReportCompilationPipeline(taskId, {
      isCancelled: () => cancelled,
      onStatus: setStatus,
      onProgress: setProgress,
      onDownloadUrl: setDownloadUrl,
      onComplete: () => {
        onComplete();
        toast({ title: "Report Ready", description: "PDF compiled and ready for download." });
      },
      onError: (errMsg) => {
        startedGenerationRef.current = null;
        setStatus("Report generation failed.");
        toast({ title: "PDF Generation Error", description: errMsg, variant: "destructive" });
      },
    });

    return () => {
      cancelled = true;
    };
  }, [step, taskId, onComplete, toast]);

  const downloadFile = async () => {
    let transientUrl: string | null = null;
    let targetUrl = downloadUrl;
    if (!targetUrl && taskId) {
      setIsDownloadingPdf(true);
      try {
        const blob = await api.downloadReportBlob(taskId);
        transientUrl = window.URL.createObjectURL(blob);
        targetUrl = transientUrl;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Report is still generating.";
        toast({ title: "Not Ready", description: msg });
        setIsDownloadingPdf(false);
        return;
      } finally {
        setIsDownloadingPdf(false);
      }
    }

    if (targetUrl) {
      const link = document.createElement("a");
      link.href = targetUrl;
      const baseName = filename ? filename.replace(/\.[^/.]+$/, "") : "Analysis";
      link.download = `${baseName}_Report.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      if (transientUrl) {
        const urlToRevoke = transientUrl;
        window.setTimeout(() => window.URL.revokeObjectURL(urlToRevoke), 1000);
      }
    } else {
      toast({ title: "Not Ready", description: "Report is still generating." });
    }
  };

  const downloadExport = async (format: ExportFormat) => {
    if (!taskId) return;
    try {
      toast({ title: "Preparing Export", description: `Packaging ${format.toUpperCase()} export file...` });
      const blob = await api.downloadExportBlob(taskId, format);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const safeName = filename ? filename.replace(/\.[^/.]+$/, "") : "dataset";
      const fileExt = format === "excel" ? "xlsx" : format;
      link.download = format === "html" ? `Report_${safeName}.html` : `Cleaned_${safeName}.${fileExt}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      toast({ title: "Export Downloaded", description: `Successfully exported ${format.toUpperCase()} file.` });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not download export.";
      toast({ title: "Export Failed", description: msg, variant: "destructive" });
    }
  };

  const handleDeleteExtract = async () => {
    if (!taskId) return;
    if (!window.confirm("Permanently delete this extracted dataset and purge cached files from storage?")) return;
    setIsDeletingExtract(true);
    try {
      await api.deleteSnowflakeExtract(taskId);
      toast({
        title: "Extract Purged",
        description: "Local data extract permanently deleted from storage.",
      });
      onReset();
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Could not delete extract.";
      toast({ title: "Purge Failed", description: msg, variant: "destructive" });
    } finally {
      setIsDeletingExtract(false);
    }
  };

  return {
    progress,
    status,
    remoteHiplotPayload,
    isDownloadingPdf,
    isDeletingExtract,
    downloadFile,
    downloadExport,
    handleDeleteExtract,
  };
}
