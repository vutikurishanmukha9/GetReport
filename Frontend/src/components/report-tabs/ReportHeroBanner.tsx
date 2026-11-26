import { BookOpen, ChevronRight, Download, RefreshCw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { AnimatedDeleteButton, Button, ShimmerActionButton } from "@/components/ui/button";
import { AnimatedNumber } from "@/components/motion/animated-number";
import type { AnalysisResult, ConfidenceScores, DatasetInfo, ExportFormat } from "@/types/api";
import { getSealGradeConfig } from "./gradeTheme";

interface ReportHeroBannerProps {
  filename: string;
  info: DatasetInfo;
  analysis: AnalysisResult;
  confidence: ConfidenceScores;
  datasetGrade: string;
  datasetConfidence: number;
  isDownloadingPdf: boolean;
  isDeletingExtract: boolean;
  onDownloadPdf: () => void;
  onExportHtml: (format: ExportFormat) => void;
  onDeleteExtract: () => void;
  onReset: () => void;
}

export function ReportHeroBanner({
  filename,
  info,
  analysis,
  confidence,
  datasetGrade,
  datasetConfidence,
  isDownloadingPdf,
  isDeletingExtract,
  onDownloadPdf,
  onExportHtml,
  onDeleteExtract,
  onReset,
}: ReportHeroBannerProps) {
  const seal = getSealGradeConfig(datasetGrade);
  const displayName = filename ? filename.replace(/\.[^/.]+$/, "") : "Dataset Analysis";

  return (
    <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 p-6 sm:p-8 bg-card border border-border/80 rounded-3xl shadow-premium relative overflow-hidden">
      <div className="flex items-center gap-5 z-10">
        <div
          className={`flex flex-col items-center justify-center h-20 w-20 sm:h-24 sm:w-24 rounded-2xl ${seal.container} border shadow-sm shrink-0 transition-colors`}
        >
          <span className={`text-[9px] font-sans font-bold tracking-[0.2em] uppercase leading-none mb-1 ${seal.label}`}>
            GRADE
          </span>
          <span className={`text-3xl sm:text-4xl font-sans font-black leading-none tracking-tight ${seal.grade}`}>
            {datasetGrade}
          </span>
          <span className={`text-[11px] font-sans font-bold mt-1 tabular-nums ${seal.score}`}>
            <AnimatedNumber value={datasetConfidence} format={(n) => n.toFixed(1) + "%"} duration={0.8} />
          </span>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-2xl sm:text-3xl font-sans font-bold text-foreground tracking-[-0.025em] leading-tight">
              {displayName}
            </h1>
            {analysis?.snowflake_meta && (
              <Badge
                variant="outline"
                className="text-[10px] font-sans font-medium uppercase bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.04] dark:text-zinc-300 dark:border-white/10 inline-flex items-center gap-1.5"
              >
                <span>Snowflake</span>
                <span className="px-1 py-0.5 text-[8px] font-sans font-semibold uppercase tracking-widest text-stone-700 bg-stone-300/80 border border-stone-300 dark:text-zinc-400 dark:bg-white/[0.06] dark:border-white/10 rounded leading-none">
                  Beta
                </span>
              </Badge>
            )}
            {analysis?.snowflake_meta?.truncated && (
              <Badge variant="outline" className="text-[10px] font-sans font-medium uppercase bg-rose-500/15 text-rose-400 border-rose-500/30">
                Truncated at Limit (150MB Ceiling)
              </Badge>
            )}
            {analysis?.snowflake_meta?.sampled && (
              <Badge
                variant="outline"
                className="text-[10px] font-sans font-medium uppercase bg-stone-200/80 text-stone-800 border-stone-300 dark:bg-white/[0.06] dark:text-zinc-300 dark:border-white/10"
              >
                Sampled (10% Bernoulli)
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs sm:text-[13px] text-muted-foreground font-sans font-medium flex-wrap">
            <span>
              <span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums">
                <AnimatedNumber value={info.rows} format={(n) => Math.round(n).toLocaleString()} duration={0.8} />
              </span>{" "}
              rows
            </span>
            <span className="text-stone-400 dark:text-white/20">•</span>
            <span>
              <span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums">
                <AnimatedNumber value={info.columns.length} format={(n) => String(Math.round(n))} duration={0.8} />
              </span>{" "}
              columns
            </span>
            <span className="text-stone-400 dark:text-white/20">•</span>
            <span>
              <span className="font-semibold text-stone-900 dark:text-zinc-200 tabular-nums">
                <AnimatedNumber value={info.memory_usage_mb || 0} format={(n) => n.toFixed(2) + " MB"} duration={0.8} />
              </span>{" "}
              in-memory
            </span>
          </div>
          <div className="text-xs font-sans text-muted-foreground/85 flex items-center gap-2.5 pt-0.5 flex-wrap">
            <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700 dark:text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400" />
              <span className="font-semibold tabular-nums">
                <AnimatedNumber value={confidence.high_confidence_count} format={(n) => String(Math.round(n))} duration={0.8} />
              </span>{" "}
              High Confidence Columns
            </span>
            <span className="text-stone-400 dark:text-white/20">•</span>
            <span className="inline-flex items-center gap-1.5 font-medium text-stone-600 dark:text-zinc-400">
              <span className="font-semibold tabular-nums text-stone-900 dark:text-zinc-300">
                <AnimatedNumber value={confidence.low_confidence_count} format={(n) => String(Math.round(n))} duration={0.8} />
              </span>{" "}
              Need Attention
            </span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2.5 z-10 w-full md:w-auto">
        <ShimmerActionButton
          onClick={onDownloadPdf}
          disabled={isDownloadingPdf}
          className="flex-1 md:flex-none"
          icon={<Download className="h-4 w-4 text-white dark:text-black stroke-[2.5]" />}
          trailingIcon={<ChevronRight className="h-4 w-4 text-white dark:text-black stroke-[2.5]" />}
        >
          <span>{isDownloadingPdf ? "Downloading…" : "Download PDF"}</span>
        </ShimmerActionButton>

        <Button
          size="lg"
          variant="outline"
          className="flex-1 md:flex-none rounded-xl border-stone-300 bg-stone-100 hover:bg-stone-200 text-stone-800 dark:border-white/10 dark:bg-white/[0.04] dark:hover:bg-white/[0.08] dark:text-foreground shadow-xs transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center justify-center gap-2 font-sans font-semibold text-sm"
          onClick={() => onExportHtml("html")}
        >
          <BookOpen className="h-4 w-4 text-primary" />
          <span>HTML Briefing</span>
        </Button>

        {analysis?.snowflake_meta && (
          <AnimatedDeleteButton
            size="sm"
            variant="danger"
            label="Purge Extract"
            confirmLabel="Purge Disk Cache?"
            confirmRequired={true}
            disabled={isDeletingExtract}
            loading={isDeletingExtract}
            onDelete={onDeleteExtract}
            title="Permanently purge cached extract from disk"
          />
        )}

        <Button
          size="lg"
          variant="ghost"
          className="rounded-xl text-muted-foreground hover:text-foreground font-sans font-medium text-xs flex items-center gap-1.5 px-3"
          onClick={onReset}
          title="Audit a different file"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">New Audit</span>
        </Button>
      </div>
    </div>
  );
}
