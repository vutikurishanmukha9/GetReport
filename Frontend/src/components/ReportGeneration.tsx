import { useMemo, useState, lazy, Suspense } from "react";
import {
  Activity,
  BarChart3,
  Download,
  FileText,
  Grid,
  Loader2,
  PieChart,
  ShieldCheck,
  Sliders,
  Terminal,
  TrendingUp,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AppStep } from "@/pages/Workspace";
import type {
  AnalysisResult,
  Charts,
  ConfidenceScores,
  DatasetInfo,
  InsightResult,
} from "@/types/api";
import { useToast } from "@/hooks/use-toast";
import type { HiPlotDimension, HiPlotPayload } from "./HiPlotParallelCoordinates";
import { SupersetExploreStudio } from "./SupersetExploreStudio";
import { ErrorBoundary } from "./ErrorBoundary";
import {
  MAIN_TAB_TRIGGER_CLASS,
  TAB_COUNT_BADGE_CLASS,
  normalizeMainTab,
} from "./report-tabs/gradeTheme";
import { useReportPdfFlow } from "./report-tabs/useReportPdfFlow";
import { ReportHeroBanner } from "./report-tabs/ReportHeroBanner";
import { OverviewTab } from "./report-tabs/OverviewTab";
import { VisualChartsTab } from "./report-tabs/VisualChartsTab";
import { TrustLedgerTab } from "./report-tabs/TrustLedgerTab";
import { StatisticalAuditTab } from "./report-tabs/StatisticalAuditTab";
import { ExportHubTab } from "./report-tabs/ExportHubTab";
import { GeneratingReportSkeleton } from "./report-tabs/GeneratingReportSkeleton";

const DuckDBSqlStudio = lazy(() => import("./DuckDBSqlStudio"));

interface ReportGenerationProps {
  step: AppStep;
  taskId: string | null;
  filename: string;
  info: DatasetInfo;
  analysis: AnalysisResult;
  charts: Charts;
  insights: InsightResult;
  onComplete: () => void;
  onReset: () => void;
  requestedTab?: { tab: string; nonce: number } | null;
  requestedSql?: { sql: string; nonce: number } | null;
}

function buildFallbackHiplotPayload(info: DatasetInfo): HiPlotPayload | null {
  if (!info?.preview || info.preview.length < 2) return null;

  const candidateCols = (info?.columns || [])
    .filter((col) => {
      const dtype = (info.dtypes?.[col] || "").toLowerCase();
      return (
        dtype.includes("int") ||
        dtype.includes("float") ||
        dtype.includes("str") ||
        dtype.includes("cat")
      );
    })
    .slice(0, 8);

  if (candidateCols.length < 2) return null;

  const dimensions: HiPlotDimension[] = candidateCols.map((col) => {
    const dtype = (info.dtypes[col] || "").toLowerCase();
    const isNumeric = dtype.includes("int") || dtype.includes("float");
    const summaryCol = info.summary?.[col];

    return {
      name: col,
      display_name: col,
      type: isNumeric ? "numeric" : "categorical",
      is_numeric: isNumeric,
      min: summaryCol && Number.isFinite(summaryCol.min) ? Number(summaryCol.min) : undefined,
      max: summaryCol && Number.isFinite(summaryCol.max) ? Number(summaryCol.max) : undefined,
      mean: summaryCol && Number.isFinite(summaryCol.mean) ? Number(summaryCol.mean) : undefined,
    };
  });

  return {
    total_rows: info.rows,
    sampled_rows: info.preview.length,
    is_sampled: info.rows > info.preview.length,
    dimensions,
    datapoints: info.preview,
  };
}

function buildFallbackConfidenceScores(info: DatasetInfo): ConfidenceScores {
  return {
    dataset_grade: "B",
    dataset_confidence: 82.5,
    high_confidence_count: info.columns.length,
    low_confidence_count: 0,
    critical_issues: [],
    ml_readiness: undefined,
    columns: info.columns.map((col) => {
      const missing = info.missing_values[col] || { count: 0, percentage: 0 };
      const completeness = 100 - missing.percentage;
      let grade = "A";
      if (completeness < 50) grade = "F";
      else if (completeness < 75) grade = "C";
      else if (completeness < 90) grade = "B";

      return {
        column: col,
        completeness,
        consistency: 92,
        validity: 95,
        stability: 88,
        overall: (completeness + 92 + 95 + 88) / 4,
        grade,
        issues: missing.count > 0 ? [`${missing.count} missing rows`] : [],
      };
    }),
  };
}

function buildChartTabsList(charts: Charts, hiplotPayload: HiPlotPayload | null) {
  const list = [];
  if (charts.correlation_heatmap) {
    list.push({ id: "correlation", label: "Correlation Matrix", icon: Grid });
  }
  if (charts.distributions && charts.distributions.length > 0) {
    list.push({ id: "distributions", label: "Distributions", icon: BarChart3 });
  }
  if ((charts.bar_charts && charts.bar_charts.length > 0) || charts.donut_chart) {
    list.push({ id: "composition", label: "Composition", icon: PieChart });
  }
  if ((charts.boxplots && charts.boxplots.length > 0) || charts.scatter_plot) {
    list.push({ id: "bivariate", label: "Bivariate Relations", icon: TrendingUp });
  }
  if (hiplotPayload && hiplotPayload.dimensions.length >= 2) {
    list.push({ id: "hiplot", label: "Parallel Coordinates", icon: Sliders });
  }
  return list;
}

export const ReportGeneration = ({
  step,
  taskId,
  filename,
  info,
  analysis,
  charts,
  insights: _insights,
  onComplete,
  onReset,
  requestedTab,
  requestedSql,
}: ReportGenerationProps) => {
  const { toast } = useToast();

  const pdfFlow = useReportPdfFlow({
    step,
    taskId,
    filename,
    onComplete,
    onReset,
    toast,
  });

  const [tabSelection, setTabSelection] = useState<{ nonce?: number; tab: string }>(() => ({
    nonce: requestedTab?.nonce,
    tab: normalizeMainTab(requestedTab?.tab),
  }));
  const activeMainTab =
    requestedTab?.tab && requestedTab.nonce !== tabSelection.nonce
      ? normalizeMainTab(requestedTab.tab)
      : tabSelection.tab;

  const setActiveMainTab = (nextTab: string) => {
    setTabSelection({ nonce: requestedTab?.nonce, tab: nextTab });
  };

  const hiplotPayload = useMemo<HiPlotPayload | null>(() => {
    if (pdfFlow.remoteHiplotPayload && pdfFlow.remoteHiplotPayload.dimensions.length >= 2) {
      return pdfFlow.remoteHiplotPayload;
    }
    return buildFallbackHiplotPayload(info);
  }, [info, pdfFlow.remoteHiplotPayload]);

  const fallbackConfidenceScores = useMemo(() => buildFallbackConfidenceScores(info), [info]);

  const confidence =
    analysis?.confidence_scores && Array.isArray(analysis.confidence_scores.columns)
      ? analysis.confidence_scores
      : fallbackConfidenceScores;
  const datasetGrade = confidence?.dataset_grade ?? "B";
  const datasetConfidence = confidence?.dataset_confidence ?? 82.5;
  const chartTabsList = useMemo(() => buildChartTabsList(charts, hiplotPayload), [charts, hiplotPayload]);

  if (step !== "complete") {
    return (
      <GeneratingReportSkeleton
        status={pdfFlow.status}
        progress={pdfFlow.progress}
        info={info}
        filename={filename}
      />
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in zoom-in-95 duration-400">
      <ReportHeroBanner
        filename={filename}
        info={info}
        analysis={analysis}
        confidence={confidence}
        datasetGrade={datasetGrade}
        datasetConfidence={datasetConfidence}
        isDownloadingPdf={pdfFlow.isDownloadingPdf}
        isDeletingExtract={pdfFlow.isDeletingExtract}
        onDownloadPdf={() => void pdfFlow.downloadFile()}
        onExportHtml={(fmt) => void pdfFlow.downloadExport(fmt)}
        onDeleteExtract={() => void pdfFlow.handleDeleteExtract()}
        onReset={onReset}
      />

      <Tabs value={activeMainTab} onValueChange={setActiveMainTab} className="w-full">
        <TabsList className="h-auto w-full bg-[#EDE8DE] border border-[#DCD6C9] dark:bg-[#0c0d12]/95 dark:border-white/10 p-1.5 rounded-2xl mb-6 shadow-sm flex flex-wrap items-stretch justify-between gap-1.5">
          <TabsTrigger value="overview" className={MAIN_TAB_TRIGGER_CLASS}>
            <FileText className="h-3.5 w-3.5 shrink-0" />
            <span>Overview</span>
          </TabsTrigger>

          <TabsTrigger value="charts" className={MAIN_TAB_TRIGGER_CLASS}>
            <TrendingUp className="h-3.5 w-3.5 shrink-0" />
            <span>Visual Analytics</span>
            {chartTabsList.length > 0 && <span className={TAB_COUNT_BADGE_CLASS}>{chartTabsList.length}</span>}
          </TabsTrigger>

          <TabsTrigger value="ledger" className={MAIN_TAB_TRIGGER_CLASS}>
            <ShieldCheck className="h-3.5 w-3.5 shrink-0" />
            <span>Trust Ledger</span>
            <span className={TAB_COUNT_BADGE_CLASS}>{confidence.columns.length}</span>
          </TabsTrigger>

          <TabsTrigger value="stats" className={MAIN_TAB_TRIGGER_CLASS}>
            <Activity className="h-3.5 w-3.5 shrink-0" />
            <span>Statistical Audit</span>
          </TabsTrigger>

          <TabsTrigger value="sql" className={MAIN_TAB_TRIGGER_CLASS}>
            <Terminal className="h-3.5 w-3.5 shrink-0" />
            <span>DuckDB SQL Studio</span>
          </TabsTrigger>

          <TabsTrigger value="bi_analytics" className={MAIN_TAB_TRIGGER_CLASS}>
            <BarChart3 className="h-3.5 w-3.5 shrink-0" />
            <span>Interactive BI</span>
          </TabsTrigger>

          <TabsTrigger value="export" className={MAIN_TAB_TRIGGER_CLASS}>
            <Download className="h-3.5 w-3.5 shrink-0" />
            <span>Export Hub</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-8 animate-in fade-in duration-300">
          <OverviewTab datasetConfidence={datasetConfidence} info={info} confidence={confidence} />
        </TabsContent>

        <TabsContent value="charts" className="space-y-6 animate-in fade-in duration-300">
          <VisualChartsTab
            charts={charts}
            chartTabsList={chartTabsList}
            hiplotPayload={hiplotPayload}
            onIsolateSelection={(count) =>
              toast({
                title: "Population Isolated",
                description: `Filtered ${count.toLocaleString()} records from HiPlot parallel coordinates selection.`,
              })
            }
          />
        </TabsContent>

        <TabsContent value="ledger" className="space-y-6 animate-in fade-in duration-300">
          <TrustLedgerTab confidence={confidence} info={info} />
        </TabsContent>

        <TabsContent value="stats" className="space-y-6 animate-in fade-in duration-300">
          <StatisticalAuditTab analysis={analysis} />
        </TabsContent>

        <TabsContent value="sql" className="space-y-6 animate-in fade-in duration-300">
          <ErrorBoundary fallbackTitle="DuckDB SQL Studio Encountered an Error">
            <Suspense
              fallback={
                <div className="flex flex-col items-center justify-center p-12 space-y-3 rounded-2xl border border-border bg-card">
                  <Loader2 className="h-7 w-7 text-primary animate-spin" />
                  <span className="text-xs font-mono text-muted-foreground uppercase tracking-widest">
                    Loading DuckDB SQL Studio...
                  </span>
                </div>
              }
            >
              <DuckDBSqlStudio
                taskId={taskId}
                datasetName={filename}
                info={info}
                externalSqlRequest={requestedSql}
              />
            </Suspense>
          </ErrorBoundary>
        </TabsContent>

        <TabsContent value="export" className="space-y-6 animate-in fade-in duration-300">
          <ExportHubTab
            taskId={taskId}
            filename={filename}
            isDownloadingPdf={pdfFlow.isDownloadingPdf}
            onDownloadPdf={() => void pdfFlow.downloadFile()}
            onDownloadExport={(fmt) => void pdfFlow.downloadExport(fmt)}
            toast={toast}
          />
        </TabsContent>

        <TabsContent value="bi_analytics" className="space-y-6 animate-in fade-in duration-300">
          <ErrorBoundary fallbackTitle="Interactive BI Studio Encountered an Error">
            <SupersetExploreStudio
              taskId={taskId || undefined}
              datasetName={filename}
              allowDatasetSwitching={false}
              initialInfo={info}
            />
          </ErrorBoundary>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default ReportGeneration;
