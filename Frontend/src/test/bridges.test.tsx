import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { api, type StatusResponse } from "@/services/api";
import { IssueLedger } from "@/components/IssueLedger";
import { FileUpload } from "@/components/FileUpload";
import { Dashboard } from "@/pages/Dashboard";
import { ChatInterface } from "@/components/ChatInterface";
import { SupersetExploreStudio } from "@/components/SupersetExploreStudio";
import { ReportGeneration } from "@/components/ReportGeneration";
import { useTaskStatus } from "@/hooks/useTaskStatus";
import type {
  AnalysisResult,
  DatasetInfo,
  InsightResult,
  IssueLedgerData,
} from "@/types/api";

class StubResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

class StubIntersectionObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

class StubEventSource {
  close() {}
}

function makeSampleInfo(): DatasetInfo {
  return {
    rows: 20,
    columns: ["dept", "revenue"],
    dtypes: { dept: "Utf8", revenue: "Float64" },
    summary: { revenue: { count: 20, mean: 150 } },
    preview: [
      { dept: "Engineering", revenue: 200 },
      { dept: "Sales", revenue: 100 },
    ],
    missing_values: {
      dept: { count: 0, percentage: 0 },
      revenue: { count: 0, percentage: 0 },
    },
    duplicate_rows: 0,
    numeric_columns: ["revenue"],
    categorical_columns: ["dept"],
    memory_usage_mb: 0.1,
  };
}

function makeSampleAnalysis(): AnalysisResult {
  return {
    metadata: {},
    summary: { revenue: { count: 20, mean: 150, std: 50, min: 100, max: 200 } },
    correlation: {},
    strong_correlations: [],
    outliers: {},
    categorical_distribution: {},
    column_quality_flags: {},
    timing_ms: 42,
  };
}

function makeSampleInsights(): InsightResult {
  return {
    insights_text: "Revenue is concentrated in Engineering.",
    model_used: "test-model",
    prompt_tokens: 10,
    completion_tokens: 10,
    total_tokens: 20,
    response_time_ms: 15,
    retries_attempted: 0,
    success: true,
    fallback_reason: "",
  };
}

describe("frontend bridge regressions", () => {
  beforeEach(() => {
    localStorage.clear();
    Object.defineProperty(globalThis, "ResizeObserver", {
      writable: true,
      configurable: true,
      value: StubResizeObserver,
    });
    Object.defineProperty(globalThis, "IntersectionObserver", {
      writable: true,
      configurable: true,
      value: StubIntersectionObserver,
    });
    Object.defineProperty(globalThis, "EventSource", {
      writable: true,
      configurable: true,
      value: StubEventSource,
    });
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      writable: true,
      configurable: true,
      value: vi.fn(),
    });
    Object.defineProperty(window.URL, "createObjectURL", {
      writable: true,
      configurable: true,
      value: vi.fn(() => "blob:http://localhost/mock-report"),
    });
    Object.defineProperty(window.URL, "revokeObjectURL", {
      writable: true,
      configurable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("IssueLedger supports inline custom fix modification and triggers refresh", async () => {
    const ledgerData: IssueLedgerData = {
      issues: [
        {
          id: "iss-1",
          issue_type: "missing_values",
          severity: "high",
          column: "income",
          affected_rows: 5,
          affected_pct: 10,
          description: "Missing customer income values",
          suggested_fix: "Impute median income",
          fix_code: "df['income'] = df['income'].fillna(df['income'].median())",
          status: "pending",
          user_note: "",
        },
      ],
      summary: { pending: 1, approved: 0, rejected: 0, modified: 0, total: 1 },
      locked: false,
      locked_at: null,
      created_at: "2026-01-01T00:00:00Z",
    };

    const modifySpy = vi.spyOn(api, "modifyIssue").mockResolvedValueOnce({
      issues: ledgerData.issues,
      locked: false,
      locked_at: null,
      created_at: "2026-01-01T00:00:00Z",
      message: "Issue modified",
      issue_id: "iss-1",
      fix_code: "df = df.dropna(subset=['income'])",
      summary: { pending: 0, approved: 0, rejected: 0, modified: 1, total: 1 },
    });
    const onRefresh = vi.fn().mockResolvedValue(undefined);

    render(
      <IssueLedger
        taskId="t-iss"
        data={ledgerData}
        onRefresh={onRefresh}
        onProceed={vi.fn()}
      />,
    );

    expect(screen.getByText("Modified")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /view polars code/i }));

    const editBtn = await screen.findByRole("button", { name: /edit fix/i });
    fireEvent.click(editBtn);

    const codeEditor = await screen.findByDisplayValue("df['income'] = df['income'].fillna(df['income'].median())");
    fireEvent.change(codeEditor, { target: { value: "df = df.dropna(subset=['income'])" } });
    fireEvent.click(screen.getByRole("button", { name: /save modification/i }));

    await waitFor(() => {
      expect(modifySpy).toHaveBeenCalledWith("t-iss", "iss-1", "df = df.dropna(subset=['income'])", undefined);
      expect(onRefresh).toHaveBeenCalledTimes(1);
    });
  });

  it("FileUpload resolves education_perf, saas_churn, and hr_attrition sample presets without falling back", () => {
    vi.spyOn(api, "getTaskStatus").mockResolvedValue({
      task_id: "t-sample",
      status: "PENDING",
      progress: 0,
      message: "Queued",
    });
    vi.spyOn(api, "subscribeTaskStatus").mockReturnValue(() => {});

    const { rerender } = render(
      <MemoryRouter>
        <FileUpload onFileUploaded={vi.fn()} initialSampleId="education_perf" />
      </MemoryRouter>,
    );
    expect(screen.getByText("university_academic_cohort.csv")).toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <FileUpload onFileUploaded={vi.fn()} initialSampleId="saas_churn" />
      </MemoryRouter>,
    );
    expect(screen.getByText("saas_retention_telemetry.csv")).toBeInTheDocument();

    rerender(
      <MemoryRouter>
        <FileUpload onFileUploaded={vi.fn()} initialSampleId="hr_attrition" />
      </MemoryRouter>,
    );
    expect(screen.getByText("workforce_compensation_attrition.csv")).toBeInTheDocument();
  });

  it("FileUpload single sample uploads matching preset filename and locks issues before starting analysis", async () => {
    let waitingHandler: ((data: StatusResponse) => void) | undefined;
    vi.spyOn(api, "getTaskStatus").mockResolvedValue({
      task_id: "t-gate",
      status: "PENDING",
      progress: 0,
      message: "Queued",
    });
    vi.spyOn(api, "subscribeTaskStatus").mockImplementation((_taskId, callbacks) => {
      waitingHandler = callbacks.onWaitingForUser;
      return () => {};
    });

    const uploadSpy = vi.spyOn(api, "uploadFile").mockResolvedValueOnce({
      task_id: "t-gate",
      message: "File queued",
    });
    const callOrder: string[] = [];
    const lockSpy = vi.spyOn(api, "lockIssues").mockImplementationOnce(async () => {
      callOrder.push("lockIssues");
      return {
        issues: [],
        summary: { pending: 0, approved: 1, rejected: 0, modified: 0, total: 1 },
        locked: true,
        locked_at: "2026-01-01T00:00:00Z",
        created_at: "2026-01-01T00:00:00Z",
        message: "Locked",
      };
    });
    const startSpy = vi.spyOn(api, "startAnalysis").mockImplementationOnce(async () => {
      callOrder.push("startAnalysis");
      return { task_id: "t-gate", message: "Analysis queued" };
    });

    const ledgerIssues = [
      {
        id: "iss-gpa",
        issue_type: "missing_values",
        severity: "medium" as const,
        column: "term_gpa",
        affected_rows: 1,
        affected_pct: 12.5,
        description: "Missing GPA value",
        suggested_fix: "Fill median",
        fix_code: "df['term_gpa'].fillna(df['term_gpa'].median())",
        status: "approved" as const,
        user_note: "",
      },
    ];
    vi.spyOn(api, "approveAllIssues").mockResolvedValueOnce({
      issues: ledgerIssues,
      summary: { pending: 0, approved: 1, rejected: 0, modified: 0, total: 1 },
      locked: false,
      locked_at: null,
      created_at: "2026-01-01T00:00:00Z",
      message: "All approved",
      count: 1,
    });
    vi.spyOn(api, "getIssues").mockResolvedValueOnce({
      issues: ledgerIssues,
      summary: { pending: 0, approved: 1, rejected: 0, modified: 0, total: 1 },
      locked: false,
      locked_at: null,
      created_at: "2026-01-01T00:00:00Z",
    });

    render(
      <MemoryRouter>
        <FileUpload onFileUploaded={vi.fn()} initialSampleId="education_perf" />
      </MemoryRouter>,
    );
    fireEvent.click(screen.getByRole("button", { name: /start analysis & audit/i }));

    await waitFor(() => {
      expect(uploadSpy).toHaveBeenCalledTimes(1);
    });
    expect(uploadSpy.mock.calls[0]?.[0]?.name).toBe("university_academic_cohort.csv");

    act(() => {
      waitingHandler?.({
        task_id: "t-gate",
        status: "WAITING_FOR_USER",
        progress: 45,
        message: "Inspection complete",
        result: {
          filename: "university_academic_cohort.csv",
          raw_file_path: "/tmp/university_academic_cohort.csv",
          stage: "INSPECTION",
          preview: [],
          quality_report: {
            total_rows: 8,
            columns: [],
            issues: [
              {
                type: "missing_values",
                column: "term_gpa",
                count: 1,
                severity: "medium",
                suggestion: "Fill median",
              },
            ],
            preview: [],
          },
          issue_ledger: {
            issues: [
              {
                id: "iss-gpa",
                issue_type: "missing_values",
                severity: "medium",
                column: "term_gpa",
                affected_rows: 1,
                affected_pct: 12.5,
                description: "Missing GPA value",
                suggested_fix: "Fill median",
                fix_code: "df['term_gpa'].fillna(df['term_gpa'].median())",
                status: "pending",
                user_note: "",
              },
            ],
            summary: { pending: 1, approved: 0, rejected: 0, modified: 0, total: 1 },
            locked: false,
            locked_at: null,
            created_at: "2026-01-01T00:00:00Z",
          },
        },
      });
    });

    const startAnalysisBtn = await screen.findByRole("button", { name: /^start analysis$/i });
    fireEvent.click(startAnalysisBtn);
    expect(lockSpy).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /approve all/i }));
    await waitFor(() => {
      expect(screen.queryByText(/Pending Ledger Issue/i)).not.toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^start analysis$/i }));

    await waitFor(() => {
      expect(lockSpy).toHaveBeenCalledWith("t-gate");
      expect(startSpy).toHaveBeenCalledTimes(1);
    });
    expect(callOrder).toEqual(["lockIssues", "startAnalysis"]);
  });

  it("Dashboard loads recent datasets from backend and renders real empty state instead of fake demo row", async () => {
    const recentSpy = vi.spyOn(api, "listRecentDatasets").mockResolvedValueOnce({
      datasets: [
        {
          task_id: "real-ds-1",
          filename: "enterprise_orders_2026.csv",
          rows: 1500,
          columns_count: 10,
          columns: ["order_id", "amount"],
          numeric_columns: ["amount"],
          categorical_columns: ["order_id"],
          created_at: "2026-10-01T12:00:00Z",
        },
      ],
    });

    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText("enterprise_orders_2026.csv")).toBeInTheDocument();
    });
    expect(screen.queryByText(/demo_audit_01/i)).not.toBeInTheDocument();
    expect(recentSpy).toHaveBeenCalledWith(20);

    cleanup();
    vi.spyOn(api, "listRecentDatasets").mockResolvedValueOnce({ datasets: [] });
    render(
      <MemoryRouter>
        <Dashboard />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText("No Audits Found")).toBeInTheDocument();
    });
  });

  it("ChatInterface hydrates saved Golden KPIs, deletes via server id, and restores welcome prompt on Clear", async () => {
    vi.spyOn(api, "getGoldenQueries").mockResolvedValueOnce([
      {
        id: "gq-srv-1",
        task_id: "t-chat",
        question: "Top revenue region",
        sql_query: "SELECT region FROM dataset LIMIT 1",
        description: "Saved test",
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);
    const deleteSpy = vi.spyOn(api, "deleteGoldenQuery").mockResolvedValueOnce({
      status: "ok",
      deleted: true,
      query_id: "gq-srv-1",
    });

    render(<ChatInterface taskId="t-chat" />);

    await waitFor(() => {
      expect(screen.getByText("Golden KPIs (1):")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByLabelText("Delete Golden KPI Top revenue region"));
    await waitFor(() => {
      expect(deleteSpy).toHaveBeenCalledWith("t-chat", "gq-srv-1");
    });

    fireEvent.click(screen.getByRole("button", { name: /clear chat/i }));
    const confirmBtn = await screen.findByRole("button", { name: /clear all\?/i });
    fireEvent.click(confirmBtn);
    expect(screen.getByText(/completed the data audit and created the full pdf report/i)).toBeInTheDocument();
  });

  it("SupersetExploreStudio Refresh button forces a fresh executeQuery request", async () => {
    vi.spyOn(api, "getSupersetSchema").mockResolvedValue({
      task_id: "t-bi",
      total_rows: 10,
      total_columns: 2,
      columns: [
        {
          name: "dept",
          type: 1,
          type_name: "VARCHAR",
          is_numeric: false,
          is_temporal: false,
          null_count: 0,
          unique_values: 2,
        },
        {
          name: "revenue",
          type: 2,
          type_name: "DOUBLE",
          is_numeric: true,
          is_temporal: false,
          null_count: 0,
          unique_values: 10,
        },
      ],
      numeric_columns: ["revenue"],
      categorical_columns: ["dept"],
      temporal_columns: [],
      color_palettes: ["emerald"],
    });

    const querySpy = vi.spyOn(api, "querySupersetExplore").mockResolvedValue({
      task_id: "t-bi",
      chart_type: "bar",
      dimension: "dept",
      aggregate: "count",
      metric_label: "count",
      row_limit: 50,
      kpis: {
        metric_label: "count",
        dimension_label: "dept",
        total_rows: 10,
        unique_categories: 2,
        sum: 300,
        mean: 150,
        min: 100,
        max: 200,
      },
      records: [],
      total_groups: 2,
      echarts_option: {},
    });

    render(
      <SupersetExploreStudio
        taskId="t-bi"
        datasetName="sales.csv"
        initialInfo={{
          rows: 10,
          columns: ["dept", "revenue"],
          numeric_columns: ["revenue"],
          categorical_columns: ["dept"],
        }}
      />,
    );

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(screen.getByRole("button", { name: /refresh/i }));

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledTimes(2);
    });
  });

  it("useTaskStatus resets progress/state on disconnect and falls back to WebSocket on SSE error", async () => {
    let sseErrorCb: ((err: Event) => void) | undefined;
    let sseProgressCb: ((data: StatusResponse) => void) | undefined;

    vi.spyOn(api, "getTaskStatus").mockResolvedValue({
      task_id: "t-ws",
      status: "PROCESSING",
      progress: 10,
      message: "Running",
    });
    vi.spyOn(api, "subscribeTaskStatus").mockImplementation((_taskId, callbacks) => {
      sseProgressCb = callbacks.onProgress;
      sseErrorCb = callbacks.onError;
      return () => {};
    });
    const wsSpy = vi.spyOn(api, "subscribeWebSocketStatus").mockReturnValue(() => {});

    const initialTaskId: string | undefined = "t-ws";
    const { result, rerender } = renderHook(
      (id: string | undefined) => useTaskStatus(id),
      { initialProps: initialTaskId },
    );

    await waitFor(() => {
      expect(sseProgressCb).toBeDefined();
    });

    act(() => {
      sseProgressCb?.({
        task_id: "t-ws",
        status: "PROCESSING",
        progress: 64,
        message: "Halfway done",
      });
    });
    expect(result.current.progress).toBe(64);

    act(() => {
      sseErrorCb?.(new Event("error"));
    });
    expect(wsSpy).toHaveBeenCalledWith("t-ws", expect.any(Object));

    rerender(undefined);
    expect(result.current.progress).toBe(0);
    expect(result.current.message).toBe("");
    expect(result.current.error).toBeNull();
    expect(result.current.result).toBeNull();
  });

  it("ReportGeneration polls report readiness, preserves PDF blob URL, and renders Overview & Export Hub without Graph & Concepts", async () => {
    const statusSpy = vi.spyOn(api, "getReportStatus").mockResolvedValue({ status: "ready" });
    const blobSpy = vi
      .spyOn(api, "downloadReportBlob")
      .mockResolvedValue(new Blob(["%PDF-1.4\n"], { type: "application/pdf" }));
    vi.spyOn(api, "getHiPlotData").mockResolvedValue({
      total_rows: 20,
      sampled_rows: 20,
      dimensions: [],
      datapoints: [],
    });
    vi.spyOn(api, "getSqlSchema").mockResolvedValue({
      task_id: "t-rep",
      success: true,
      engine: "duckdb",
      total_tables: 0,
      tables: [],
    });

    const onComplete = vi.fn();
    const { rerender } = render(
      <ReportGeneration
        step="generating"
        taskId="t-rep"
        filename="quarterly_sales.csv"
        info={makeSampleInfo()}
        analysis={makeSampleAnalysis()}
        charts={{}}
        insights={makeSampleInsights()}
        onComplete={onComplete}
        onReset={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(statusSpy).toHaveBeenCalledWith("t-rep");
      expect(blobSpy).toHaveBeenCalledWith("t-rep");
      expect(onComplete).toHaveBeenCalledTimes(1);
    });

    expect(window.URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(window.URL.revokeObjectURL).not.toHaveBeenCalled();

    rerender(
      <ReportGeneration
        step="complete"
        taskId="t-rep"
        filename="quarterly_sales.csv"
        info={makeSampleInfo()}
        analysis={makeSampleAnalysis()}
        charts={{}}
        insights={makeSampleInsights()}
        onComplete={onComplete}
        onReset={vi.fn()}
      />,
    );

    expect(screen.getByText("Overview")).toBeInTheDocument();
    expect(screen.queryByText("Graph & Concepts")).not.toBeInTheDocument();
    expect(screen.getByText("Export Hub")).toBeInTheDocument();
    const tabsList = screen.getByRole("tablist");
    expect(tabsList.className).toContain("justify-between");
    expect(tabsList.className).toContain("flex-wrap");
    expect(tabsList.className).not.toContain("flex-nowrap");
  });
});
