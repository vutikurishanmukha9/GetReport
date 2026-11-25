import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { api } from "@/services/api";
import { DuckDBSqlStudio } from "@/components/DuckDBSqlStudio";
import { Analytics } from "@/pages/Analytics";
import type { DatasetInfo } from "@/types/api";

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

function makeInfo(): DatasetInfo {
  return {
    rows: 25,
    columns: ["dept", "revenue"],
    dtypes: { dept: "VARCHAR", revenue: "DOUBLE" },
    summary: { revenue: { count: 25, mean: 180 } },
    preview: [
      { dept: "Engineering", revenue: 240 },
      { dept: "Sales", revenue: 120 },
    ],
    missing_values: {
      dept: { count: 0, percentage: 0 },
      revenue: { count: 0, percentage: 0 },
    },
    duplicate_rows: 0,
    numeric_columns: ["revenue"],
    categorical_columns: ["dept"],
    memory_usage_mb: 0.2,
  };
}

describe("DuckDB SQL Studio UI/UX & integrations", () => {
  beforeEach(() => {
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
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      writable: true,
      configurable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders dual-theme workbench, hydrates schema & saved KPIs, executes queries with limit, formats SQL, pins KPI, and handles errors", async () => {
    const schemaSpy = vi.spyOn(api, "getSqlSchema").mockResolvedValue({
      task_id: "t-sql",
      success: true,
      engine: "DuckDB-InProcess",
      total_tables: 1,
      tables: [
        {
          table_name: "dataset",
          row_count: 25,
          column_count: 2,
          columns: [
            {
              name: "dept",
              type: "VARCHAR",
              nullable: false,
              sample_values: ["Engineering", "Sales"],
            },
            {
              name: "revenue",
              type: "DOUBLE",
              nullable: true,
              sample_values: ["240.5", "120"],
            },
          ],
        },
      ],
    });

    vi.spyOn(api, "getGoldenQueries").mockResolvedValue([
      {
        id: "gq-existing",
        task_id: "t-sql",
        question: "Existing Revenue Benchmark",
        sql_query: "SELECT dept, SUM(revenue) FROM dataset GROUP BY dept;",
        description: "Baseline",
        created_at: "2026-01-01T00:00:00Z",
      },
    ]);

    const querySpy = vi.spyOn(api, "querySql").mockResolvedValueOnce({
      task_id: "t-sql",
      sql: "SELECT dept, SUM(revenue) AS total_rev FROM dataset GROUP BY dept;",
      columns: ["dept", "total_rev"],
      records: [
        { dept: "Engineering", total_rev: 2400 },
        { dept: "Sales", total_rev: 1200 },
      ],
      total_returned: 2,
      capped: false,
      execution_ms: 14,
    });

    const saveKpiSpy = vi.spyOn(api, "saveGoldenQuery").mockResolvedValueOnce({
      id: "gq-new-1",
      task_id: "t-sql",
      question: "Revenue by Dept",
      sql_query: "SELECT * FROM dataset WHERE dept = 'Engineering' LIMIT 25;",
      description: "Saved from DuckDB SQL Studio",
      created_at: "2026-01-02T00:00:00Z",
    });

    render(<DuckDBSqlStudio taskId="t-sql" datasetName="sales_q4.csv" info={makeInfo()} />);

    await waitFor(() => {
      expect(schemaSpy).toHaveBeenCalledWith("t-sql");
    });

    expect(screen.getByText("Ready to Query In-Memory Arrow View")).toBeInTheDocument();
    expect(screen.getByText("Top dept")).toBeInTheDocument();
    expect(screen.getByText("Stats: revenue")).toBeInTheDocument();

    // Expand column details & insert sample literal
    fireEvent.click(screen.getByLabelText("Inspect column dept"));
    expect(await screen.findByText("Engineering")).toBeInTheDocument();

    // Change Max rows limit and Format SQL
    fireEvent.change(screen.getByLabelText("Max rows:"), { target: { value: "250" } });
    const textarea = screen.getByLabelText("SQL Query Input");
    fireEvent.change(textarea, {
      target: { value: "select dept, sum(revenue) as total_rev from dataset group by dept;" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^format$/i }));
    expect(textarea).toHaveValue(
      "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept;",
    );

    // Run query & verify results console + JSON switch + custom KPI pin
    fireEvent.click(screen.getByRole("button", { name: /^run query$/i }));

    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledWith(
        "t-sql",
        "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept;",
        250,
      );
      expect(screen.getByText("14ms")).toBeInTheDocument();
      expect(screen.getByText("2,400")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^json$/i }));
    expect(screen.getByText(/JSON Preview/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^table$/i }));

    fireEvent.click(screen.getByRole("button", { name: /save kpi/i }));
    const kpiTitle = await screen.findByLabelText("Benchmark Title:");
    fireEvent.change(kpiTitle, { target: { value: "Revenue by Dept" } });
    fireEvent.click(screen.getByRole("button", { name: /pin benchmark/i }));

    await waitFor(() => {
      expect(saveKpiSpy).toHaveBeenCalledWith(
        "t-sql",
        "Revenue by Dept",
        "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept;",
        "Saved from DuckDB SQL Studio",
      );
    });

    // Inspector Saved tab displays both existing and newly pinned KPIs
    fireEvent.click(screen.getByRole("button", { name: /^saved\s+\d+$/i }));
    expect(await screen.findByText("Revenue by Dept")).toBeInTheDocument();
    expect(screen.getByText("Existing Revenue Benchmark")).toBeInTheDocument();

    // Error state displays actionable security hint and recovery button
    querySpy.mockRejectedValueOnce(new Error("Security Violation: DROP is not permitted."));
    fireEvent.click(screen.getByRole("button", { name: /^run query$/i }));

    await waitFor(() => {
      expect(screen.getByText("Query Evaluation Failed")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /run describe dataset/i })).toBeInTheDocument();
    });
  });

  it("Analytics page switches between Visual BI Studio and DuckDB SQL Studio cleanly via URL params", async () => {
    vi.spyOn(api, "listRecentDatasets").mockResolvedValue({
      datasets: [
        {
          task_id: "t-analytics-1",
          filename: "finance_2026.csv",
          rows: 100,
          columns_count: 3,
          columns: ["region", "spend", "month"],
          numeric_columns: ["spend"],
          categorical_columns: ["region"],
          created_at: "2026-01-01T00:00:00Z",
        },
      ],
    });
    vi.spyOn(api, "getSqlSchema").mockResolvedValue({
      task_id: "t-analytics-1",
      success: true,
      engine: "DuckDB-InProcess",
      total_tables: 1,
      tables: [
        {
          table_name: "dataset",
          row_count: 100,
          column_count: 2,
          columns: [
            { name: "region", type: "VARCHAR", nullable: false, sample_values: ["EMEA"] },
            { name: "spend", type: "DOUBLE", nullable: false, sample_values: ["500"] },
          ],
        },
      ],
    });
    vi.spyOn(api, "getGoldenQueries").mockResolvedValue([]);

    render(
      <MemoryRouter initialEntries={["/analytics?taskId=t-analytics-1&studio=sql&sql=SELECT%20COUNT(*)%20FROM%20dataset;"]}>
        <Analytics />
      </MemoryRouter>,
    );

    expect(await screen.findByText("DuckDB Analytical SQL Studio")).toBeInTheDocument();
    expect(await screen.findByLabelText("SQL Query Input")).toHaveValue(
      "SELECT COUNT(*) FROM dataset;",
    );
  });
});
