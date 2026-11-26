import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { EditorView } from "@codemirror/view";
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

function mockBaseSchemaAndSaved() {
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

  vi.spyOn(api, "getSqlColumnProfile").mockResolvedValue({
    task_id: "t-sql",
    success: true,
    column: "dept",
    column_type: "VARCHAR",
    total_records: 25,
    non_null_count: 25,
    null_count: 0,
    approx_distinct: 2,
    min_value: "Engineering",
    max_value: "Sales",
    mean_value: null,
    std_deviation: null,
  });

  return { schemaSpy };
}

function clickButtonText(label: string) {
  const el = screen.getByText(label);
  const btn = el.closest("button");
  if (!btn) {
    throw new Error(`Expected button ancestor for "${label}"`);
  }
  fireEvent.click(btn);
}

describe("DuckDB SQL Studio UI/UX & integrations", () => {
  beforeEach(() => {
    window.localStorage.clear();
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
    if (!Range.prototype.getClientRects) {
      Range.prototype.getClientRects = () => ({
        length: 0,
        item: () => null,
        [Symbol.iterator]: function* () {},
      });
    }
    if (!Range.prototype.getBoundingClientRect) {
      Range.prototype.getBoundingClientRect = () => new DOMRect(0, 0, 0, 0);
    }
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("hydrates schema profile, updates limit Select, formats CodeMirror SQL, executes query, and switches Plan, Chart & JSON views", async () => {
    const { schemaSpy } = mockBaseSchemaAndSaved();
    const querySpy = vi.spyOn(api, "querySql").mockResolvedValueOnce({
      task_id: "t-sql",
      sql: "SELECT dept, SUM(revenue) AS total_rev FROM dataset GROUP BY dept;",
      columns: ["dept", "total_rev"],
      column_types: ["VARCHAR", "DOUBLE"],
      records: [
        { dept: "Engineering", total_rev: 2400 },
        { dept: "Sales", total_rev: 1200 },
      ],
      total_returned: 2,
      capped: false,
      execution_ms: 14,
    });
    vi.spyOn(api, "explainSql").mockResolvedValueOnce({
      task_id: "t-sql",
      sql: "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept",
      plan: [{ name: "PROJECTION", children: [{ name: "HASH_GROUP_BY", extra_info: { "Estimated Cardinality": "2" } }] }],
      raw_plan: "[]",
    });

    render(<DuckDBSqlStudio taskId="t-sql" datasetName="sales_q4.csv" info={makeInfo()} />);

    await waitFor(() => {
      expect(schemaSpy).toHaveBeenCalledWith("t-sql");
    });

    expect(screen.getByText("Ready to Query In-Memory Arrow View")).toBeInTheDocument();
    expect(screen.getByText("Top dept")).toBeInTheDocument();
    expect(screen.getByText("Stats: revenue")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Inspect column dept"));
    expect(await screen.findByText("Engineering")).toBeInTheDocument();
    expect(await screen.findByText(/Fill: 100%/)).toBeInTheDocument();

    fireEvent.keyDown(screen.getByLabelText("Max rows:"), { key: "Enter" });
    fireEvent.click(await screen.findByText("250 rows"));

    await waitFor(() => {
      expect(EditorView.findFromDOM(screen.getByLabelText("SQL Query Input"))).not.toBeNull();
    });
    const cmView = EditorView.findFromDOM(screen.getByLabelText("SQL Query Input"));
    expect(cmView).not.toBeNull();
    act(() => {
      cmView?.dispatch({
        changes: {
          from: 0,
          to: cmView.state.doc.length,
          insert: "select dept, sum(revenue) as total_rev from dataset group by dept;",
        },
      });
    });

    clickButtonText("Format");
    await waitFor(() => {
      expect(EditorView.findFromDOM(screen.getByLabelText("SQL Query Input"))?.state.doc.toString()).toBe(
        "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept;",
      );
    });

    clickButtonText("Run Query");
    await waitFor(() => {
      expect(querySpy).toHaveBeenCalledWith(
        "t-sql",
        "select dept, sum(revenue) as total_rev\nFROM dataset\nGROUP BY dept;",
        250,
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      );
      expect(screen.getByText("14ms")).toBeInTheDocument();
      expect(screen.getByText("2,400")).toBeInTheDocument();
    });

    clickButtonText("Explain");
    expect(await screen.findByText("Physical Execution Plan")).toBeInTheDocument();
    expect(screen.getByText("HASH_GROUP_BY")).toBeInTheDocument();

    clickButtonText("Chart");
    expect(await screen.findByLabelText("SQL Result Chart")).toBeInTheDocument();

    clickButtonText("JSON");
    expect(screen.getByText(/JSON Preview/)).toBeInTheDocument();
    clickButtonText("Table");

    fireEvent.click(screen.getByTitle("Inspect row #1"));
    expect(await screen.findByText("Row 1 Inspector")).toBeInTheDocument();
  }, 15000);

  it("pins Golden KPIs, manages Saved tab & worksheet tabs, and renders error recovery hints", async () => {
    mockBaseSchemaAndSaved();
    vi.spyOn(api, "querySql")
      .mockResolvedValueOnce({
        task_id: "t-sql",
        sql: "SELECT * FROM dataset LIMIT 25;",
        columns: ["dept", "revenue"],
        records: [{ dept: "Engineering", revenue: 2400 }],
        total_returned: 1,
        capped: true,
        execution_ms: 9,
      })
      .mockRejectedValueOnce(new Error("Security Violation: DROP is not permitted."));

    const saveKpiSpy = vi.spyOn(api, "saveGoldenQuery").mockResolvedValueOnce({
      id: "gq-new-1",
      task_id: "t-sql",
      question: "Revenue by Dept",
      sql_query: "SELECT * FROM dataset LIMIT 25;",
      description: "Saved from DuckDB SQL Studio",
      created_at: "2026-01-02T00:00:00Z",
    });

    render(<DuckDBSqlStudio taskId="t-sql" datasetName="sales_q4.csv" info={makeInfo()} />);

    fireEvent.click(screen.getByLabelText("New worksheet tab"));
    expect(screen.getByText("Query 2")).toBeInTheDocument();

    clickButtonText("Run Query");
    expect(await screen.findByText("9ms")).toBeInTheDocument();
    expect(screen.getByText(/Capped at 500/)).toBeInTheDocument();

    clickButtonText("Save KPI");
    const kpiTitle = await screen.findByLabelText("Benchmark Title:");
    fireEvent.change(kpiTitle, { target: { value: "Revenue by Dept" } });
    clickButtonText("Pin Benchmark");

    await waitFor(() => {
      expect(saveKpiSpy).toHaveBeenCalledWith(
        "t-sql",
        "Revenue by Dept",
        "SELECT * FROM dataset LIMIT 25;",
        "Saved from DuckDB SQL Studio",
      );
    });

    clickButtonText("Saved");
    expect(await screen.findByText("Revenue by Dept")).toBeInTheDocument();
    expect(screen.getByText("Existing Revenue Benchmark")).toBeInTheDocument();

    clickButtonText("Run Query");
    await waitFor(() => {
      expect(screen.getByText("Query Evaluation Failed")).toBeInTheDocument();
      expect(screen.getByText("Run DESCRIBE dataset")).toBeInTheDocument();
    });
  }, 15000);

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
    await waitFor(() => {
      const el = screen.getByLabelText("SQL Query Input");
      expect(EditorView.findFromDOM(el)?.state.doc.toString()).toBe("SELECT COUNT(*) FROM dataset;");
    });
  });
});
