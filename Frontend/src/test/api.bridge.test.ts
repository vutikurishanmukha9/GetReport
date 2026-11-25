import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/services/api";

describe("api bridge contracts", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("downloadReportBlob rejects HTTP 202 and JSON payloads instead of returning corrupt PDF blobs", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ status: "processing", message: "Still compiling charts" }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    );
    await expect(api.downloadReportBlob("task-202")).rejects.toThrow("Still compiling charts");

    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ detail: "Report generation pending" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    await expect(api.downloadReportBlob("task-json")).rejects.toThrow("Report generation pending");

    fetchSpy.mockResolvedValueOnce(
      new Response("%PDF-1.4\n1 0 obj\n<<>>\nendobj", {
        status: 200,
        headers: { "Content-Type": "application/pdf" },
      }),
    );
    const pdfBlob = await api.downloadReportBlob("task-ready");
    expect(pdfBlob.size).toBeGreaterThan(0);
  });

  it("downloadExportBlob requests xlsx format and surfaces JSON error details", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    fetchSpy.mockResolvedValueOnce(
      new Response("mock-xlsx-bytes", {
        status: 200,
        headers: { "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" },
      }),
    );
    const blob = await api.downloadExportBlob("task-xlsx", "xlsx");
    expect(blob.size).toBeGreaterThan(0);
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/jobs\/task-xlsx\/export\/xlsx$/),
      expect.objectContaining({ headers: expect.any(Object) }),
    );

    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "Cleaned workbook expired" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await expect(api.downloadExportBlob("task-xlsx", "xlsx")).rejects.toThrow("Cleaned workbook expired");
  });

  it("modifyIssue sends custom fix_code and note to /issues/:id/modify", async () => {
    const responsePayload = {
      issues: [],
      locked: false,
      locked_at: null,
      created_at: "2026-01-01T00:00:00Z",
      message: "Issue modified",
      issue_id: "iss-1",
      fix_code: "df = df.dropna()",
      summary: { pending: 0, approved: 0, rejected: 0, modified: 1, total: 1 },
    };
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify(responsePayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const res = await api.modifyIssue("task-mod", "iss-1", "df = df.dropna()", "drop sparse rows");
    expect(res.issue_id).toBe("iss-1");
    expect(res.summary.modified).toBe(1);
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/jobs\/task-mod\/issues\/iss-1\/modify$/),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ fix_code: "df = df.dropna()", note: "drop sparse rows" }),
      }),
    );
  });

  it("getHiPlotData encodes query parameters cleanly", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({ total_rows: 100, sampled_rows: 100, dimensions: [], datapoints: [] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const hiplot = await api.getHiPlotData("task/with space", 500, ["col_a", "col_b"]);
    expect(hiplot.sampled_rows).toBe(100);
    expect(fetchSpy.mock.calls[0]?.[0]).toMatch(/\/analytics\/explore\/task%2Fwith%20space\/hiplot\?.*max_rows=500/);
  });

  it("querySupersetExplore forwards AbortSignal cancellation to fetch", async () => {
    const controller = new AbortController();
    let capturedSignal: AbortSignal | null | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((_input, init) => {
      capturedSignal = init?.signal;
      controller.abort();
      return Promise.reject(new DOMException("The operation was aborted.", "AbortError"));
    });

    await expect(
      api.querySupersetExplore(
        {
          task_id: "task-bi",
          chart_type: "table",
          dimension: "dept",
          aggregate: "count",
          row_limit: 50,
          sort_order: "desc",
          color_palette: "emerald",
        },
        controller.signal,
      ),
    ).rejects.toThrow(/abort/i);

    expect(capturedSignal).toBeDefined();
    expect(capturedSignal?.aborted).toBe(true);
  });
});
