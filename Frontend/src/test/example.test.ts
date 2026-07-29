import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "@/services/api";

describe("API client", () => {
  afterEach(() => vi.restoreAllMocks());

  it("uploads a file as multipart data without forcing a content type", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ task_id: "task-1", message: "queued" }), { status: 200 }),
    );

    await expect(api.uploadFile(new File(["name,value"], "report.csv", { type: "text/csv" }))).resolves.toEqual({
      task_id: "task-1",
      message: "queued",
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/upload",
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    );
  });

  it("turns JSON API failures into useful errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "Task not found" }), {
        status: 404,
        headers: { "content-type": "application/json" },
      }),
    );

    await expect(api.getTaskStatus("missing")).rejects.toThrow("Task not found");
  });
});
