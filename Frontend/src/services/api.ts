import type { IssueLedgerData, ApiResponse, InspectionResult, CleaningRulesMap } from "@/types/api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const REQUEST_TIMEOUT_MS = 45_000;

// ─── Response Types ─────────────────────────────────────────────────────────

export type JobTaskResult = ApiResponse | InspectionResult | { message?: string; stage?: string };

export interface StatusResponse {
    task_id: string;
    status: string;
    progress: number;
    message: string;
    result?: JobTaskResult | null;
    error?: string | null;
    report_download_url?: string | null;
}

export interface ReportStatusResponse {
    status: "ready" | "generating" | "not_started" | "failed";
    download_url?: string;
}

// ─── Client ─────────────────────────────────────────────────────────────────

async function fetchClient<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = new Headers(options.headers);
    if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
    }

    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let response: Response;
    try {
        response = await fetch(url, { ...options, headers, signal: options.signal || controller.signal });
    } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
            throw new Error("The request timed out. Please check the service and try again.");
        }
        throw new Error("Unable to reach the analysis service. Please check your connection and try again.");
    } finally {
        window.clearTimeout(timeout);
    }

    if (!response.ok) {
        let errorMessage = `HTTP Error ${response.status}`;
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch {
                errorMessage = response.statusText || errorMessage;
            }
        } else {
            try {
                const text = await response.text();
                if (text && text.length < 200) {
                    errorMessage = text;
                } else {
                    errorMessage = response.statusText || errorMessage;
                }
            } catch {
                errorMessage = response.statusText || errorMessage;
            }
        }
        throw new Error(errorMessage);
    }

    return response.json();
}

export const api = {
    /**
     * Upload a file for processing (cleaning, analysis, charts, insights).
     */
    uploadFile: async (file: File): Promise<{ task_id: string; message: string }> => {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            let errorMessage = `Upload failed: ${response.statusText}`;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                try {
                    const data = await response.json();
                    errorMessage = data.detail || errorMessage;
                } catch {}
            } else {
                try {
                    const text = await response.text();
                    if (text && text.length < 200) {
                        errorMessage = text;
                    }
                } catch {}
            }
            throw new Error(errorMessage);
        }
        return response.json();
    },

    getTaskStatus: async (taskId: string): Promise<StatusResponse> => {
        return fetchClient<StatusResponse>(`/status/${taskId}`);
    },

    /**
     * Chat with the processed report (RAG).
     */
    chatWithJob: async (
        taskId: string, 
        question: string,
        chatHistory?: { role: string; content: string }[]
    ): Promise<{ answer: string; sources: string[]; suggested_followups?: string[] }> => {
        return fetchClient<{ answer: string; sources: string[]; suggested_followups?: string[] }>(`/jobs/${taskId}/chat`, {
            method: "POST",
            body: JSON.stringify({ question, chat_history: chatHistory }),
        });
    },

    /**
     * Stream RAG chat tokens in real-time.
     */
    streamChatWithJob: async (
        taskId: string,
        question: string,
        onToken: (token: string) => void,
        onMetadata: (metadata: { sources: string[]; suggested_followups?: string[] }) => void,
        onDone: () => void,
        onError: (err: Error) => void,
        chatHistory?: { role: string; content: string }[]
    ): Promise<void> => {
        try {
            const response = await fetch(`${API_BASE_URL}/jobs/${taskId}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, chat_history: chatHistory })
            });

            if (!response.ok || !response.body) {
                throw new Error(`Chat stream failed (${response.status})`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    try {
                        const parsed = JSON.parse(trimmed);
                        if (parsed.type === "metadata") {
                            onMetadata({
                                sources: parsed.sources || [],
                                suggested_followups: parsed.suggested_followups
                            });
                        } else if (parsed.type === "token") {
                            onToken(parsed.token || "");
                        } else if (parsed.type === "done") {
                            onDone();
                            return;
                        } else if (parsed.type === "error") {
                            onError(new Error(parsed.error || "Streaming error"));
                            return;
                        }
                    } catch {
                        console.warn("Failed to parse chat stream chunk:", trimmed);
                    }
                }
            }
            onDone();
        } catch (err) {
            onError(err instanceof Error ? err : new Error(String(err)));
        }
    },

    /**
     * Generate PDF on the server using stored analysis results.
     */
    generatePersistentReport: async (taskId: string): Promise<{ message: string; path: string | null }> => {
        return fetchClient<{ message: string; path: string | null }>(`/jobs/${taskId}/report`, {
            method: "POST",
        });
    },

    /**
     * Check if the PDF report is ready for download.
     */
    getReportStatus: async (taskId: string): Promise<ReportStatusResponse> => {
        return fetchClient<ReportStatusResponse>(`/jobs/${taskId}/report/status`);
    },

    /**
     * Download the already generated PDF.
     */
    downloadReportBlob: async (taskId: string): Promise<Blob> => {
        const response = await fetch(`${API_BASE_URL}/jobs/${taskId}/report`);
        if (!response.ok) throw new Error("Failed to download report");
        return response.blob();
    },

    /**
     * Upload and join multiple files on a primary key column.
     */
    uploadJoinedFiles: async (files: File[], joinKey: string, joinType: string = "inner"): Promise<{ task_id: string; message: string }> => {
        const formData = new FormData();
        files.forEach(f => formData.append("files", f));
        formData.append("join_key", joinKey);
        formData.append("join_type", joinType);

        const response = await fetch(`${API_BASE_URL}/upload/join`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            let errorMessage = `Joined upload failed: ${response.statusText}`;
            try {
                const data = await response.json();
                errorMessage = data.detail || errorMessage;
            } catch {}
            throw new Error(errorMessage);
        }
        return response.json();
    },

    /**
     * Download multi-format export (CSV, Parquet, HTML).
     */
    downloadExportBlob: async (taskId: string, format: "csv" | "parquet" | "html"): Promise<Blob> => {
        const response = await fetch(`${API_BASE_URL}/jobs/${taskId}/export/${format}`);
        if (!response.ok) throw new Error(`Failed to export ${format.toUpperCase()} file`);
        return response.blob();
    },

    /**
     * Stage 2: Resume analysis with cleaning rules.
     */
    startAnalysis: async (taskId: string, rules: CleaningRulesMap): Promise<{ message: string }> => {
        return fetchClient<{ message: string }>(`/jobs/${taskId}/analyze`, {
            method: "POST",
            body: JSON.stringify({ rules }),
        });
    },

    /**
     * Get issues ledger for a job.
     */
    getIssues: async (taskId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues`);
    },

    /**
     * Approve a single issue.
     */
    approveIssue: async (taskId: string, issueId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues/${issueId}/approve`, {
            method: "POST",
            body: JSON.stringify({}),
        });
    },

    /**
     * Reject a single issue.
     */
    rejectIssue: async (taskId: string, issueId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues/${issueId}/reject`, {
            method: "POST",
            body: JSON.stringify({}),
        });
    },

    /**
     * Approve all pending issues.
     */
    approveAllIssues: async (taskId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues/approve-all`, {
            method: "POST",
        });
    },

    /**
     * Reject all pending issues.
     */
    rejectAllIssues: async (taskId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues/reject-all`, {
            method: "POST",
        });
    },

    /**
     * Lock the issue ledger.
     */
    lockIssues: async (taskId: string): Promise<IssueLedgerData> => {
        return fetchClient<IssueLedgerData>(`/jobs/${taskId}/issues/lock`, {
            method: "POST",
        });
    },

    /**
     * Get the WebSocket URL for real-time status updates.
     */
    getWebSocketUrl: (taskId: string): string => {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = API_BASE_URL.replace(/^https?:\/\//, "");
        return `${protocol}//${host}/ws/status/${taskId}`;
    },
};
