import type { ApiResponse, AnalysisResult, Charts, CleaningRulesMap } from "@/types/api";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ─── Response Types ─────────────────────────────────────────────────────────

export interface StatusResponse {
    task_id: string;
    status: string;
    progress: number;
    message: string;
    result?: Record<string, any> | null;
    error?: string | null;
    report_download_url?: string | null;
}

export interface ReportStatusResponse {
    status: "ready" | "generating" | "not_started";
    path?: string;
}

// ─── Client ─────────────────────────────────────────────────────────────────

async function fetchClient<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const headers = {
        "Content-Type": "application/json",
        ...options.headers,
    };

    const response = await fetch(url, {
        ...options,
        headers,
    });

    if (!response.ok) {
        let errorMessage = `HTTP Error ${response.status}`;
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
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
            } catch (e) {
                errorMessage = response.statusText || errorMessage;
            }
        }
        throw new Error(errorMessage);
    }

    // Handle Blob responses specially
    if (options.headers && (options.headers as any)["Content-Type"] === undefined && endpoint.includes("/report") && options.method === "GET") {
         // This is a bit hacky but covers the downloadReportBlob case where we shouldn't parse JSON
         // Actually, let's handle it in the specific method
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
            // Fetch automatically sets Content-Type for FormData, do NOT set it manually
        });

        if (!response.ok) {
            let errorMessage = `Upload failed: ${response.statusText}`;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                try {
                    const data = await response.json();
                    errorMessage = data.detail || errorMessage;
                } catch (e) {}
            } else {
                try {
                    const text = await response.text();
                    if (text && text.length < 200) {
                        errorMessage = text;
                    }
                } catch (e) {}
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
    chatWithJob: async (taskId: string, question: string): Promise<{ answer: string; sources: string[] }> => {
        return fetchClient<{ answer: string; sources: string[] }>(`/jobs/${taskId}/chat`, {
            method: "POST",
            body: JSON.stringify({ question }),
        });
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
     * Stage 2: Resume analysis with cleaning rules.
     */
    startAnalysis: async (taskId: string, rules: Record<string, any>): Promise<{ message: string }> => {
        return fetchClient<{ message: string }>(`/jobs/${taskId}/analyze`, {
            method: "POST",
            body: JSON.stringify({ rules }),
        });
    },

    /**
     * Get issues ledger for a job.
     */
    getIssues: async (taskId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues`);
    },

    /**
     * Approve a single issue.
     */
    approveIssue: async (taskId: string, issueId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues/${issueId}/approve`, {
            method: "POST",
            body: JSON.stringify({}),
        });
    },

    /**
     * Reject a single issue.
     */
    rejectIssue: async (taskId: string, issueId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues/${issueId}/reject`, {
            method: "POST",
            body: JSON.stringify({}),
        });
    },

    /**
     * Approve all pending issues.
     */
    approveAllIssues: async (taskId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues/approve-all`, {
            method: "POST",
        });
    },

    /**
     * Reject all pending issues.
     */
    rejectAllIssues: async (taskId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues/reject-all`, {
            method: "POST",
        });
    },

    /**
     * Lock the issue ledger.
     */
    lockIssues: async (taskId: string): Promise<any> => {
        return fetchClient<any>(`/jobs/${taskId}/issues/lock`, {
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
