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
        // Try to parse error message from JSON
        let errorMessage = `HTTP Error ${response.status}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.detail || errorData.message || errorMessage;
        } catch (e) {
            // Ignore JSON parse error, use status text
            errorMessage = response.statusText || errorMessage;
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
            try {
                const data = await response.json();
                errorMessage = data.detail || errorMessage;
            } catch (e) {}
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
};
