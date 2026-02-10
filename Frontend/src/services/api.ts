import axios from "axios";
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

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-Type": "application/json",
    },
});

export const api = {
    /**
     * Upload a file for processing (cleaning, analysis, charts, insights).
     */
    uploadFile: async (file: File): Promise<{ task_id: string; message: string }> => {
        const formData = new FormData();
        formData.append("file", file);

        const response = await apiClient.post("/upload", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
            timeout: 60000,
        });
        return response.data;
    },

    getTaskStatus: async (taskId: string): Promise<StatusResponse> => {
        const response = await apiClient.get(`/status/${taskId}`);
        return response.data;
    },

    /**
     * Chat with the processed report (RAG).
     */
    chatWithJob: async (taskId: string, question: string): Promise<{ answer: string; sources: string[] }> => {
        const response = await apiClient.post(`/jobs/${taskId}/chat`, { question });
        return response.data;
    },

    /**
     * Generate PDF on the server using stored analysis results.
     */
    generatePersistentReport: async (taskId: string): Promise<{ message: string; path: string | null }> => {
        const response = await apiClient.post(`/jobs/${taskId}/report`);
        return response.data;
    },

    /**
     * Check if the PDF report is ready for download.
     */
    getReportStatus: async (taskId: string): Promise<ReportStatusResponse> => {
        const response = await apiClient.get(`/jobs/${taskId}/report/status`);
        return response.data;
    },

    /**
     * Download the already generated PDF.
     */
    downloadReportBlob: async (taskId: string): Promise<Blob> => {
        const response = await apiClient.get(`/jobs/${taskId}/report`, {
            responseType: "blob",
        });
        return response.data;
    },

    /**
     * Stage 2: Resume analysis with cleaning rules.
     */
    startAnalysis: async (taskId: string, rules: Record<string, any>): Promise<{ message: string }> => {
        const response = await apiClient.post(`/jobs/${taskId}/analyze`, { rules });
        return response.data;
    },
};
