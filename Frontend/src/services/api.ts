import axios from "axios";
import type { ApiResponse, AnalysisResult, Charts } from "@/types/api";

const API_BASE_URL = "http://localhost:8000/api";

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
            timeout: 60000, // 60s timeout for large files/AI
        });
        return response.data;
    },

    getTaskStatus: async (taskId: string): Promise<any> => {
        const response = await apiClient.get(`/status/${taskId}`);
        return response.data;
    },

    /**
     * Generate and download the PDF report.
     */
    generateReport: async (
        filename: string,
        analysis: AnalysisResult,
        charts: Charts,
        insightsText: string // We pass insights separately to merge into analysis if needed?
        // Actually endpoints.py takes { filename, analysis, charts }.
        // And report_generator looks for insights in analysis_results['insights']?
    ): Promise<Blob> => {

        // Ensure insights are included in the analysis object for the PDF generator
        // @ts-ignore
        const analysisWithInsights = { ...analysis };
        // @ts-ignore
        if (insightsText && !analysisWithInsights.insights) {
            // @ts-ignore
            analysisWithInsights.insights = insightsText;
        }

        const payload = {
            filename,
            analysis: analysisWithInsights,
            charts,
        };

        const response = await apiClient.post("/generate-report", payload, {
            responseType: "blob", // Important for file download
            timeout: 30000,
        });
        return response.data;
    },
};
