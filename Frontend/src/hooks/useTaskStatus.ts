import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "@/services/api";

/**
 * Status response from the backend.
 */
export interface TaskStatusData {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  result?: Record<string, any> | null;
  error?: string | null;
  report_download_url?: string | null;
}

interface UseTaskStatusOptions {
  /** Polling interval in ms (fallback if WS fails). Default: 3000 */
  pollingInterval?: number;
  /** Whether to auto-connect. Default: true */
  enabled?: boolean;
}

/**
 * Custom hook for real-time task status updates.
 * 
 * Strategy:
 * 1. Attempts WebSocket connection first (real-time, low latency)
 * 2. Falls back to HTTP polling if WebSocket fails
 * 3. Auto-cleans up on unmount or task completion
 * 
 * Usage:
 * ```tsx
 * const { status, progress, message, result, error, isConnected } = useTaskStatus(taskId);
 * ```
 */
export function useTaskStatus(
  taskId: string | null,
  options: UseTaskStatusOptions = {}
) {
  const { pollingInterval = 3000, enabled = true } = options;
  
  const [data, setData] = useState<TaskStatusData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionType, setConnectionType] = useState<"ws" | "polling" | "none">("none");
  
  const wsRef = useRef<WebSocket | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mountedRef = useRef(true);

  // Terminal states — stop listening when reached
  const isTerminal = data?.status === "COMPLETED" || data?.status === "FAILED";

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setIsConnected(false);
    setConnectionType("none");
  }, []);

  // Start HTTP polling fallback
  const startPolling = useCallback(() => {
    if (!taskId || pollingRef.current) return;
    
    setConnectionType("polling");
    setIsConnected(true);
    
    const poll = async () => {
      if (!mountedRef.current) return;
      try {
        const response = await api.getTaskStatus(taskId);
        if (mountedRef.current) {
          setData(response);
          // Stop polling on terminal states
          if (response.status === "COMPLETED" || response.status === "FAILED") {
            cleanup();
          }
        }
      } catch (err) {
        console.error("[useTaskStatus] Polling error:", err);
      }
    };
    
    // Immediate first poll
    poll();
    pollingRef.current = setInterval(poll, pollingInterval);
  }, [taskId, pollingInterval, cleanup]);

  // Try WebSocket first, fall back to polling
  useEffect(() => {
    if (!taskId || !enabled || isTerminal) return;
    
    mountedRef.current = true;
    
    // Construct WS URL from current page location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = import.meta.env.VITE_API_URL 
      ? new URL(import.meta.env.VITE_API_URL).host 
      : "localhost:8000";
    const wsUrl = `${protocol}//${host}/api/ws/status/${taskId}`;
    
    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        if (mountedRef.current) {
          setIsConnected(true);
          setConnectionType("ws");
          console.log("[useTaskStatus] WebSocket connected");
        }
      };
      
      ws.onmessage = (event) => {
        if (!mountedRef.current) return;
        try {
          const parsed: TaskStatusData = JSON.parse(event.data);
          setData(parsed);
          
          if (parsed.status === "COMPLETED" || parsed.status === "FAILED") {
            cleanup();
          }
        } catch (err) {
          console.error("[useTaskStatus] Failed to parse WS message:", err);
        }
      };
      
      ws.onerror = () => {
        console.warn("[useTaskStatus] WebSocket error, falling back to polling");
        ws.close();
      };
      
      ws.onclose = () => {
        if (mountedRef.current && !isTerminal) {
          // WebSocket closed unexpectedly — fallback to polling
          wsRef.current = null;
          startPolling();
        }
      };
      
    } catch (err) {
      // WebSocket constructor failed — fallback to polling
      console.warn("[useTaskStatus] WebSocket unavailable, using polling");
      startPolling();
    }
    
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, [taskId, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    /** Full status data object */
    data,
    /** Current status string (e.g., "PROCESSING", "COMPLETED") */
    status: data?.status ?? null,
    /** Progress percentage (0-100) */
    progress: data?.progress ?? 0,
    /** Human-readable status message */
    message: data?.message ?? "",
    /** Final result (only when COMPLETED) */
    result: data?.result ?? null,
    /** Error message (only when FAILED) */
    error: data?.error ?? null,
    /** Whether actively receiving updates */
    isConnected,
    /** Connection method: "ws" | "polling" | "none" */
    connectionType,
    /** Whether task has reached a terminal state */
    isTerminal,
  };
}
