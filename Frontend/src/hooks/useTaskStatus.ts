import { useState, useEffect, useRef, useCallback } from 'react';
import { api, StatusResponse } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

export type TaskStatus = 'PENDING' | 'PROCESSING' | 'WAITING_FOR_USER' | 'COMPLETED' | 'FAILED';

interface UseTaskStatusResult {
  status: TaskStatus | 'CONNECTING' | 'DISCONNECTED';
  progress: number;
  message: string;
  result: any | null;
  error: string | null;
  isConnected: boolean;
  connect: (taskId: string) => void;
  disconnect: () => void;
}

export const useTaskStatus = (activeTaskId?: string): UseTaskStatusResult => {
  const [status, setStatus] = useState<UseTaskStatusResult['status']>('CONNECTING');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const taskIdRef = useRef<string | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  // Auto-connect if activeTaskId is provided
  useEffect(() => {
    if (activeTaskId) {
      connect(activeTaskId);
    } else {
      disconnect();
    }
  }, [activeTaskId]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setIsConnected(false);
    setStatus('DISCONNECTED');
  }, []);

  const connect = useCallback((taskId: string) => {
    // Prevent duplicate connections
    if (wsRef.current?.readyState === WebSocket.OPEN && taskIdRef.current === taskId) {
      return;
    }

    disconnect();
    taskIdRef.current = taskId;
    setStatus('CONNECTING');

    const url = api.getWebSocketUrl(taskId);
    console.log(`Connecting to WebSocket: ${url}`);

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket Connected");
      setIsConnected(true);
      setStatus('PROCESSING'); // Assume processing initially or wait for message
      // Reset retry logic if any
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Expected format: { task_id, status, progress, message, result?, error? }

        if (data.status) setStatus(data.status.toUpperCase() as TaskStatus);
        if (data.progress !== undefined) setProgress(data.progress);
        if (data.message) setMessage(data.message);
        if (data.result) setResult(data.result);
        if (data.error) setError(data.error);

        if (['COMPLETED', 'FAILED'].includes(data.status?.toUpperCase())) {
          ws.close(); // Clean close on terminal state
        }
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e);
      }
    };

    ws.onerror = (e) => {
      console.error("WebSocket Error:", e);
      setError("Connection error");
      // Do not close here, let onclose handle it
    };

    ws.onclose = (event) => {
      console.log("WebSocket Disconnected", event.code, event.reason);
      setIsConnected(false);
      wsRef.current = null;

      // Reconnect logic?
      // If checking status of a long-running job, we should retry.
      // But if the job explicitly completed/failed (closed by us), don't retry.
      if (taskIdRef.current === taskId && status !== 'COMPLETED' && status !== 'FAILED') {
        // Retry in 3s
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log("Reconnecting...");
          connect(taskId);
        }, 3000);
      }
    };

  }, [status, disconnect]);

  return {
    status,
    progress,
    message,
    result,
    error,
    isConnected,
    connect,
    disconnect
  };
};
