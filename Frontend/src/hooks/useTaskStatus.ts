import { useState, useEffect, useRef, useCallback } from 'react';
import { api, StatusResponse, JobTaskResult } from '@/services/api';

export type TaskStatus = 'PENDING' | 'PROCESSING' | 'WAITING_FOR_USER' | 'COMPLETED' | 'FAILED';

function isValidTaskStatus(status: string): status is TaskStatus {
  return ['PENDING', 'PROCESSING', 'WAITING_FOR_USER', 'COMPLETED', 'FAILED'].includes(status);
}

interface UseTaskStatusResult {
  status: TaskStatus | 'CONNECTING' | 'DISCONNECTED';
  progress: number;
  message: string;
  result: JobTaskResult | null;
  error: string | null;
  isConnected: boolean;
  connect: (taskId: string) => void;
  disconnect: () => void;
}

export const useTaskStatus = (activeTaskId?: string): UseTaskStatusResult => {
  const [status, setStatus] = useState<UseTaskStatusResult['status']>('CONNECTING');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<JobTaskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const watchdogTimerRef = useRef<NodeJS.Timeout | null>(null);
  const retryCountRef = useRef<number>(0);
  const taskIdRef = useRef<string | null>(null);
  const statusRef = useRef<UseTaskStatusResult['status']>('CONNECTING');

  const setTaskStatus = useCallback((newStatus: UseTaskStatusResult['status']) => {
    statusRef.current = newStatus;
    setStatus(newStatus);
  }, []);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const clearWatchdog = useCallback(() => {
    if (watchdogTimerRef.current) {
      clearTimeout(watchdogTimerRef.current);
      watchdogTimerRef.current = null;
    }
  }, []);

  // HTTP Polling fallback function (runs as resilient backup)
  const pollStatus = useCallback(async (taskId: string) => {
    try {
      const data: StatusResponse = await api.getTaskStatus(taskId);
      if (!data) return;

      if (data.status) {
        const upper = data.status.toUpperCase();
        if (isValidTaskStatus(upper)) {
          setTaskStatus(upper);
        }
      }
      if (data.progress !== undefined) setProgress((prev) => Math.max(prev, data.progress));
      if (data.message) setMessage(data.message);
      if (data.result) setResult(data.result);
      if (data.error) setError(data.error);

      if (['COMPLETED', 'FAILED'].includes(data.status?.toUpperCase())) {
        stopPolling();
        clearWatchdog();
        if (wsRef.current) {
          try { wsRef.current.close(); } catch { /* ignore close error */ }
          wsRef.current = null;
        }
      }
    } catch (err) {
      console.warn("HTTP Status poll warning:", err);
    }
  }, [setTaskStatus, stopPolling, clearWatchdog]);

  const startPolling = useCallback((taskId: string) => {
    stopPolling();
    pollStatus(taskId);
    pollIntervalRef.current = setInterval(() => {
      pollStatus(taskId);
    }, 1500);
  }, [pollStatus, stopPolling]);

  // Reset watchdog on frame arrival (if no frame for 25s, reconnect WS)
  const resetWatchdog = useCallback((taskId: string) => {
    clearWatchdog();
    watchdogTimerRef.current = setTimeout(() => {
      if (taskIdRef.current === taskId && !['COMPLETED', 'FAILED'].includes(statusRef.current)) {
        console.warn("WebSocket watchdog timeout (no frames for 25s). Reconnecting...");
        if (wsRef.current) {
          try { wsRef.current.close(); } catch { /* ignore close error */ }
        }
      }
    }, 25000);
  }, [clearWatchdog]);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.close(); } catch { /* ignore close error */ }
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    clearWatchdog();
    stopPolling();
    setIsConnected(false);
    retryCountRef.current = 0;
    setTaskStatus('DISCONNECTED');
  }, [setTaskStatus, stopPolling, clearWatchdog]);

  const connect = useCallback((taskId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && taskIdRef.current === taskId) {
      return;
    }

    disconnect();
    taskIdRef.current = taskId;
    setTaskStatus('CONNECTING');

    // Parallel HTTP Polling Backup
    startPolling(taskId);

    try {
      const url = api.getWebSocketUrl(taskId);
      // eslint-disable-next-line react-doctor/effect-needs-cleanup
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        retryCountRef.current = 0; // Reset backoff on success
        setTaskStatus('PROCESSING');
        resetWatchdog(taskId);
      };

      ws.onmessage = (event) => {
        resetWatchdog(taskId);
        try {
          const data = JSON.parse(event.data);

          // Handle server ping frame -> send pong
          if (data.type === 'ping') {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
            }
            return;
          }

          if (data.status) {
            const upper = data.status.toUpperCase();
            if (isValidTaskStatus(upper)) {
              setTaskStatus(upper);
            }
          }
          if (data.progress !== undefined) setProgress((prev) => Math.max(prev, data.progress));
          if (data.message) setMessage(data.message);
          if (data.result) setResult(data.result);
          if (data.error) setError(data.error);

          if (['COMPLETED', 'FAILED'].includes(data.status?.toUpperCase())) {
            stopPolling();
            clearWatchdog();
            try { ws.close(); } catch { /* ignore close error */ }
          }
        } catch (e) {
          console.error("Failed to parse WebSocket message:", e);
        }
      };

      ws.onerror = () => {
        console.warn("WebSocket status channel unavailable; HTTP polling active.");
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;
        clearWatchdog();

        if (taskIdRef.current === taskId && statusRef.current !== 'COMPLETED' && statusRef.current !== 'FAILED') {
          // Exponential backoff with random jitter (1s, 2s, 4s, 8s, 16s, 30s max)
          const attempt = retryCountRef.current;
          retryCountRef.current += 1;
          const delay = Math.min(30000, 1000 * Math.pow(1.5, attempt)) + Math.random() * 500;

          reconnectTimeoutRef.current = setTimeout(() => {
            if (taskIdRef.current === taskId && statusRef.current !== 'COMPLETED' && statusRef.current !== 'FAILED') {
              connect(taskId);
            }
          }, delay);
        }
      };
    } catch (e) {
      console.warn("WebSocket init error; active HTTP polling handling status:", e);
    }
  }, [disconnect, setTaskStatus, startPolling, stopPolling, resetWatchdog, clearWatchdog]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  // Auto-connect if activeTaskId is provided
  useEffect(() => {
    if (activeTaskId) {
      connect(activeTaskId);
    } else {
      disconnect();
    }
    return () => {
      disconnect();
    };
  }, [activeTaskId, connect, disconnect]);

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


