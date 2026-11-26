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

interface UseTaskStatusOptions {
  onStatusUpdate?: (data: StatusResponse) => void;
}

export const useTaskStatus = (
  activeTaskId?: string,
  options?: UseTaskStatusOptions,
): UseTaskStatusResult => {
  const [status, setStatus] = useState<UseTaskStatusResult['status']>('CONNECTING');
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [result, setResult] = useState<JobTaskResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const onStatusUpdateRef = useRef(options?.onStatusUpdate);
  useEffect(() => {
    onStatusUpdateRef.current = options?.onStatusUpdate;
  }, [options?.onStatusUpdate]);

  const eventSourceUnsubRef = useRef<(() => void) | null>(null);
  const wsUnsubRef = useRef<(() => void) | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const watchdogTimerRef = useRef<NodeJS.Timeout | null>(null);
  const retryCountRef = useRef<number>(0);
  const pollFailureCountRef = useRef<number>(0);
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

  // Update component state from any incoming StatusResponse packet
  const applyStatusUpdate = useCallback((data: StatusResponse) => {
    if (!data) return;

    const prevStatus = statusRef.current;
    const upper = data.status ? data.status.toUpperCase() : undefined;
    if (upper && isValidTaskStatus(upper)) {
      setTaskStatus(upper);
    }

    if (data.progress !== undefined) {
      // Allow reset if returning from WAITING_FOR_USER into a fresh processing stage or a new connection
      const resetWatermark =
        (prevStatus === 'WAITING_FOR_USER' && upper === 'PROCESSING') ||
        prevStatus === 'CONNECTING' ||
        prevStatus === 'DISCONNECTED';
      setProgress((prev) => (resetWatermark ? data.progress : Math.max(prev, data.progress)));
    }
    if (data.message !== undefined) setMessage(data.message);
    if (data.result !== undefined) setResult(data.result);
    if (data.error !== undefined) setError(data.error);

    if (upper && ['COMPLETED', 'FAILED'].includes(upper)) {
      stopPolling();
      clearWatchdog();
      if (eventSourceUnsubRef.current) {
        eventSourceUnsubRef.current();
        eventSourceUnsubRef.current = null;
      }
      if (wsUnsubRef.current) {
        wsUnsubRef.current();
        wsUnsubRef.current = null;
      }
    }

    onStatusUpdateRef.current?.(data);
  }, [setTaskStatus, stopPolling, clearWatchdog]);

  // HTTP Polling fallback function (runs as resilient backup with backoff)
  const pollStatus = useCallback(async (taskId: string) => {
    try {
      const data: StatusResponse = await api.getTaskStatus(taskId);
      pollFailureCountRef.current = 0;
      applyStatusUpdate(data);
    } catch (err: any) {
      pollFailureCountRef.current = (pollFailureCountRef.current || 0) + 1;
      console.warn("HTTP Status poll warning:", err);

      // Backoff if rate-limited or repeatedly failing to avoid request storms
      const isRateLimited = err?.message?.includes("429") || err?.message?.includes("Rate limit");
      const backoffDelay = isRateLimited ? 8000 : Math.min(10000, 2500 + pollFailureCountRef.current * 1500);

      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = setInterval(() => {
          pollStatus(taskId);
        }, backoffDelay);
      }
    }
  }, [applyStatusUpdate]);

  const startPolling = useCallback((taskId: string) => {
    stopPolling();
    pollFailureCountRef.current = 0;
    pollStatus(taskId);
    pollIntervalRef.current = setInterval(() => {
      pollStatus(taskId);
    }, 2500);
  }, [pollStatus, stopPolling]);

  // Reset watchdog on frame arrival (if no frame for 25s, reconnect)
  const resetWatchdog = useCallback((taskId: string) => {
    clearWatchdog();
    watchdogTimerRef.current = setTimeout(() => {
      if (taskIdRef.current === taskId && !['COMPLETED', 'FAILED'].includes(statusRef.current)) {
        console.warn("Real-time stream watchdog timeout (no frames for 25s). Reconnecting...");
        startPolling(taskId);
      }
    }, 25000);
  }, [clearWatchdog, startPolling]);

  const disconnect = useCallback(() => {
    if (eventSourceUnsubRef.current) {
      eventSourceUnsubRef.current();
      eventSourceUnsubRef.current = null;
    }
    if (wsUnsubRef.current) {
      wsUnsubRef.current();
      wsUnsubRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    clearWatchdog();
    stopPolling();
    taskIdRef.current = null;
    setIsConnected(false);
    retryCountRef.current = 0;
    pollFailureCountRef.current = 0;
    setProgress(0);
    setMessage('');
    setResult(null);
    setError(null);
    setTaskStatus('DISCONNECTED');
  }, [setTaskStatus, stopPolling, clearWatchdog]);

  const startWebSocketChannel = useCallback((taskId: string) => {
    if (wsUnsubRef.current) {
      wsUnsubRef.current();
      wsUnsubRef.current = null;
    }
    try {
      startPolling(taskId);
      const unsubWs = api.subscribeWebSocketStatus(taskId, {
        onOpen: () => {
          setIsConnected(true);
          retryCountRef.current = 0;
          resetWatchdog(taskId);
        },
        onMessage: (data) => {
          resetWatchdog(taskId);
          if ("status" in data) {
            applyStatusUpdate(data);
          }
        },
        onError: () => {
          console.warn("WebSocket status channel unavailable; HTTP polling active.");
        },
        onClose: () => {
          setIsConnected(false);
          wsUnsubRef.current = null;
          clearWatchdog();

          if (taskIdRef.current === taskId && statusRef.current !== 'COMPLETED' && statusRef.current !== 'FAILED') {
            const attempt = retryCountRef.current;
            retryCountRef.current += 1;
            const delay = Math.min(30000, 1000 * Math.pow(1.5, attempt)) + Math.random() * 500;

            reconnectTimeoutRef.current = setTimeout(() => {
              if (taskIdRef.current === taskId && statusRef.current !== 'COMPLETED' && statusRef.current !== 'FAILED') {
                startWebSocketChannel(taskId);
              }
            }, delay);
          }
        }
      });
      wsUnsubRef.current = unsubWs;
    } catch (e) {
      console.warn("WebSocket init error; active HTTP polling handling status:", e);
    }
  }, [startPolling, resetWatchdog, clearWatchdog, applyStatusUpdate]);

  const connect = useCallback((taskId: string) => {
    disconnect();
    taskIdRef.current = taskId;
    setTaskStatus('CONNECTING');

    // Immediate state fetch to give instant feedback
    pollStatus(taskId);

    // ── 1. Primary Strategy: Server-Sent Events (SSE) Stream ──────────
    if ("EventSource" in globalThis) {
      try {
        const unsub = api.subscribeTaskStatus(taskId, {
          onProgress: (data) => {
            setIsConnected(true);
            resetWatchdog(taskId);
            applyStatusUpdate(data);
          },
          onWaitingForUser: (data) => {
            setIsConnected(true);
            resetWatchdog(taskId);
            applyStatusUpdate(data);
          },
          onComplete: (data) => {
            setIsConnected(true);
            applyStatusUpdate(data);
            stopPolling();
          },
          onError: (err) => {
            console.warn("SSE stream interrupted or unavailable; falling back to WebSocket & polling.", err);
            // Teardown SSE and upgrade into WebSocket + HTTP polling resilience
            if (eventSourceUnsubRef.current) {
              eventSourceUnsubRef.current();
              eventSourceUnsubRef.current = null;
            }
            if (taskIdRef.current === taskId && !['COMPLETED', 'FAILED'].includes(statusRef.current)) {
              startWebSocketChannel(taskId);
            }
          }
        });

        eventSourceUnsubRef.current = unsub;
        return;
      } catch (sseErr) {
        console.warn("Failed to initialize SSE EventSource:", sseErr);
      }
    }

    // ── 2. Fallback Strategy: WebSocket + Polling ────────────────────
    startWebSocketChannel(taskId);
  }, [disconnect, setTaskStatus, stopPolling, resetWatchdog, applyStatusUpdate, pollStatus, startWebSocketChannel]);

  // Auto-connect and ensure full lifecycle cleanup of timers, SSE, and sockets
  useEffect(() => {
    if (activeTaskId) {
      connect(activeTaskId);
    } else {
      disconnect();
    }
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (watchdogTimerRef.current) {
        clearTimeout(watchdogTimerRef.current);
        watchdogTimerRef.current = null;
      }
      if (eventSourceUnsubRef.current) {
        eventSourceUnsubRef.current();
        eventSourceUnsubRef.current = null;
      }
      if (wsUnsubRef.current) {
        wsUnsubRef.current();
        wsUnsubRef.current = null;
      }
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
