import React, { useEffect, useRef, useState } from "react";
import { embedDashboard } from "@superset-ui/embedded-sdk";
import {
  BarChart3,
  Maximize2,
  Minimize2,
  RefreshCw,
  Terminal,
  Check,
  Copy,
  AlertCircle,
  Server,
  Database,
  ArrowUpRight,
  CheckCircle2
} from "lucide-react";

interface CompanionStatus {
  enabled: boolean;
  status: "online" | "offline" | "unhealthy" | "disabled" | "unknown";
  superset_url: string;
  message: string;
  instructions?: string;
}

interface SupersetDashboardProps {
  dashboardId?: string;
  taskId?: string;
  datasetName?: string;
  onClose?: () => void;
}

export const SupersetDashboard: React.FC<SupersetDashboardProps> = ({
  dashboardId = "1",
  taskId,
  datasetName,
  onClose,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<CompanionStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [syncState, setSyncState] = useState<{
    syncing: boolean;
    synced: boolean;
    tableName?: string;
    rows?: number;
    message?: string;
  }>({ syncing: false, synced: false });

  const dockerCommand = "docker compose -f docker-compose.superset.yml up -d";

  const syncDatasetToSuperset = async () => {
    if (!taskId) return;
    setSyncState(prev => ({ ...prev, syncing: true }));

    try {
      const syncRes = await fetch(`/api/analytics/sync/${taskId}`, { method: "POST" });
      const syncData = await syncRes.json();
      if (syncRes.ok) {
        setSyncState({
          syncing: false,
          synced: true,
          tableName: syncData.table_name,
          rows: syncData.rows_synced,
          message: syncData.message,
        });
      } else {
        setSyncState(prev => ({
          ...prev,
          syncing: false,
          message: syncData.detail || "Dataset sync was queued.",
        }));
      }
    } catch (err: any) {
      console.warn("Dataset sync notice:", err);
      setSyncState(prev => ({
        ...prev,
        syncing: false,
        message: "Dataset is prepared for Superset import.",
      }));
    }
  };

  const fetchStatusAndEmbed = async () => {
    setLoading(true);
    setErrorMessage(null);

    try {
      // Check health
      const statusRes = await fetch("/api/analytics/status");
      const statusData: CompanionStatus = await statusRes.json();
      setStatus(statusData);

      if (statusData.status !== "online") {
        setLoading(false);
        return;
      }

      // If we have a taskId, trigger dataset sync into Superset
      if (taskId && !syncState.synced) {
        await syncDatasetToSuperset();
      }

      // Request guest token
      const tokenRes = await fetch("/api/analytics/guest-token", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dashboard_id: dashboardId, task_id: taskId }),
      });
      const tokenData = await tokenRes.json();

      if (tokenData.status === "online" && tokenData.guest_token && containerRef.current) {
        containerRef.current.innerHTML = "";

        await embedDashboard({
          id: tokenData.dashboard_id,
          supersetDomain: tokenData.superset_url,
          mountPoint: containerRef.current,
          fetchGuestToken: () => Promise.resolve(tokenData.guest_token),
          dashboardUiConfig: {
            hideTitle: true,
            hideTab: false,
            hideChartControls: false,
            filters: {
              expanded: false,
            },
            urlParams: {
              themeMode: "dark",
            },
          },
          iframeSandboxExtras: [
            "allow-top-navigation",
            "allow-popups-to-escape-sandbox",
          ],
        });
      } else {
        setErrorMessage(tokenData.message || "Failed to embed dashboard.");
      }
    } catch (err: any) {
      console.warn("Analytics status check failed:", err);
      setStatus({
        enabled: true,
        status: "offline",
        superset_url: "http://localhost:8088",
        message: "Companion service not detected.",
        instructions: "Run docker compose -f docker-compose.superset.yml up -d",
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatusAndEmbed();
  }, [dashboardId, taskId]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isOnline = status?.status === "online";

  return (
    <div
      className={`w-full rounded-xl border border-slate-800 bg-[#0d121f] overflow-hidden ${
        isFullscreen ? "fixed inset-0 z-50 rounded-none p-4 bg-[#090d16]" : ""
      }`}
    >
      {/* Top Header / Action Toolbar */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-[#111728] flex flex-wrap items-center justify-between gap-3">
        
        {/* Left: Component Title & Bound Dataset Info */}
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300">
            <BarChart3 className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white">
                Interactive BI Analytics (Apache Superset)
              </span>
              {taskId && (
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono">
                  Task: {taskId.slice(0, 8)}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400 mt-0.5">
              <span>{datasetName ? `Dataset: ${datasetName}` : "Audited dataset view"}</span>
              {syncState.tableName && (
                <>
                  <span>•</span>
                  <span className="text-emerald-400 font-mono">
                    Table: {syncState.tableName} ({syncState.rows?.toLocaleString() ?? 0} rows)
                  </span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Right: Controls & Indicators */}
        <div className="flex items-center gap-2.5">
          {/* Sync Dataset Button (When task is present) */}
          {taskId && (
            <button
              onClick={syncDatasetToSuperset}
              disabled={syncState.syncing}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-white/15 bg-white/[0.05] text-xs font-medium text-zinc-300 hover:bg-white/10 transition disabled:opacity-50"
              title="Sync current cleaned dataset into Superset analytical store"
            >
              <Database className={`h-3.5 w-3.5 ${syncState.syncing ? "animate-spin" : ""}`} />
              <span>{syncState.syncing ? "Syncing..." : syncState.synced ? "Dataset Synced" : "Sync to Superset"}</span>
            </button>
          )}

          {/* Connection Status Pill */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono border ${
              isOnline
                ? "bg-emerald-950/60 text-emerald-400 border-emerald-800"
                : "bg-amber-950/60 text-amber-400 border-amber-800"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                isOnline ? "bg-emerald-400" : "bg-amber-400"
              }`}
            />
            {isOnline ? "Companion Online" : "Companion Offline"}
          </div>

          {/* Sync Connection Button */}
          <button
            onClick={fetchStatusAndEmbed}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
            title="Refresh connection"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Reload</span>
          </button>

          {/* Fullscreen Button */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:bg-slate-700 transition"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? (
              <Minimize2 className="h-3.5 w-3.5" />
            ) : (
              <Maximize2 className="h-3.5 w-3.5" />
            )}
          </button>

          {onClose && (
            <button
              onClick={onClose}
              className="px-2.5 py-1 rounded border border-slate-700 text-xs text-slate-400 hover:text-white"
            >
              Close
            </button>
          )}
        </div>
      </div>

      {/* Main Container Area */}
      <div className="p-5">
        
        {/* State 1: Active Embedded Iframe */}
        {isOnline && (
          <div
            ref={containerRef}
            className="w-full min-h-[640px] rounded-lg overflow-hidden border border-slate-800 bg-[#070a12]"
            style={{ height: isFullscreen ? "calc(100vh - 100px)" : "680px" }}
          />
        )}

        {/* State 2: Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center min-h-[480px] gap-3 text-slate-400">
            <RefreshCw className="h-6 w-6 animate-spin text-slate-400" />
            <p className="text-xs font-mono text-slate-400">
              Connecting to Superset & binding dataset...
            </p>
          </div>
        )}

        {/* State 3: Standby / Companion Offline Panel */}
        {!isOnline && !loading && (
          <div className="space-y-6">
            
            {/* Direct Notice Box */}
            <div className="rounded-lg border border-amber-800/60 bg-amber-950/20 p-4 flex items-start gap-3">
              <Server className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-semibold text-white">
                  Apache Superset Companion Is Standing By
                </h4>
                <p className="text-xs text-slate-300 mt-0.5 leading-relaxed">
                  Your uploaded dataset <span className="font-mono text-white">({datasetName || "current file"})</span> has been cleaned and audited by GetReport. To explore it in interactive Superset dashboards, launch the Superset companion container.
                </p>
              </div>
            </div>

            {/* Launch Instructions */}
            <div className="rounded-lg border border-slate-800 bg-[#111728] p-5 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  Terminal Launch Command
                </span>
                <span className="text-xs font-mono text-slate-400">
                  Runs locally on port 8088
                </span>
              </div>

              {/* Code Box */}
              <div className="flex items-center justify-between gap-3 p-3 rounded-lg bg-[#070a12] border border-slate-800">
                <div className="flex items-center gap-2 overflow-x-auto">
                  <Terminal className="h-4 w-4 text-slate-500 shrink-0" />
                  <code className="text-xs font-mono text-slate-200 select-all">
                    {dockerCommand}
                  </code>
                </div>
                <button
                  onClick={() => copyToClipboard(dockerCommand)}
                  className="flex items-center gap-1.5 px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 transition shrink-0"
                >
                  {copied ? (
                    <>
                      <Check className="h-3 w-3 text-emerald-400" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs">
                <div className="p-3 rounded border border-slate-800/80 bg-[#0d121f]">
                  <span className="font-semibold text-slate-200 block mb-1">1. Launch Container</span>
                  <span className="text-slate-400">Run the command above to start Superset, Postgres, and Redis.</span>
                </div>
                <div className="p-3 rounded border border-slate-800/80 bg-[#0d121f]">
                  <span className="font-semibold text-slate-200 block mb-1">2. Auto-Wired Dataset</span>
                  <span className="text-slate-400">GetReport automatically syncs your cleaned table into Superset.</span>
                </div>
                <div className="p-3 rounded border border-slate-800/80 bg-[#0d121f]">
                  <span className="font-semibold text-slate-200 block mb-1">3. Live Dashboard</span>
                  <span className="text-slate-400">Click Reload above to view interactive charts without leaving GetReport.</span>
                </div>
              </div>
            </div>

            {/* Dataset Binding Status */}
            <div className="rounded-lg border border-slate-800 bg-[#111728] p-5 space-y-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-300 block">
                Dataset Pipeline Binding
              </span>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-2.5 rounded bg-[#070a12] border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Audit Task</span>
                  <span className="text-slate-200 mt-0.5 block truncate">{taskId ? taskId.slice(0, 16) : "Active Job"}</span>
                </div>
                <div className="p-2.5 rounded bg-[#070a12] border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Target Table</span>
                  <span className="text-slate-200 mt-0.5 block truncate">{syncState.tableName || (taskId ? `dataset_${taskId.slice(0, 12)}` : "dataset_auto")}</span>
                </div>
                <div className="p-2.5 rounded bg-[#070a12] border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Data Status</span>
                  <span className="text-emerald-400 mt-0.5 block">Audited & Cleaned</span>
                </div>
                <div className="p-2.5 rounded bg-[#070a12] border border-slate-800">
                  <span className="text-slate-400 block text-[11px]">Auth Method</span>
                  <span className="text-slate-200 mt-0.5 block truncate">Guest Token HMAC</span>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* Error notification if token request fails */}
        {errorMessage && (
          <div className="mt-4 flex items-center gap-2 p-3 rounded-lg bg-rose-950/40 border border-rose-800 text-rose-300 text-xs">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

      </div>
    </div>
  );
};
