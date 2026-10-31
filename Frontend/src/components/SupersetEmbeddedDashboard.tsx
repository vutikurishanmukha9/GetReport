import React, { useEffect, useRef, useState } from "react";
import { embedDashboard, type EmbeddedDashboard } from "@superset-ui/embedded-sdk";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ExternalLink,
  RefreshCw,
  Server,
  Database,
  Terminal,
  Check,
  Copy,
  AlertTriangle,
  Loader2,
  CheckCircle2,
} from "lucide-react";

interface SupersetEmbeddedDashboardProps {
  taskId?: string;
  datasetName?: string;
  dashboardId?: string;
}

export const SupersetEmbeddedDashboard: React.FC<SupersetEmbeddedDashboardProps> = ({
  taskId,
  datasetName,
  dashboardId = "1",
}) => {
  const { toast } = useToast();
  const mountRef = useRef<HTMLDivElement | null>(null);
  const embeddedInstanceRef = useRef<EmbeddedDashboard | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [status, setStatus] = useState<"checking" | "online" | "offline" | "error">("checking");
  const [supersetUrl, setSupersetUrl] = useState<string>("http://localhost:8088");
  const [instructions, setInstructions] = useState<string | null>(null);
  const [syncing, setSyncing] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  // 1. Check companion health
  const checkHealthAndEmbed = async () => {
    setLoading(true);
    setStatus("checking");

    try {
      const health = await api.getAnalyticsStatus();
      setSupersetUrl(health.superset_url || "http://localhost:8088");

      if (health.status !== "online") {
        setStatus("offline");
        setInstructions(
          health.instructions || "docker compose -f docker/docker-compose.superset.yml up -d"
        );
        setLoading(false);
        return;
      }

      setStatus("online");
      embedTargetDashboard(health.superset_url);
    } catch (err: any) {
      console.warn("Could not reach analytics companion status endpoint:", err);
      setStatus("offline");
      setInstructions("docker compose -f docker/docker-compose.superset.yml up -d");
      setLoading(false);
    }
  };

  // 2. Mount embedded Superset dashboard iframe
  const embedTargetDashboard = async (domain: string) => {
    if (!mountRef.current) return;

    try {
      // Unmount previous instance if any
      if (embeddedInstanceRef.current) {
        try {
          embeddedInstanceRef.current.unmount();
        } catch {
          // ignore
        }
        embeddedInstanceRef.current = null;
      }

      // Clear container
      mountRef.current.innerHTML = "";

      const instance = await embedDashboard({
        id: dashboardId,
        supersetDomain: domain,
        mountPoint: mountRef.current,
        fetchGuestToken: async () => {
          const res = await api.getSupersetGuestToken(dashboardId, taskId);
          if (!res.guest_token) {
            throw new Error(res.message || "Failed to issue Superset guest token.");
          }
          return res.guest_token;
        },
        dashboardUiConfig: {
          hideTitle: false,
          hideTab: false,
          hideChartControls: false,
          filters: {
            visible: true,
            expanded: false,
          },
        },
      });

      embeddedInstanceRef.current = instance;
      setLoading(false);
    } catch (embedErr: any) {
      console.error("Superset embedding error:", embedErr);
      setStatus("error");
      setLoading(false);
      toast({
        title: "Embedding Failed",
        description: embedErr.message || "Could not mount embedded Superset dashboard.",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    checkHealthAndEmbed();

    return () => {
      if (embeddedInstanceRef.current) {
        try {
          embeddedInstanceRef.current.unmount();
        } catch {
          // ignore
        }
        embeddedInstanceRef.current = null;
      }
    };
  }, [dashboardId, taskId]);

  // 3. Sync dataset to Superset
  const handleSyncDataset = async () => {
    if (!taskId) {
      toast({
        title: "No Dataset",
        description: "Please select an audited dataset first.",
        variant: "destructive",
      });
      return;
    }

    setSyncing(true);
    try {
      const res = await api.syncDatasetToSuperset(taskId);
      toast({
        title: "Dataset Synced",
        description: res.message || "Cleaned dataset registered in Superset.",
      });
      // Refresh iframe
      checkHealthAndEmbed();
    } catch (e: any) {
      toast({
        title: "Sync Failed",
        description: e.message || "Failed to sync dataset to Superset.",
        variant: "destructive",
      });
    } finally {
      setSyncing(false);
    }
  };

  const copyCommand = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast({ title: "Copied Command", description: "Command copied to clipboard." });
  };

  return (
    <div className="w-full rounded-2xl border border-slate-800 bg-[#07080a] text-slate-100 overflow-hidden flex flex-col">
      {/* ─── Top Companion Toolbar ─── */}
      <div className="px-4 py-3 border-b border-slate-800 bg-[#0c0d12] flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <div className="h-9 w-9 rounded-xl bg-slate-800 border border-slate-700/80 flex items-center justify-center text-zinc-300 shrink-0">
            <Server className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold tracking-tight text-white">
                Apache Superset BI Companion
              </span>
              <Badge
                variant="outline"
                className={`text-[10px] px-2 py-0.5 rounded-full font-mono uppercase tracking-wider ${
                  status === "online"
                    ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                    : status === "checking"
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                    : "bg-rose-500/10 text-rose-400 border-rose-500/30"
                }`}
              >
                {status === "checking" ? "Verifying..." : status}
              </Badge>
            </div>
            <p className="text-xs text-slate-400 mt-0.5 font-mono">
              Host: <span className="text-slate-300">{supersetUrl}</span> • Dashboard #{dashboardId}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {taskId && status === "online" && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleSyncDataset}
              disabled={syncing}
              className="h-8 rounded-lg border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 text-xs gap-1.5"
            >
              <Database className={`h-3.5 w-3.5 ${syncing ? "animate-spin" : ""}`} />
              <span>{syncing ? "Syncing..." : "Sync Dataset"}</span>
            </Button>
          )}

          <Button
            size="sm"
            variant="outline"
            onClick={checkHealthAndEmbed}
            disabled={loading}
            className="h-8 rounded-lg border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 gap-1.5"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </Button>

          {status === "online" && (
            <a
              href={supersetUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-200 transition"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              <span>Open Superset</span>
            </a>
          )}
        </div>
      </div>

      {/* ─── Main Content Canvas ─── */}
      <div className="min-h-[580px] sm:min-h-[660px] flex-1 flex flex-col relative bg-[#07080a]">
        {/* Loading Overlay */}
        {loading && (
          <div className="absolute inset-0 bg-[#07080a]/80 backdrop-blur-xs flex flex-col items-center justify-center z-20 gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-zinc-300" />
            <div className="text-center">
              <span className="text-sm font-semibold text-white block">
                Connecting to Superset Companion...
              </span>
              <span className="text-xs text-slate-400 font-mono mt-0.5 block">
                Negotiating scoped Guest Token for authenticated embedding
              </span>
            </div>
          </div>
        )}

        {/* Offline / Setup Instructions Card */}
        {status === "offline" && !loading && (
          <div className="flex-1 flex items-center justify-center p-6">
            <Card className="max-w-xl w-full border border-slate-800 bg-[#0c0d12] p-6 space-y-5 rounded-2xl shadow-xl">
              <div className="flex items-start gap-3.5">
                <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 shrink-0">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-base font-semibold text-white">
                    Superset Companion Container Offline
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed">
                    The external Apache Superset BI service is not currently running. You can explore your datasets using the{" "}
                    <span className="text-slate-200 font-medium">Native In-Memory Explorer</span>, or launch the containerized Superset companion stack via Docker:
                  </p>
                </div>
              </div>

              {/* Startup Command */}
              <div className="space-y-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold block">
                  Terminal Startup Command
                </span>
                <div className="flex items-center justify-between p-3 rounded-xl bg-[#07080a] border border-slate-800 font-mono text-xs text-slate-200 overflow-x-auto">
                  <div className="flex items-center gap-2">
                    <Terminal className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <code className="text-slate-200 whitespace-nowrap">
                      docker compose -f docker/docker-compose.superset.yml up -d
                    </code>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() =>
                      copyCommand("docker compose -f docker/docker-compose.superset.yml up -d")
                    }
                    className="h-7 px-2 text-slate-400 hover:text-white"
                  >
                    {copied ? (
                      <Check className="h-3.5 w-3.5 text-emerald-400" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Retry Action */}
              <div className="pt-2 flex items-center justify-end gap-3 border-t border-slate-800/80">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={checkHealthAndEmbed}
                  className="rounded-xl border-slate-700 bg-slate-800 text-xs font-medium text-slate-200 hover:text-white gap-2"
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  <span>Verify Connection Again</span>
                </Button>
              </div>
            </Card>
          </div>
        )}

        {/* Embedded Iframe Container */}
        <div
          ref={mountRef}
          className={`w-full flex-1 min-h-[600px] overflow-hidden ${
            status === "online" ? "block" : "hidden"
          }`}
          style={{ height: "100%" }}
        />
      </div>
    </div>
  );
};

