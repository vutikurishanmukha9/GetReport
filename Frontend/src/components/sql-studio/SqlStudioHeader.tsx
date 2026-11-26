import React from "react";
import {
  Maximize2,
  Minimize2,
  PanelLeftClose,
  PanelLeftOpen,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { RecentDatasetItem } from "@/types/api";

interface SqlStudioHeaderProps {
  selectedTaskId: string;
  activeTableName: string;
  resolvedDatasetName: string;
  resolvedRowCount: number;
  columnCount: number;
  allowDatasetSwitching: boolean;
  availableDatasets: RecentDatasetItem[];
  loadingDatasets: boolean;
  onSelectDataset: (taskId: string) => void;
  showInspector: boolean;
  onToggleInspector: () => void;
  isFullscreen: boolean;
  onToggleFullscreen: () => void;
}

export const SqlStudioHeader: React.FC<SqlStudioHeaderProps> = ({
  selectedTaskId,
  activeTableName,
  resolvedDatasetName,
  resolvedRowCount,
  columnCount,
  allowDatasetSwitching,
  availableDatasets,
  loadingDatasets,
  onSelectDataset,
  showInspector,
  onToggleInspector,
  isFullscreen,
  onToggleFullscreen,
}) => {
  return (
    <div className="px-4 sm:px-6 py-4 border-b border-stone-300/80 dark:border-white/10 bg-[#F3EFE6]/90 dark:bg-[#101218] flex flex-col xl:flex-row xl:items-center justify-between gap-3">
      <div className="flex items-start sm:items-center gap-3">
        <div className="h-10 w-10 rounded-xl bg-[#800000]/10 border border-[#800000]/25 dark:bg-emerald-500/10 dark:border-emerald-500/25 flex items-center justify-center shrink-0">
          <Terminal className="h-5 w-5 text-[#800000] dark:text-emerald-400" />
        </div>
        <div className="space-y-0.5">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-base sm:text-lg font-display font-bold tracking-tight text-stone-900 dark:text-white">
              DuckDB SQL Studio
            </h3>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-emerald-500/15 text-emerald-800 border border-emerald-500/25 dark:bg-emerald-500/10 dark:text-emerald-400 dark:border-emerald-500/25">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400" />
              In-Process Arrow
            </span>
            <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-medium bg-stone-200/80 text-stone-700 border border-stone-300 dark:bg-white/[0.06] dark:text-zinc-300 dark:border-white/10">
              view: <strong>{activeTableName || "dataset"}</strong>
            </span>
          </div>
          <p className="text-[11px] font-sans text-stone-600 dark:text-zinc-400 tabular-nums">
            {selectedTaskId
              ? `${resolvedDatasetName} · ${resolvedRowCount.toLocaleString()} rows × ${columnCount} columns · Read-only sandbox`
              : "Select an audited dataset to run read-only analytical SQL queries"}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {allowDatasetSwitching && (
          <div className="flex items-center gap-1.5">
            <Select
              value={selectedTaskId || undefined}
              disabled={loadingDatasets || availableDatasets.length === 0}
              onValueChange={onSelectDataset}
            >
              <SelectTrigger
                id="duckdb-dataset-select"
                aria-label="Active dataset"
                className="h-8 max-w-[240px] rounded-[7px] border-stone-300 bg-white px-2.5 text-xs font-mono text-stone-900 dark:border-white/12 dark:bg-[#0D0F16] dark:text-zinc-200"
              >
                <SelectValue
                  placeholder={
                    loadingDatasets
                      ? "Loading datasets…"
                      : availableDatasets.length === 0
                        ? "No datasets uploaded"
                        : "Select dataset"
                  }
                />
              </SelectTrigger>
              <SelectContent align="end">
                {availableDatasets.map((ds) => (
                  <SelectItem key={ds.task_id} value={ds.task_id} className="text-xs font-mono">
                    {ds.filename} ({ds.rows.toLocaleString()} rows)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onToggleInspector}
          className="h-8 rounded-[7px] border-stone-300 bg-white/90 hover:bg-white text-stone-800 dark:border-white/12 dark:bg-white/[0.05] dark:hover:bg-white/[0.09] dark:text-zinc-200 text-xs gap-1.5 cursor-pointer"
          title={showInspector ? "Hide Inspector panel" : "Show Inspector panel"}
        >
          {showInspector ? <PanelLeftClose className="h-3.5 w-3.5" /> : <PanelLeftOpen className="h-3.5 w-3.5" />}
          <span>{showInspector ? "Hide Inspector" : "Show Inspector"}</span>
        </Button>

        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={onToggleFullscreen}
          className="h-8 rounded-[7px] border-stone-300 bg-white/90 hover:bg-white text-stone-800 dark:border-white/12 dark:bg-white/[0.05] dark:hover:bg-white/[0.09] dark:text-zinc-200 text-xs gap-1.5 cursor-pointer"
          aria-label={isFullscreen ? "Exit fullscreen focus mode" : "Enter fullscreen focus mode"}
          title={isFullscreen ? "Exit Focus Mode (Esc)" : "Expand SQL Studio"}
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
          <span className="hidden sm:inline">{isFullscreen ? "Exit Focus" : "Expand"}</span>
        </Button>
      </div>
    </div>
  );
};
