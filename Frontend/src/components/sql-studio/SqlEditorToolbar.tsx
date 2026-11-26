import { useEffect, useState } from "react";
import { Check, Code2, Copy, Cpu, FileCode2, Play, Plus, Search, Square, Trash2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { ROW_LIMIT_OPTIONS, type QueryRowLimit, type SqlWorksheetTab } from "./types";

interface SqlEditorToolbarProps {
  worksheets: SqlWorksheetTab[];
  activeWorksheetId: string;
  onSelectWorksheet: (id: string) => void;
  onAddWorksheet: () => void;
  onCloseWorksheet: (id: string) => void;
  lineCount: number;
  charCount: number;
  sqlLimit: QueryRowLimit;
  onSqlLimitChange: (limit: QueryRowLimit) => void;
  sqlRunning: boolean;
  canRun: boolean;
  hasSelection: boolean;
  onFormat: () => void;
  onExplain: () => void;
  onCopy: () => void;
  onClear: () => void;
  onRun: () => void;
  onCancel: () => void;
  onOpenCommandPalette: () => void;
}

const DOCK_BUTTON_CLASS =
  "h-full px-2.5 rounded-[5px] text-xs font-medium gap-1.5 text-stone-700 hover:text-stone-950 hover:bg-stone-100/90 dark:text-zinc-300 dark:hover:text-white dark:hover:bg-white/[0.08] transition-colors [&_svg]:size-3.5";

const TOOLTIP_CONTENT_CLASS =
  "rounded-[6px] border border-stone-200 bg-white px-2.5 py-1 text-[11px] font-sans text-stone-800 shadow-lg dark:border-white/12 dark:bg-[#11131A] dark:text-zinc-100";

export function SqlEditorToolbar({
  worksheets,
  activeWorksheetId,
  onSelectWorksheet,
  onAddWorksheet,
  onCloseWorksheet,
  lineCount,
  charCount,
  sqlLimit,
  onSqlLimitChange,
  sqlRunning,
  canRun,
  hasSelection,
  onFormat,
  onExplain,
  onCopy,
  onClear,
  onRun,
  onCancel,
  onOpenCommandPalette,
}: SqlEditorToolbarProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return undefined;
    const timer = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const handleCopyClick = () => {
    onCopy();
    setCopied(true);
  };

  const handleLimitChange = (value: string) => {
    const matched = ROW_LIMIT_OPTIONS.find((opt) => String(opt) === value);
    if (matched !== undefined) {
      onSqlLimitChange(matched);
    }
  };

  return (
    <TooltipProvider delayDuration={250}>
      <div className="bg-[#F3EFE6] dark:bg-[#141721] border-b border-stone-200 dark:border-white/10">
        <div className="px-3.5 pt-2 pb-0 flex items-center justify-between gap-2 border-b border-stone-200/80 dark:border-white/[0.06] overflow-x-auto scrollbar-none">
          <div className="flex items-center gap-1 min-w-0">
            {worksheets.map((tab) => {
              const isActive = tab.id === activeWorksheetId;
              return (
                <div
                  key={tab.id}
                  className={cn(
                    "group inline-flex items-center gap-1.5 h-7 px-2.5 rounded-t-[6px] border border-b-0 text-[11px] font-mono transition-colors shrink-0",
                    isActive
                      ? "bg-white dark:bg-[#0B0D13] border-stone-300/90 dark:border-white/12 font-semibold text-stone-900 dark:text-zinc-100"
                      : "bg-transparent border-transparent text-stone-500 hover:text-stone-800 hover:bg-stone-200/50 dark:text-zinc-400 dark:hover:text-zinc-200 dark:hover:bg-white/[0.04]",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelectWorksheet(tab.id)}
                    className="inline-flex items-center gap-1.5 cursor-pointer bg-transparent border-0 p-0 text-inherit font-inherit"
                  >
                    <FileCode2
                      className={cn(
                        "h-3 w-3 shrink-0",
                        isActive ? "text-[#800000] dark:text-emerald-400" : "text-stone-400 dark:text-zinc-500",
                      )}
                    />
                    <span>{tab.title}</span>
                  </button>
                  {worksheets.length > 1 && (
                    <button
                      type="button"
                      aria-label={`Close ${tab.title}`}
                      onClick={() => onCloseWorksheet(tab.id)}
                      className="opacity-60 hover:opacity-100 text-stone-500 hover:text-red-700 dark:text-zinc-400 dark:hover:text-red-400 cursor-pointer p-0.5 rounded"
                    >
                      <X className="h-2.5 w-2.5" />
                    </button>
                  )}
                </div>
              );
            })}

            <button
              type="button"
              onClick={onAddWorksheet}
              title="New SQL worksheet tab"
              aria-label="New worksheet tab"
              className="inline-flex items-center justify-center h-6 w-6 rounded-[5px] text-stone-500 hover:text-stone-900 hover:bg-stone-200/70 dark:text-zinc-400 dark:hover:text-white dark:hover:bg-white/[0.08] cursor-pointer shrink-0 mb-0.5"
            >
              <Plus className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="flex items-center gap-2 shrink-0 pb-1">
            <span className="text-[11px] font-mono text-stone-500 dark:text-zinc-400 hidden sm:inline tabular-nums">
              {lineCount} {lineCount === 1 ? "line" : "lines"} · {charCount} chars
            </span>
            <button
              type="button"
              onClick={onOpenCommandPalette}
              title="Open SQL Command Palette (Ctrl+K)"
              className="inline-flex items-center gap-1 h-6 px-2 rounded-[5px] border border-stone-300/80 dark:border-white/10 bg-white/80 dark:bg-white/[0.04] hover:bg-white dark:hover:bg-white/[0.09] text-[10px] font-mono text-stone-600 dark:text-zinc-300 cursor-pointer"
            >
              <Search className="h-2.5 w-2.5" />
              <span className="hidden md:inline">Commands</span>
              <kbd className="text-[9px] opacity-75">Ctrl+K</kbd>
            </button>
          </div>
        </div>

        <div className="px-3.5 py-2 flex flex-wrap items-center justify-between gap-2">
          <div className="inline-flex items-center gap-1.5">
            <label
              htmlFor="duckdb-row-limit"
              className="text-[11px] font-mono font-medium text-stone-600 dark:text-zinc-400 select-none cursor-pointer"
            >
              Max rows:
            </label>
            <Select value={String(sqlLimit)} onValueChange={handleLimitChange}>
              <SelectTrigger
                id="duckdb-row-limit"
                aria-label="Max rows:"
                className="h-8 w-auto min-w-[4.75rem] rounded-[7px] border border-stone-300/90 bg-white px-2.5 gap-1.5 text-xs font-mono font-semibold text-stone-900 shadow-2xs hover:bg-stone-50 hover:border-stone-400/80 focus:ring-1 focus:ring-[#800000]/30 focus:ring-offset-0 dark:border-white/12 dark:bg-[#0D0F16] dark:text-zinc-100 dark:hover:bg-white/[0.07] dark:hover:border-white/20 dark:focus:ring-white/25 tabular-nums transition-colors cursor-pointer [&>svg]:h-3.5 [&>svg]:w-3.5 [&>svg]:text-stone-500 dark:[&>svg]:text-zinc-400"
              >
                <SelectValue>{sqlLimit}</SelectValue>
              </SelectTrigger>
              <SelectContent
                align="start"
                className="min-w-[8.75rem] rounded-lg border border-stone-200/90 bg-white p-1 shadow-xl dark:border-white/12 dark:bg-[#11131A]"
              >
                {ROW_LIMIT_OPTIONS.map((opt) => (
                  <SelectItem
                    key={opt}
                    value={String(opt)}
                    className="rounded-[5px] py-1.5 text-xs font-mono tabular-nums cursor-pointer"
                  >
                    {opt} rows
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <div
              role="group"
              aria-label="SQL editor actions"
              className="inline-flex items-center h-8 p-[3px] gap-0.5 rounded-[7px] border border-stone-300/90 bg-white shadow-2xs dark:border-white/12 dark:bg-[#0D0F16]"
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button type="button" size="xs" variant="ghost" onClick={onFormat} className={DOCK_BUTTON_CLASS}>
                    <Code2 className="text-stone-500 dark:text-zinc-400" />
                    <span>Format</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className={TOOLTIP_CONTENT_CLASS}>
                  Format SQL keywords
                </TooltipContent>
              </Tooltip>

              <Separator orientation="vertical" className="h-3.5 w-[1px] bg-stone-200 dark:bg-white/10" />

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    disabled={sqlRunning || !canRun}
                    onClick={onExplain}
                    className={DOCK_BUTTON_CLASS}
                  >
                    <Cpu className="text-stone-500 dark:text-zinc-400" />
                    <span>Explain</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className={TOOLTIP_CONTENT_CLASS}>
                  Inspect DuckDB physical query plan
                </TooltipContent>
              </Tooltip>

              <Separator orientation="vertical" className="h-3.5 w-[1px] bg-stone-200 dark:bg-white/10" />

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={handleCopyClick}
                    className={DOCK_BUTTON_CLASS}
                  >
                    {copied ? (
                      <Check className="text-emerald-600 dark:text-emerald-400" />
                    ) : (
                      <Copy className="text-stone-500 dark:text-zinc-400" />
                    )}
                    <span className={copied ? "text-emerald-700 dark:text-emerald-400" : undefined}>
                      {copied ? "Copied" : "Copy"}
                    </span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className={TOOLTIP_CONTENT_CLASS}>
                  Copy query to clipboard
                </TooltipContent>
              </Tooltip>

              <Separator orientation="vertical" className="h-3.5 w-[1px] bg-stone-200 dark:bg-white/10" />

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    size="xs"
                    variant="ghost"
                    onClick={onClear}
                    className="group h-full px-2.5 rounded-[5px] text-xs font-medium gap-1.5 text-stone-600 hover:text-red-700 hover:bg-red-50/90 dark:text-zinc-400 dark:hover:text-red-400 dark:hover:bg-red-500/10 transition-colors [&_svg]:size-3.5"
                  >
                    <Trash2 className="text-stone-500 group-hover:text-red-600 dark:text-zinc-400 dark:group-hover:text-red-400 transition-colors" />
                    <span>Clear</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className={TOOLTIP_CONTENT_CLASS}>
                  Clear editor and results
                </TooltipContent>
              </Tooltip>
            </div>

            {sqlRunning ? (
              <Button
                type="button"
                size="xs"
                onClick={onCancel}
                className="h-8 px-3.5 rounded-[7px] text-xs font-semibold gap-1.5 bg-red-700 hover:bg-red-800 text-white border border-red-800 shadow-xs [&_svg]:size-3"
              >
                <Square className="fill-current" />
                <span>Cancel</span>
              </Button>
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    size="xs"
                    disabled={!canRun}
                    onClick={onRun}
                    className="btn-maroon h-8 px-3.5 rounded-[7px] text-xs font-semibold gap-1.5 bg-[#800000] hover:bg-[#6b0000] text-white border border-[#800000] shadow-xs dark:bg-white dark:hover:bg-zinc-200 dark:text-zinc-950 dark:border-white/90 [&_svg]:size-3.5"
                  >
                    <Play className="fill-current" />
                    <span>{hasSelection ? "Run Selection" : "Run Query"}</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className={TOOLTIP_CONTENT_CLASS}>
                  Execute SQL {hasSelection ? "selection" : "query"}{" "}
                  <kbd className="ml-1 rounded bg-stone-100 px-1 py-0.5 font-mono text-[10px] dark:bg-white/10">
                    Ctrl+Enter
                  </kbd>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
}
