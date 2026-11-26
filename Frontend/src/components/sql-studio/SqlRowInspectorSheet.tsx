import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { DataRow, PrimitiveValue } from "@/types/api";
import { isFiniteNumber } from "./utils";

interface SqlRowInspectorSheetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  rowIndex: number | null;
  rowData: DataRow | null;
  columns: string[];
  columnTypes?: string[];
  onCopyText: (title: string, description: string, text: string) => void;
}

function tryFormatJsonString(val: PrimitiveValue): string | null {
  if (val === null || val === undefined || val.constructor !== String) return null;
  const trimmed = val.trim();
  if (!(trimmed.startsWith("{") || trimmed.startsWith("["))) return null;
  try {
    return JSON.stringify(JSON.parse(trimmed), null, 2);
  } catch {
    return null;
  }
}

export function SqlRowInspectorSheet({
  open,
  onOpenChange,
  rowIndex,
  rowData,
  columns,
  columnTypes,
  onCopyText,
}: SqlRowInspectorSheetProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="w-full sm:max-w-lg bg-[#FAF8F5] dark:bg-[#0D0F16] border-stone-300 dark:border-white/15 overflow-y-auto"
      >
        <SheetHeader className="pb-4 border-b border-stone-200 dark:border-white/10">
          <div className="flex items-center justify-between gap-2 pr-6">
            <div>
              <SheetTitle className="font-display text-base font-bold text-stone-900 dark:text-white">
                Row {rowIndex ?? ""} Inspector
              </SheetTitle>
              <SheetDescription className="text-xs font-mono text-stone-500 dark:text-zinc-400">
                {columns.length} column{columns.length === 1 ? "" : "s"}
              </SheetDescription>
            </div>
            {rowData && (
              <Button
                type="button"
                size="xs"
                variant="outline"
                onClick={() =>
                  onCopyText("Row Copied", `Row #${rowIndex ?? ""} copied as JSON.`, JSON.stringify(rowData, null, 2))
                }
                className="h-7 px-2.5 text-xs font-mono gap-1.5 rounded-[6px]"
              >
                <Copy className="h-3 w-3" />
                Copy Row JSON
              </Button>
            )}
          </div>
        </SheetHeader>

        {rowData && (
          <div className="py-4 space-y-3">
            {columns.map((col, idx) => {
              const rawVal = rowData[col];
              const colType = columnTypes?.[idx];
              const prettyJson = tryFormatJsonString(rawVal);

              return (
                <div
                  key={col}
                  className="rounded-lg border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#131620] p-3 space-y-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-mono text-xs font-bold text-stone-900 dark:text-zinc-100 truncate">
                        {col}
                      </span>
                      {colType && (
                        <span className="px-1.5 py-0.5 rounded bg-stone-100 dark:bg-white/[0.06] text-[9px] font-mono text-stone-500 dark:text-zinc-400">
                          {colType}
                        </span>
                      )}
                    </div>
                    {rawVal !== null && rawVal !== undefined && (
                      <button
                        type="button"
                        onClick={() => onCopyText("Field Copied", `${col} copied.`, String(rawVal))}
                        className="text-stone-400 hover:text-stone-800 dark:text-zinc-500 dark:hover:text-zinc-200 p-1 rounded cursor-pointer"
                        aria-label={`Copy value of ${col}`}
                      >
                        <Copy className="h-3 w-3" />
                      </button>
                    )}
                  </div>

                  {rawVal === null || rawVal === undefined ? (
                    <span className="inline-block px-2 py-0.5 rounded bg-stone-100 dark:bg-white/[0.06] text-[11px] font-sans italic text-stone-400 dark:text-zinc-500">
                      null
                    </span>
                  ) : prettyJson ? (
                    <pre className="p-2 rounded bg-[#FAF8F5] dark:bg-black/30 text-[11px] font-mono text-stone-800 dark:text-emerald-300 overflow-x-auto max-h-56 leading-relaxed">
                      {prettyJson}
                    </pre>
                  ) : (
                    <div className="font-mono text-xs text-stone-800 dark:text-zinc-200 break-words whitespace-pre-wrap">
                      {isFiniteNumber(rawVal)
                        ? rawVal.toLocaleString(undefined, { maximumFractionDigits: 6 })
                        : String(rawVal)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
