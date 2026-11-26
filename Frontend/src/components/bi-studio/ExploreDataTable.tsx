import { ArrowUpDown, ChevronLeft, ChevronRight, Copy, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { COLOR_PALETTES } from "./constants";

interface ExploreDataTableProps {
  dimension: string;
  secondaryDimension: string;
  metricLabel?: string;
  colorPalette: string;
  tableSearch: string;
  onSearchChange: (val: string) => void;
  tableSortCol: string;
  onSortClick: (col: string) => void;
  tablePage: number;
  onPageChange: (updater: number | ((p: number) => number)) => void;
  tablePageSize: number;
  onPageSizeChange: (size: number) => void;
  totalPages: number;
  filteredRecordsCount: number;
  paginatedRecords: any[];
  totalMetricSum: number;
  totalRowCount: number;
  onFilterValueClick: (val: any) => void;
  onCopyRowJson: (row: any) => void;
}

export function ExploreDataTable({
  dimension,
  secondaryDimension,
  metricLabel,
  colorPalette,
  tableSearch,
  onSearchChange,
  tableSortCol,
  onSortClick,
  tablePage,
  onPageChange,
  tablePageSize,
  onPageSizeChange,
  totalPages,
  filteredRecordsCount,
  paginatedRecords,
  totalMetricSum,
  totalRowCount,
  onFilterValueClick,
  onCopyRowJson,
}: ExploreDataTableProps) {
  const paletteColors = COLOR_PALETTES.find((p) => p.id === colorPalette)?.preview || ["#1FA8C9"];

  return (
    <div className="flex-1 flex flex-col space-y-3 min-h-[420px]">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 pb-2">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-stone-400 dark:text-zinc-500" />
          <input
            type="text"
            placeholder={`Search ${dimension} or values...`}
            value={tableSearch}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-stone-50 dark:bg-[#14151c] border border-stone-200 dark:border-white/10 rounded-xl text-stone-900 dark:text-white placeholder:text-stone-400 dark:placeholder:text-zinc-500 focus:outline-none focus:border-stone-400 dark:focus:border-white/30 font-sans shadow-2xs"
          />
        </div>

        <div className="flex items-center gap-2 self-end sm:self-auto">
          <span className="text-[11px] text-stone-500 dark:text-zinc-400 font-sans">Rows per page:</span>
          <select
            aria-label="Rows per page"
            value={tablePageSize}
            onChange={(e) => onPageSizeChange(Number(e.target.value))}
            className="bg-stone-50 dark:bg-[#14151c] border border-stone-200 dark:border-white/10 rounded-lg px-2 py-1 text-xs text-stone-800 dark:text-zinc-200 font-mono shadow-2xs"
          >
            <option value={10}>10</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </div>
      </div>

      <div className="flex-1 overflow-x-auto rounded-xl border border-stone-200/90 dark:border-white/10 bg-white dark:bg-[#090a0f] scrollbar-thin">
        <table className="w-full text-left text-xs font-sans">
          <thead className="bg-stone-50/90 text-stone-700 dark:bg-[#13141b] dark:text-zinc-300 uppercase text-[10.5px] font-semibold tracking-wider border-b border-stone-200 dark:border-white/10">
            <tr>
              <th className="py-2.5 px-3 w-12 text-center text-stone-400 dark:text-zinc-500">#</th>
              <th className="py-2.5 px-3">
                <button
                  type="button"
                  onClick={() => onSortClick(dimension)}
                  className="w-full flex items-center gap-1.5 text-left cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                >
                  <span>{dimension}</span>
                  <ArrowUpDown className={`h-3 w-3 ${tableSortCol === dimension ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                </button>
              </th>
              {secondaryDimension && (
                <th className="py-2.5 px-3">
                  <button
                    type="button"
                    onClick={() => onSortClick(secondaryDimension)}
                    className="w-full flex items-center gap-1.5 text-left cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                  >
                    <span>{secondaryDimension}</span>
                    <ArrowUpDown className={`h-3 w-3 ${tableSortCol === secondaryDimension ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                  </button>
                </th>
              )}
              <th className="py-2.5 px-3 min-w-[200px]">
                <button
                  type="button"
                  onClick={() => onSortClick("value")}
                  className="w-full flex items-center gap-1.5 text-left cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                >
                  <span>{metricLabel || "Metric Value"} &amp; Share</span>
                  <ArrowUpDown className={`h-3 w-3 ${tableSortCol === "value" ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                </button>
              </th>
              <th className="py-2.5 px-3 text-right">
                <button
                  type="button"
                  onClick={() => onSortClick("count")}
                  className="w-full flex items-center justify-end gap-1.5 cursor-pointer hover:text-stone-900 dark:hover:text-white transition select-none"
                >
                  <span>Records</span>
                  <ArrowUpDown className={`h-3 w-3 ${tableSortCol === "count" ? "text-emerald-600 dark:text-emerald-400" : "opacity-40"}`} />
                </button>
              </th>
              <th className="py-2.5 px-3 text-right w-16">Action</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-stone-100 dark:divide-white/[0.05] text-stone-800 dark:text-zinc-200">
            {paginatedRecords.length > 0 ? (
              paginatedRecords.map((row: any, idx: number) => {
                const globalIdx = (tablePage - 1) * tablePageSize + idx + 1;
                const rowVal =
                  row.value !== null && row.value !== undefined && row.value.constructor === Number && Number.isFinite(row.value)
                    ? Number(row.value)
                    : 0;
                const pctOfTotal = totalMetricSum > 0 ? (Math.abs(rowVal) / totalMetricSum) * 100 : 0;
                const stableRowKey = `${String(row[dimension] ?? "")}::${secondaryDimension ? String(row[secondaryDimension] ?? "") : ""}::${String(row.value ?? "")}::${String(row.count ?? "")}`;

                return (
                  <tr key={stableRowKey} className="hover:bg-stone-50/80 dark:hover:bg-white/[0.04] transition group">
                    <td className="py-2.5 px-3 text-center text-stone-400 dark:text-zinc-500 font-mono text-[11px]">{globalIdx}</td>
                    <td className="py-2.5 px-3 font-medium text-stone-900 dark:text-white">
                      <button
                        type="button"
                        onClick={() => onFilterValueClick(row[dimension])}
                        className="text-left hover:text-emerald-600 dark:hover:text-emerald-400 transition cursor-pointer flex items-center gap-1.5"
                        title={`Filter by ${dimension} = "${row[dimension]}"`}
                      >
                        <span>{row[dimension] ?? "—"}</span>
                      </button>
                    </td>
                    {secondaryDimension && (
                      <td className="py-2.5 px-3 text-stone-700 dark:text-zinc-300 font-mono text-[11px]">
                        {row[secondaryDimension] ?? "—"}
                      </td>
                    )}
                    <td className="py-2.5 px-3">
                      <div className="space-y-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="font-bold text-stone-900 dark:text-white font-mono text-[11.5px] tabular-nums">
                            {row.value !== null && row.value !== undefined && row.value.constructor === Number && Number.isFinite(row.value)
                              ? Number(row.value).toLocaleString(undefined, { maximumFractionDigits: 2 })
                              : row.value ?? "—"}
                          </span>
                          <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-mono font-medium">
                            {totalMetricSum > 0 ? `${pctOfTotal.toFixed(1)}%` : ""}
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-stone-200/70 dark:bg-white/10 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${Math.min(100, Math.max(3, pctOfTotal))}%`,
                              backgroundColor: paletteColors[0] || "#10b981",
                            }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono text-[11px] text-stone-600 dark:text-zinc-400 tabular-nums">
                      {row.count?.toLocaleString() ?? "—"}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        type="button"
                        onClick={() => onCopyRowJson(row)}
                        className="p-1 rounded-md text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 hover:bg-stone-100 dark:hover:bg-white/10 transition cursor-pointer"
                        title="Copy row data"
                      >
                        <Copy className="h-3 w-3" />
                      </button>
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={6} className="py-8 text-center text-stone-500 dark:text-zinc-400 font-sans">
                  No records matching &ldquo;{tableSearch}&rdquo;
                </td>
              </tr>
            )}
          </tbody>

          {filteredRecordsCount > 0 && (
            <tfoot className="bg-stone-50/90 dark:bg-[#13141b] border-t border-stone-200 dark:border-white/10 font-sans font-semibold text-[11px] text-stone-900 dark:text-white">
              <tr>
                <td className="py-2.5 px-3 text-center text-stone-500 dark:text-zinc-400">∑</td>
                <td className="py-2.5 px-3">Total ({filteredRecordsCount} categories)</td>
                {secondaryDimension && <td className="py-2.5 px-3">—</td>}
                <td className="py-2.5 px-3 font-mono tabular-nums">
                  {totalMetricSum.toLocaleString(undefined, { maximumFractionDigits: 2 })} (100%)
                </td>
                <td className="py-2.5 px-3 text-right font-mono tabular-nums">{totalRowCount.toLocaleString()}</td>
                <td className="py-2.5 px-3 text-right">—</td>
              </tr>
            </tfoot>
          )}
        </table>
      </div>

      <div className="flex items-center justify-between text-xs text-stone-600 dark:text-zinc-400 pt-1 font-sans">
        <span className="text-[11px]">
          Showing <strong className="text-stone-900 dark:text-white font-mono">{(tablePage - 1) * tablePageSize + 1}</strong> to{" "}
          <strong className="text-stone-900 dark:text-white font-mono">{Math.min(tablePage * tablePageSize, filteredRecordsCount)}</strong> of{" "}
          <strong className="text-stone-900 dark:text-white font-mono">{filteredRecordsCount}</strong> groups
        </span>

        <div className="flex items-center gap-1.5">
          <Button
            size="sm"
            variant="outline"
            disabled={tablePage <= 1}
            onClick={() => onPageChange((p) => Math.max(1, p - 1))}
            className="h-7 px-2 rounded-lg text-xs gap-1 border-stone-200 bg-white hover:bg-stone-100 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
          >
            <ChevronLeft className="h-3.5 w-3.5" />
            <span>Prev</span>
          </Button>
          <span className="text-[11px] px-2 font-mono">
            {tablePage} / {totalPages}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={tablePage >= totalPages}
            onClick={() => onPageChange((p) => Math.min(totalPages, p + 1))}
            className="h-7 px-2 rounded-lg text-xs gap-1 border-stone-200 bg-white hover:bg-stone-100 dark:border-white/10 dark:bg-[#14151c] dark:hover:bg-white/[0.08] cursor-pointer"
          >
            <span>Next</span>
            <ChevronRight className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
