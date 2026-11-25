import React, { useState, useCallback } from "react";
import { Copy, Check } from "lucide-react";
import { AnimatedNumber } from "@/components/motion/animated-number";

export interface ExploreTelemetryDeckProps {
  queryResult: {
    kpis: {
      metric_label: string;
      dimension_label: string;
      total_rows: number;
      sum: number | null;
      mean: number | null;
      max: number | null;
      min: number | null;
      unique_categories: number;
    };
    records: Array<Record<string, string | number | boolean | null | undefined>>;
    total_groups: number;
    secondary_metric_label?: string;
  };
  aggregate: string;
  metric: string;
  dimension: string;
}

type CompactScalarValue = string | number | boolean | null | undefined;

export const formatSmartCompact = (rawVal: CompactScalarValue): string => {
  if (rawVal === null || rawVal === undefined || rawVal === "") return "—";
  const num = Number(rawVal);
  if (isNaN(num)) return String(rawVal);
  const abs = Math.abs(num);
  if (abs === 0) return "0";
  if (!isFinite(abs)) return "∞";
  if (abs >= 1e15) return (num / 1e15).toFixed(2) + "Q";
  if (abs >= 1e12) return (num / 1e12).toFixed(2) + "T";
  if (abs >= 1e9) return (num / 1e9).toFixed(2) + "B";
  if (abs >= 1e6) return (num / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (num / 1e3).toFixed(1) + "K";
  if (abs < 0.001 && abs > 0) return num.toExponential(2);
  return Number.isInteger(num) ? num.toLocaleString() : num.toFixed(2);
};

export const formatDimensionValue = (val: CompactScalarValue): string => {
  if (val === null || val === undefined || val === "") return "—";
  const str = String(val).trim();
  // ISO date strings: e.g. "2019-05-01 00:00:00" -> "May 1, 2019"
  if (/^\d{4}-\d{2}-\d{2}/.test(str)) {
    const datePart = str.split(/[ T]/)[0];
    const parts = datePart.split("-").map(Number);
    if (parts.length === 3 && parts[0] && parts[1] >= 1 && parts[1] <= 12) {
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      return parts[2] ? `${monthNames[parts[1] - 1]} ${parts[2]}, ${parts[0]}` : `${monthNames[parts[1] - 1]} ${parts[0]}`;
    }
  }
  if (/^\d{4}-\d{2}$/.test(str)) {
    const [y, m] = str.split("-").map(Number);
    if (y && m >= 1 && m <= 12) {
      const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      return `${monthNames[m - 1]} ${y}`;
    }
  }
  return str;
};

const parseMetricMeta = (rawLabel: string, aggregate: string, metric: string) => {
  const label = rawLabel || (metric ? `${aggregate}(${metric})` : "Record Count");
  const match = label.match(/^([A-Z_]+)\((.+)\)$/);
  if (match) {
    const agg = match[1];
    const rawCol = match[2].replace(/_/g, " ").trim();
    const colWords = rawCol.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
    return {
      cleanName: colWords,
      operation: agg,
    };
  }
  const clean = label.replace(/_/g, " ").trim();
  const colWords = clean.split(" ").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  return {
    cleanName: colWords || "Record Count",
    operation: aggregate || "COUNT",
  };
};

export const ExploreTelemetryDeck: React.FC<ExploreTelemetryDeckProps> = ({
  queryResult,
  aggregate,
  metric,
  dimension,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = useCallback((id: string, text: string) => {
    if (!text || text === "—") return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => {
      setCopiedId(current => (current === id ? null : current));
    }, 1500);
  }, []);

  const kpis = queryResult?.kpis;
  if (!kpis) return null;

  // 1. Metric Calculation
  const isAvg = aggregate === "AVG";
  const isSum = aggregate === "SUM";
  const isMax = aggregate === "MAX";
  const isMin = aggregate === "MIN";

  const metricDisplayVal = isAvg
    ? (kpis.mean ?? kpis.sum)
    : isSum
      ? (kpis.sum ?? kpis.mean)
      : isMax
        ? (kpis.max ?? kpis.sum)
        : isMin
          ? (kpis.min ?? kpis.sum)
          : (kpis.sum ?? kpis.mean);

  const metricMeta = parseMetricMeta(kpis.metric_label, aggregate, metric);
  const metricStr = metricDisplayVal !== null && metricDisplayVal !== undefined
    ? formatSmartCompact(metricDisplayVal)
    : "—";

  // 2. Peak Performer Extraction
  const rawPeakDim = queryResult.records?.[0]?.[dimension];
  const formattedPeakDim = formatDimensionValue(rawPeakDim);
  const peakCount = queryResult.records?.[0]?.count;

  const peakRowPct = kpis.total_rows > 0 && peakCount !== undefined && peakCount !== null
    ? ((Number(peakCount) / kpis.total_rows) * 100)
    : null;

  const cleanDimName = dimension ? dimension.replace(/_/g, " ") : "cohort";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 font-sans">
      
      {/* ──── CARD 1: TOTAL RECORDS ──── */}
      <div
        onClick={() => handleCopy("records", String(kpis.total_rows))}
        className="group relative rounded-xl border border-stone-200/90 dark:border-white/[0.08] bg-white dark:bg-[#0c0d12] p-4 sm:p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)] hover:border-stone-300 dark:hover:border-white/15 transition-all duration-200 cursor-pointer select-none flex flex-col justify-between"
      >
        <div>
          <div className="flex items-center justify-between gap-2 h-5">
            <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-500 dark:text-zinc-400 truncate">
              Total Records
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              {copiedId === "records" ? (
                <span className="inline-flex items-center gap-1 text-[10px] font-medium text-stone-700 dark:text-zinc-300 bg-stone-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                  Copied
                </span>
              ) : (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCopy("records", String(kpis.total_rows));
                  }}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 transition-opacity"
                  title="Copy value"
                >
                  <Copy className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="mt-3">
            <div className="text-2xl sm:text-[28px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100">
              <AnimatedNumber value={kpis.total_rows} format={(n) => formatSmartCompact(n)} duration={0.8} />
            </div>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-stone-100 dark:border-white/[0.05] flex items-center justify-between gap-2 text-xs text-stone-500 dark:text-zinc-400">
          <span className="truncate font-medium">{Number(kpis.total_rows).toLocaleString()} in scope</span>
          <span className="shrink-0 font-medium text-stone-600 dark:text-zinc-300">100% active</span>
        </div>
      </div>

      {/* ──── CARD 2: TOTAL VOLUME / AGGREGATE ──── */}
      <div
        onClick={() => handleCopy("metric", String(metricDisplayVal ?? ""))}
        className="group relative rounded-xl border border-stone-200/90 dark:border-white/[0.08] bg-white dark:bg-[#0c0d12] p-4 sm:p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)] hover:border-stone-300 dark:hover:border-white/15 transition-all duration-200 cursor-pointer select-none flex flex-col justify-between"
      >
        <div>
          <div className="flex items-center justify-between gap-2 h-5">
            <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-500 dark:text-zinc-400 truncate">
              Total Volume
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              {copiedId === "metric" ? (
                <span className="inline-flex items-center gap-1 text-[10px] font-medium text-stone-700 dark:text-zinc-300 bg-stone-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                  Copied
                </span>
              ) : (
                <>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-stone-100 dark:bg-zinc-800 text-stone-600 dark:text-zinc-400 border border-stone-200/60 dark:border-white/5">
                    {metricMeta.operation}
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopy("metric", String(metricDisplayVal ?? ""));
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 transition-opacity"
                    title="Copy value"
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="mt-3">
            <div className="text-2xl sm:text-[28px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100">
              {metricDisplayVal !== null && metricDisplayVal !== undefined && metricDisplayVal.constructor === Number && Number.isFinite(metricDisplayVal) ? (
                <AnimatedNumber value={Number(metricDisplayVal)} format={(n) => formatSmartCompact(n)} duration={0.8} />
              ) : (
                metricStr
              )}
            </div>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-stone-100 dark:border-white/[0.05] flex items-center justify-between gap-2 text-xs text-stone-500 dark:text-zinc-400">
          <span className="truncate font-medium" title={metricMeta.cleanName}>
            {metricMeta.cleanName}
          </span>
          <span className="shrink-0 font-medium text-stone-600 dark:text-zinc-300">
            {isSum && kpis.mean !== null ? `avg ~${formatSmartCompact(kpis.mean)}` : isAvg && kpis.sum !== null ? `sum ${formatSmartCompact(kpis.sum)}` : "Aggregated"}
          </span>
        </div>
      </div>

      {/* ──── CARD 3: TOP CONTRIBUTOR ──── */}
      <div
        onClick={() => handleCopy("peak", String(rawPeakDim ?? ""))}
        className="group relative rounded-xl border border-stone-200/90 dark:border-white/[0.08] bg-white dark:bg-[#0c0d12] p-4 sm:p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)] hover:border-stone-300 dark:hover:border-white/15 transition-all duration-200 cursor-pointer select-none flex flex-col justify-between"
      >
        <div>
          <div className="flex items-center justify-between gap-2 h-5">
            <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-500 dark:text-zinc-400 truncate">
              Top Contributor
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              {copiedId === "peak" ? (
                <span className="inline-flex items-center gap-1 text-[10px] font-medium text-stone-700 dark:text-zinc-300 bg-stone-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                  Copied
                </span>
              ) : (
                <>
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-stone-100 dark:bg-zinc-800 text-stone-600 dark:text-zinc-400 border border-stone-200/60 dark:border-white/5">
                    #1
                  </span>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopy("peak", String(rawPeakDim ?? ""));
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 transition-opacity"
                    title="Copy value"
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="mt-3">
            <div className="text-xl sm:text-[24px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100 truncate" title={formattedPeakDim}>
              {formattedPeakDim}
            </div>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-stone-100 dark:border-white/[0.05] flex items-center justify-between gap-2 text-xs text-stone-500 dark:text-zinc-400">
          <span className="truncate font-medium">
            {peakCount ? `${Number(peakCount).toLocaleString()} records` : cleanDimName}
          </span>
          <span className="shrink-0 font-medium text-stone-600 dark:text-zinc-300">
            {peakRowPct !== null ? `${peakRowPct < 1 ? "<1%" : peakRowPct.toFixed(1) + "%"} share` : "Leading"}
          </span>
        </div>
      </div>

      {/* ──── CARD 4: TOTAL GROUPS ──── */}
      <div
        onClick={() => handleCopy("groups", String(queryResult.total_groups))}
        className="group relative rounded-xl border border-stone-200/90 dark:border-white/[0.08] bg-white dark:bg-[#0c0d12] p-4 sm:p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)] hover:border-stone-300 dark:hover:border-white/15 transition-all duration-200 cursor-pointer select-none flex flex-col justify-between"
      >
        <div>
          <div className="flex items-center justify-between gap-2 h-5">
            <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-500 dark:text-zinc-400 truncate">
              Total Groups
            </span>
            <div className="flex items-center gap-1.5 shrink-0">
              {copiedId === "groups" ? (
                <span className="inline-flex items-center gap-1 text-[10px] font-medium text-stone-700 dark:text-zinc-300 bg-stone-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                  <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                  Copied
                </span>
              ) : (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleCopy("groups", String(queryResult.total_groups));
                  }}
                  className="opacity-0 group-hover:opacity-100 p-0.5 rounded text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 transition-opacity"
                  title="Copy value"
                >
                  <Copy className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="mt-3">
            <div className="text-2xl sm:text-[28px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100">
              <AnimatedNumber value={queryResult.total_groups} format={(n) => formatSmartCompact(n)} duration={0.8} />
            </div>
          </div>
        </div>

        <div className="mt-3 pt-2.5 border-t border-stone-100 dark:border-white/[0.05] flex items-center justify-between gap-2 text-xs text-stone-500 dark:text-zinc-400">
          <span className="truncate font-medium">Distinct segments</span>
          <span className="shrink-0 font-medium text-stone-600 dark:text-zinc-300">
            {dimension ? dimension.replace(/_/g, " ") : "categories"}
          </span>
        </div>
      </div>

    </div>
  );
};
