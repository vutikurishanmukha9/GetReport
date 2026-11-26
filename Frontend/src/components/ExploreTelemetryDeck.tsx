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

const resolveMetricDisplayVal = (
  kpis: ExploreTelemetryDeckProps["queryResult"]["kpis"],
  aggregate: string,
): number | null => {
  if (aggregate === "AVG") return kpis.mean ?? kpis.sum;
  if (aggregate === "MAX") return kpis.max ?? kpis.sum;
  if (aggregate === "MIN") return kpis.min ?? kpis.sum;
  if (aggregate === "SUM") return kpis.sum ?? kpis.mean;
  return kpis.sum ?? kpis.mean;
};

interface TelemetryKpiCardProps {
  title: string;
  isCopied: boolean;
  onCopy: () => void;
  badge?: string;
  value: React.ReactNode;
  valueTitle?: string;
  compactValue?: boolean;
  footerLeft: React.ReactNode;
  footerLeftTitle?: string;
  footerRight: React.ReactNode;
}

function TelemetryKpiCard({
  title,
  isCopied,
  onCopy,
  badge,
  value,
  valueTitle,
  compactValue = false,
  footerLeft,
  footerLeftTitle,
  footerRight,
}: TelemetryKpiCardProps) {
  return (
    <div className="group relative rounded-xl border border-stone-200/90 dark:border-white/[0.08] bg-white dark:bg-[#0c0d12] p-4 sm:p-5 shadow-[0_1px_2px_rgba(0,0,0,0.03)] hover:border-stone-300 dark:hover:border-white/15 transition-all duration-200 select-none flex flex-col justify-between">
      <button
        type="button"
        onClick={onCopy}
        aria-label={`Copy ${title}`}
        className="absolute inset-0 rounded-xl cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#800000] dark:focus-visible:ring-emerald-400"
      />
      <div className="pointer-events-none relative z-10">
        <div className="flex items-center justify-between gap-2 h-5">
          <span className="text-[11px] font-semibold tracking-wider uppercase text-stone-500 dark:text-zinc-400 truncate">
            {title}
          </span>
          <div className="flex items-center gap-1.5 shrink-0 pointer-events-auto">
            {isCopied ? (
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-stone-700 dark:text-zinc-300 bg-stone-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">
                <Check className="h-2.5 w-2.5 text-emerald-600 dark:text-emerald-400" />
                Copied
              </span>
            ) : (
              <>
                {badge && (
                  <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-stone-100 dark:bg-zinc-800 text-stone-600 dark:text-zinc-400 border border-stone-200/60 dark:border-white/5">
                    {badge}
                  </span>
                )}
                <button
                  type="button"
                  onClick={onCopy}
                  className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100 p-0.5 rounded text-stone-400 hover:text-stone-700 dark:text-zinc-500 dark:hover:text-zinc-200 transition-opacity cursor-pointer"
                  title="Copy value"
                  aria-label={`Copy ${title} value`}
                >
                  <Copy className="h-3.5 w-3.5" />
                </button>
              </>
            )}
          </div>
        </div>

        <div className="mt-3">
          <div
            className={
              compactValue
                ? "text-xl sm:text-[24px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100 truncate"
                : "text-2xl sm:text-[28px] leading-tight font-bold tracking-tight text-stone-900 dark:text-zinc-100"
            }
            title={valueTitle}
          >
            {value}
          </div>
        </div>
      </div>

      <div className="pointer-events-none relative z-10 mt-3 pt-2.5 border-t border-stone-100 dark:border-white/[0.05] flex items-center justify-between gap-2 text-xs text-stone-500 dark:text-zinc-400">
        <span className="truncate font-medium" title={footerLeftTitle}>
          {footerLeft}
        </span>
        <span className="shrink-0 font-medium text-stone-600 dark:text-zinc-300">
          {footerRight}
        </span>
      </div>
    </div>
  );
}

function buildTelemetryMetrics(
  kpis: NonNullable<ExploreTelemetryDeckProps["queryResult"]>["kpis"],
  records: NonNullable<ExploreTelemetryDeckProps["queryResult"]>["records"],
  aggregate: string,
  metric: string,
  dimension: string,
) {
  const metricDisplayVal = resolveMetricDisplayVal(kpis, aggregate);
  const metricMeta = parseMetricMeta(kpis.metric_label, aggregate, metric);
  const hasNumericMetric = metricDisplayVal !== null && Number.isFinite(metricDisplayVal);
  const metricStr = hasNumericMetric ? formatSmartCompact(metricDisplayVal) : "—";

  let volumeFooterRight = "Aggregated";
  if (aggregate === "SUM" && kpis.mean !== null) {
    volumeFooterRight = `avg ~${formatSmartCompact(kpis.mean)}`;
  } else if (aggregate === "AVG" && kpis.sum !== null) {
    volumeFooterRight = `sum ${formatSmartCompact(kpis.sum)}`;
  }

  const rawPeakDim = records?.[0]?.[dimension];
  const formattedPeakDim = formatDimensionValue(rawPeakDim);
  const peakCount = records?.[0]?.count;
  const peakRowPct =
    kpis.total_rows > 0 && peakCount !== undefined && peakCount !== null
      ? (Number(peakCount) / kpis.total_rows) * 100
      : null;

  const cleanDimName = dimension ? dimension.replace(/_/g, " ") : "cohort";
  const shareText = peakRowPct === null ? "Leading" : `${peakRowPct < 1 ? "<1%" : `${peakRowPct.toFixed(1)}%`} share`;

  return {
    metricDisplayVal,
    metricMeta,
    hasNumericMetric,
    metricStr,
    volumeFooterRight,
    rawPeakDim,
    formattedPeakDim,
    peakFooterLeft: peakCount ? `${Number(peakCount).toLocaleString()} records` : cleanDimName,
    shareText,
    groupsFooterRight: dimension ? dimension.replace(/_/g, " ") : "categories",
  };
}

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

  const vm = buildTelemetryMetrics(kpis, queryResult.records, aggregate, metric, dimension);

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 font-sans">
      <TelemetryKpiCard
        title="Total Records"
        isCopied={copiedId === "records"}
        onCopy={() => handleCopy("records", String(kpis.total_rows))}
        value={<AnimatedNumber value={kpis.total_rows} format={(n) => formatSmartCompact(n)} duration={0.8} />}
        footerLeft={`${Number(kpis.total_rows).toLocaleString()} in scope`}
        footerRight="100% active"
      />

      <TelemetryKpiCard
        title="Total Volume"
        isCopied={copiedId === "metric"}
        onCopy={() => handleCopy("metric", String(vm.metricDisplayVal ?? ""))}
        badge={vm.metricMeta.operation}
        value={
          vm.hasNumericMetric ? (
            <AnimatedNumber value={Number(vm.metricDisplayVal)} format={(n) => formatSmartCompact(n)} duration={0.8} />
          ) : (
            vm.metricStr
          )
        }
        footerLeft={vm.metricMeta.cleanName}
        footerLeftTitle={vm.metricMeta.cleanName}
        footerRight={vm.volumeFooterRight}
      />

      <TelemetryKpiCard
        title="Top Contributor"
        isCopied={copiedId === "peak"}
        onCopy={() => handleCopy("peak", String(vm.rawPeakDim ?? ""))}
        badge="#1"
        compactValue
        value={vm.formattedPeakDim}
        valueTitle={vm.formattedPeakDim}
        footerLeft={vm.peakFooterLeft}
        footerRight={vm.shareText}
      />

      <TelemetryKpiCard
        title="Total Groups"
        isCopied={copiedId === "groups"}
        onCopy={() => handleCopy("groups", String(queryResult.total_groups))}
        value={<AnimatedNumber value={queryResult.total_groups} format={(n) => formatSmartCompact(n)} duration={0.8} />}
        footerLeft="Distinct segments"
        footerRight={vm.groupsFooterRight}
      />
    </div>
  );
};
