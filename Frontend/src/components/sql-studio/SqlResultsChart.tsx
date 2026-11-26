import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { BarChart3, LineChart, ScatterChart } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTheme } from "@/hooks/useTheme";
import type { DataRow } from "@/types/api";
import type { SqlChartType } from "./types";
import { isFiniteNumber } from "./utils";

interface SqlResultsChartProps {
  columns: string[];
  records: DataRow[];
  numericCols: Set<string>;
}

const CHART_TYPES: ReadonlyArray<{ id: SqlChartType; label: string; icon: typeof BarChart3 }> = [
  { id: "bar", label: "Bar", icon: BarChart3 },
  { id: "horizontal_bar", label: "H-Bar", icon: BarChart3 },
  { id: "line", label: "Line", icon: LineChart },
  { id: "scatter", label: "Scatter", icon: ScatterChart },
];

export function SqlResultsChart({ columns, records, numericCols }: SqlResultsChartProps) {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);

  const numericList = useMemo(() => columns.filter((col) => numericCols.has(col)), [columns, numericCols]);
  const nonNumericList = useMemo(() => columns.filter((col) => !numericCols.has(col)), [columns, numericCols]);

  const defaultX = nonNumericList[0] ?? columns[0] ?? "";
  const defaultY = numericList[0] ?? columns[1] ?? columns[0] ?? "";

  const [chartType, setChartType] = useState<SqlChartType>("bar");
  const [xCol, setXCol] = useState<string>(defaultX);
  const [yCol, setYCol] = useState<string>(defaultY);

  const resolvedX = columns.includes(xCol) ? xCol : defaultX;
  const resolvedY = columns.includes(yCol) ? yCol : defaultY;

  const slicedRecords = useMemo(() => records.slice(0, 60), [records]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !resolvedX || !resolvedY || slicedRecords.length === 0) return undefined;

    const instance = echarts.init(el, undefined, { renderer: "svg" });
    chartInstanceRef.current = instance;

    const accentColor = isDark ? "#34D399" : "#800000";
    const textColor = isDark ? "#A1A1AA" : "#57534E";
    const gridColor = isDark ? "rgba(255,255,255,0.08)" : "rgba(28,25,23,0.1)";

    const xLabels = slicedRecords.map((row) => String(row[resolvedX] ?? "null"));
    const yValues = slicedRecords.map((row) => {
      const val = row[resolvedY];
      if (isFiniteNumber(val)) return val;
      const parsed = Number(val);
      return Number.isFinite(parsed) ? parsed : 0;
    });

    const isHorizontal = chartType === "horizontal_bar";
    const isScatter = chartType === "scatter";

    const option: echarts.EChartsOption = {
      backgroundColor: "transparent",
      tooltip: {
        trigger: isScatter ? "item" : "axis",
        backgroundColor: isDark ? "#12151F" : "#FFFFFF",
        borderColor: isDark ? "rgba(255,255,255,0.14)" : "#D6D3D1",
        textStyle: { color: isDark ? "#F4F4F5" : "#1C1917", fontSize: 12 },
      },
      grid: {
        top: 24,
        right: 20,
        bottom: 42,
        left: 56,
        containLabel: true,
      },
      xAxis: isHorizontal
        ? {
            type: "value",
            axisLabel: { color: textColor, fontSize: 11 },
            splitLine: { lineStyle: { color: gridColor } },
          }
        : isScatter
          ? {
              type: "value",
              name: resolvedX,
              axisLabel: { color: textColor, fontSize: 11 },
              splitLine: { lineStyle: { color: gridColor } },
            }
          : {
              type: "category",
              data: xLabels,
              axisLabel: { color: textColor, fontSize: 11, rotate: xLabels.length > 10 ? 28 : 0 },
              axisLine: { lineStyle: { color: gridColor } },
            },
      yAxis: isHorizontal
        ? {
            type: "category",
            data: xLabels,
            axisLabel: { color: textColor, fontSize: 11 },
            axisLine: { lineStyle: { color: gridColor } },
          }
        : {
            type: "value",
            name: resolvedY,
            axisLabel: { color: textColor, fontSize: 11 },
            splitLine: { lineStyle: { color: gridColor } },
          },
      series: [
        isScatter
          ? {
              type: "scatter",
              symbolSize: 8,
              itemStyle: { color: accentColor },
              data: slicedRecords.map((row, idx) => {
                const rawX = row[resolvedX];
                const numX = isFiniteNumber(rawX) ? rawX : idx;
                return [numX, yValues[idx] ?? 0];
              }),
            }
          : {
              type: chartType === "line" ? "line" : "bar",
              name: resolvedY,
              data: yValues,
              smooth: chartType === "line",
              itemStyle: {
                color: accentColor,
                borderRadius: chartType === "line" ? 0 : isHorizontal ? [0, 4, 4, 0] : [4, 4, 0, 0],
              },
              barMaxWidth: 42,
            },
      ],
    };

    instance.setOption(option);

    const resizeObserver = new ResizeObserver(() => {
      instance.resize();
    });
    resizeObserver.observe(el);

    return () => {
      resizeObserver.disconnect();
      instance.dispose();
      chartInstanceRef.current = null;
    };
  }, [chartType, resolvedX, resolvedY, slicedRecords, isDark]);

  if (columns.length === 0 || records.length === 0) {
    return (
      <div className="rounded-xl border border-stone-300/80 dark:border-white/10 bg-white dark:bg-[#0D0F16] p-8 text-center text-xs font-mono text-stone-500 dark:text-zinc-400">
        No result rows available to plot.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-stone-300/90 dark:border-white/10 bg-white dark:bg-[#0D0F16] overflow-hidden">
      <div className="px-3.5 py-2.5 border-b border-stone-200 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#12151F] flex flex-wrap items-center justify-between gap-2.5">
        <div className="inline-flex items-center p-0.5 rounded-[7px] border border-stone-300/80 dark:border-white/10 bg-white dark:bg-[#0B0D13] gap-0.5">
          {CHART_TYPES.map((ct) => {
            const Icon = ct.icon;
            const active = chartType === ct.id;
            return (
              <button
                key={ct.id}
                type="button"
                onClick={() => setChartType(ct.id)}
                className={`px-2 py-1 rounded-[5px] text-[11px] font-medium inline-flex items-center gap-1 cursor-pointer transition-colors ${
                  active
                    ? "bg-[#800000] text-white dark:bg-white dark:text-zinc-950"
                    : "text-stone-600 hover:text-stone-900 dark:text-zinc-400 dark:hover:text-white"
                }`}
              >
                <Icon className="h-3 w-3" />
                <span>{ct.label}</span>
              </button>
            );
          })}
        </div>

        <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
          <div className="inline-flex items-center gap-1.5">
            <span className="text-[11px] text-stone-500 dark:text-zinc-400">X:</span>
            <Select value={resolvedX} onValueChange={setXCol}>
              <SelectTrigger className="h-7 w-auto min-w-[6.5rem] rounded-[6px] text-xs font-mono px-2">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {columns.map((col) => (
                  <SelectItem key={`x-${col}`} value={col} className="text-xs font-mono">
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="inline-flex items-center gap-1.5">
            <span className="text-[11px] text-stone-500 dark:text-zinc-400">Y:</span>
            <Select value={resolvedY} onValueChange={setYCol}>
              <SelectTrigger className="h-7 w-auto min-w-[6.5rem] rounded-[6px] text-xs font-mono px-2">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {columns.map((col) => (
                  <SelectItem key={`y-${col}`} value={col} className="text-xs font-mono">
                    {col}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {records.length > 60 && (
            <span className="text-[10px] text-stone-500 dark:text-zinc-400">
              Showing first 60 rows
            </span>
          )}
        </div>
      </div>

      <div ref={containerRef} className="w-full h-[320px] p-2" aria-label="SQL Result Chart" />
    </div>
  );
}
