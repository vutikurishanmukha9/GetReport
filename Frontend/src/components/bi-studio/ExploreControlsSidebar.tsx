import { BarChart3, Calendar, Filter, Plus, TrendingUp } from "lucide-react";
import { AnimatedDeleteButton, Button, ShimmerActionButton } from "@/components/ui/button";
import {
  AGGREGATIONS,
  CHART_TYPES,
  COLOR_PALETTES,
  FILTER_OPERATORS,
  TIME_GRAINS,
  type FilterItem,
  type SchemaColumnMeta,
} from "./constants";

interface ExploreControlsSidebarProps {
  mobileVisible: boolean;
  recommendations: any[];
  onApplyRecommendation: (rec: any) => void;
  chartType: string;
  setChartType: (t: string) => void;
  dimension: string;
  onChangeDimension: (d: string) => void;
  schemaColumns: SchemaColumnMeta[];
  isTemporalDimension: boolean;
  timeGrain: string;
  setTimeGrain: (tg: string) => void;
  secondaryMetric: string;
  setSecondaryMetric: (m: string) => void;
  secondaryAggregate: string;
  setSecondaryAggregate: (a: string) => void;
  numericCols: string[];
  aggregate: string;
  setAggregate: (a: string) => void;
  metric: string;
  setMetric: (m: string) => void;
  secondaryDimension: string;
  setSecondaryDimension: (d: string) => void;
  barMode: "grouped" | "stacked";
  setBarMode: (mode: "grouped" | "stacked") => void;
  filters: FilterItem[];
  onAddFilter: () => void;
  onRemoveFilter: (id: string) => void;
  onUpdateFilter: (id: string, field: keyof FilterItem, val: string) => void;
  rowLimit: number;
  setRowLimit: (n: number) => void;
  sortOrder: string;
  setSortOrder: (s: string) => void;
  colorPalette: string;
  setColorPalette: (p: string) => void;
  onMobileViewChart: () => void;
}

export function ExploreControlsSidebar({
  mobileVisible,
  recommendations,
  onApplyRecommendation,
  chartType,
  setChartType,
  dimension,
  onChangeDimension,
  schemaColumns,
  isTemporalDimension,
  timeGrain,
  setTimeGrain,
  secondaryMetric,
  setSecondaryMetric,
  secondaryAggregate,
  setSecondaryAggregate,
  numericCols,
  aggregate,
  setAggregate,
  metric,
  setMetric,
  secondaryDimension,
  setSecondaryDimension,
  barMode,
  setBarMode,
  filters,
  onAddFilter,
  onRemoveFilter,
  onUpdateFilter,
  rowLimit,
  setRowLimit,
  sortOrder,
  setSortOrder,
  colorPalette,
  setColorPalette,
  onMobileViewChart,
}: ExploreControlsSidebarProps) {
  return (
    <div
      className={`lg:col-span-4 p-4 sm:p-5 border-b lg:border-b-0 lg:border-r border-stone-200 bg-[#F4F0EA]/70 dark:border-slate-800 dark:bg-[#0c0d12] space-y-4 sm:space-y-5 ${
        mobileVisible ? "block" : "hidden lg:block"
      }`}
    >
      {recommendations.length > 0 && (
        <div className="space-y-2 pb-2 border-b border-stone-200 dark:border-slate-800/80">
          <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
            <TrendingUp className="h-3 w-3 text-emerald-600 dark:text-emerald-400" />
            <span>Suggested Explorations</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {recommendations.slice(0, 4).map((rec: any) => (
              <button
                key={rec.id}
                type="button"
                onClick={() => onApplyRecommendation(rec)}
                className="text-left text-[11px] px-2 py-1 rounded-lg border border-stone-200 bg-white hover:bg-stone-100 hover:border-[#800000]/40 text-stone-700 hover:text-stone-950 dark:border-white/10 dark:bg-[#0e1017] dark:hover:bg-white/[0.08] dark:hover:border-emerald-500/40 dark:text-zinc-300 dark:hover:text-white transition group flex items-center gap-1.5 cursor-pointer max-w-full font-sans shadow-2xs"
                title={rec.description}
              >
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 dark:bg-emerald-400 shrink-0" />
                <span className="truncate max-w-[130px] font-medium">{rec.title}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-1.5">
        <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
          Visualization Type
        </span>
        <div className="grid grid-cols-3 gap-1.5">
          {CHART_TYPES.map((ct) => {
            const Icon = ct.icon;
            const isActive = chartType === ct.id;
            return (
              <button
                key={ct.id}
                type="button"
                onClick={() => setChartType(ct.id)}
                className={`flex flex-col items-center justify-center p-2 rounded-xl border text-center transition font-sans cursor-pointer ${
                  isActive
                    ? "bg-[#800000] border-[#800000] text-white shadow-sm font-semibold dark:bg-white dark:border-white dark:text-zinc-950 dark:shadow-sm"
                    : "bg-white border-stone-200 text-stone-700 hover:bg-stone-100 hover:text-stone-950 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-400 dark:hover:bg-slate-800 dark:hover:text-slate-200"
                }`}
                title={ct.desc}
              >
                <Icon className="h-4 w-4 mb-1" />
                <span className="text-[10px] font-medium leading-tight">{ct.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="space-y-1.5">
        <label htmlFor="dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
          Dimension (Group By / X-Axis)
        </label>
        <select
          id="dimension-select"
          aria-label="Dimension (Group By / X-Axis)"
          value={dimension}
          onChange={(e) => onChangeDimension(e.target.value)}
          className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono shadow-2xs"
        >
          {schemaColumns.map((col) => (
            <option key={col.name} value={col.name}>
              {col.name} ({col.type_name})
            </option>
          ))}
        </select>
      </div>

      {isTemporalDimension && (
        <div className="space-y-1.5 p-2.5 rounded-xl bg-sky-50 border border-sky-200 dark:bg-sky-950/20 dark:border-sky-800/40">
          <div className="flex items-center gap-1.5 text-sky-700 dark:text-sky-400 text-xs font-semibold">
            <Calendar className="h-3.5 w-3.5" />
            <span>Time Grain (Date Truncation)</span>
          </div>
          <select
            id="time-grain-select"
            aria-label="Time Grain"
            value={timeGrain}
            onChange={(e) => setTimeGrain(e.target.value)}
            className="w-full bg-white border border-sky-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none font-mono dark:bg-[#0e1017] dark:border-sky-800/60 dark:text-sky-200 shadow-2xs"
          >
            {TIME_GRAINS.map((tg) => (
              <option key={tg.id} value={tg.id}>
                {tg.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {chartType === "echarts_combo" && (
        <div className="space-y-2 p-3 rounded-xl bg-purple-50 border border-purple-200 dark:bg-purple-950/20 dark:border-purple-800/40">
          <div className="flex items-center gap-1.5 text-purple-700 dark:text-purple-400 text-xs font-semibold">
            <TrendingUp className="h-3.5 w-3.5" />
            <span>Secondary Metric (Line on Right Axis)</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <select
              id="sec-metric-select"
              aria-label="Secondary Metric"
              value={secondaryMetric}
              onChange={(e) => setSecondaryMetric(e.target.value)}
              className="bg-white border border-purple-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 font-mono dark:bg-[#0e1017] dark:border-purple-800/60 dark:text-purple-200 shadow-2xs"
            >
              <option value="">None</option>
              {numericCols.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <select
              id="sec-agg-select"
              aria-label="Secondary Aggregate"
              value={secondaryAggregate}
              onChange={(e) => setSecondaryAggregate(e.target.value)}
              className="bg-white border border-purple-300 text-stone-900 text-xs rounded-lg px-2.5 py-1.5 font-mono dark:bg-[#0e1017] dark:border-purple-800/60 dark:text-purple-200 shadow-2xs"
            >
              {AGGREGATIONS.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      <div className="space-y-1.5">
        <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
          Metric &amp; Aggregation (Y-Axis)
        </span>
        <div className="grid grid-cols-2 gap-2">
          <select
            id="aggregate-select"
            aria-label="Aggregation Function"
            value={aggregate}
            onChange={(e) => setAggregate(e.target.value)}
            className="bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono shadow-2xs"
          >
            {AGGREGATIONS.map((agg) => (
              <option key={agg.id} value={agg.id}>
                {agg.label}
              </option>
            ))}
          </select>

          <select
            id="metric-select"
            aria-label="Metric Column"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            disabled={aggregate === "COUNT" && !metric}
            className="bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 dark:focus:border-white/40 font-mono disabled:opacity-50 disabled:bg-stone-100 dark:disabled:bg-slate-900 shadow-2xs"
          >
            <option value="">(Row Count)</option>
            {numericCols.map((col) => (
              <option key={col} value={col}>
                {col}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="space-y-1.5">
        <div className="flex items-center justify-between">
          <label htmlFor="secondary-dimension-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
            Secondary Breakdown
          </label>
          <span className="text-[10px] text-stone-500 dark:text-zinc-400 font-sans font-medium">
            {chartType === "echarts_heatmap" ? "Required for Heatmap" : "Optional"}
          </span>
        </div>
        <select
          id="secondary-dimension-select"
          aria-label="Secondary Breakdown"
          value={secondaryDimension}
          onChange={(e) => setSecondaryDimension(e.target.value)}
          className={`w-full bg-white border text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none font-mono dark:bg-[#0e1017] dark:text-slate-200 shadow-2xs ${
            chartType === "echarts_heatmap" && !secondaryDimension
              ? "border-amber-500/70"
              : "border-stone-300 dark:border-slate-800"
          }`}
        >
          <option value="">None (Single Series)</option>
          {schemaColumns
            .filter((c) => c.name !== dimension)
            .map((col) => (
              <option key={col.name} value={col.name}>
                {col.name}
              </option>
            ))}
        </select>
        {secondaryDimension && chartType === "echarts_bar" && (
          <div className="flex items-center justify-between pt-1.5 px-0.5">
            <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
              Bar Mode
            </span>
            <div className="flex items-center gap-1 bg-stone-200/80 dark:bg-[#090b10] p-0.5 rounded-lg border border-stone-300 dark:border-slate-800">
              <button
                type="button"
                onClick={() => setBarMode("grouped")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all cursor-pointer ${
                  barMode === "grouped"
                    ? "bg-[#800000] text-white font-bold shadow-xs dark:bg-white dark:text-zinc-950"
                    : "text-stone-700 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white"
                }`}
              >
                Grouped
              </button>
              <button
                type="button"
                onClick={() => setBarMode("stacked")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-mono transition-all cursor-pointer ${
                  barMode === "stacked"
                    ? "bg-[#800000] text-white font-bold shadow-xs dark:bg-white dark:text-zinc-950"
                    : "text-stone-700 hover:text-stone-950 dark:text-slate-400 dark:hover:text-white"
                }`}
              >
                Stacked
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="space-y-2 pt-2 border-t border-stone-200 dark:border-slate-800/80">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400">
            <Filter className="h-3 w-3 text-sky-600 dark:text-sky-400" />
            <span>Ad-Hoc Filters ({filters.length})</span>
          </div>
          <Button
            size="sm"
            variant="ghost"
            onClick={onAddFilter}
            className="h-6 px-2 text-[10px] font-medium text-sky-600 hover:text-sky-700 hover:bg-sky-50 dark:text-sky-400 dark:hover:text-sky-300 dark:hover:bg-sky-500/10 rounded-md gap-1 cursor-pointer"
          >
            <Plus className="h-3 w-3" />
            <span>Add</span>
          </Button>
        </div>

        {filters.length > 0 && (
          <div className="space-y-2">
            {filters.map((f) => (
              <div
                key={f.id}
                className="p-2 rounded-xl bg-white border border-stone-200 dark:bg-[#0e1017] dark:border-slate-800 space-y-1.5 text-xs shadow-2xs"
              >
                <div className="flex items-center gap-1.5">
                  <select
                    value={f.column}
                    onChange={(e) => onUpdateFilter(f.id, "column", e.target.value)}
                    aria-label="Filter Column"
                    className="bg-stone-50 border border-stone-300 rounded-md px-1.5 py-1 text-stone-900 text-[11px] font-mono flex-1 truncate dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200"
                  >
                    {schemaColumns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                  <select
                    value={f.operator}
                    onChange={(e) => onUpdateFilter(f.id, "operator", e.target.value)}
                    aria-label="Filter Operator"
                    className="bg-stone-50 border border-stone-300 rounded-md px-1.5 py-1 text-stone-900 text-[11px] font-mono dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200"
                  >
                    {FILTER_OPERATORS.map((op) => (
                      <option key={op.id} value={op.id}>
                        {op.label}
                      </option>
                    ))}
                  </select>
                  <AnimatedDeleteButton
                    size="icon"
                    variant="ghost"
                    className="h-7 w-7 rounded-lg text-stone-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 shrink-0 cursor-pointer"
                    onDelete={() => onRemoveFilter(f.id)}
                    title="Remove filter"
                  />
                </div>
                {!f.operator.includes("null") && (
                  <input
                    type="text"
                    placeholder="Comparison value..."
                    aria-label="Filter Comparison Value"
                    value={f.value}
                    onChange={(e) => onUpdateFilter(f.id, "value", e.target.value)}
                    className="w-full bg-stone-50 border border-stone-300 rounded-md px-2 py-1 text-stone-900 text-[11px] font-mono focus:outline-none focus:border-[#800000]/50 dark:bg-[#121215] dark:border-slate-700 dark:text-slate-200 dark:focus:border-white/30"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 pt-1">
        <div className="space-y-1">
          <label htmlFor="row-limit-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
            Row Limit
          </label>
          <select
            id="row-limit-select"
            aria-label="Row Limit"
            value={rowLimit}
            onChange={(e) => setRowLimit(Number(e.target.value))}
            className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 font-mono shadow-2xs"
          >
            <option value={5}>Top 5</option>
            <option value={10}>Top 10</option>
            <option value={15}>Top 15</option>
            <option value={25}>Top 25</option>
            <option value={50}>Top 50</option>
            <option value={100}>Top 100</option>
            <option value={250}>Top 250</option>
            <option value={500}>Top 500</option>
          </select>
        </div>

        <div className="space-y-1">
          <label htmlFor="sort-order-select" className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
            Sort Order
          </label>
          <select
            id="sort-order-select"
            aria-label="Sort Order"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value)}
            className="w-full bg-white border border-stone-300 text-stone-900 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-[#800000]/50 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-200 font-mono shadow-2xs"
          >
            <option value="desc">High to Low (Desc)</option>
            <option value="asc">Low to High (Asc)</option>
          </select>
        </div>
      </div>

      <div className="space-y-1.5 pt-2 border-t border-stone-200 dark:border-slate-800/80">
        <span className="text-[11px] font-sans font-semibold uppercase tracking-[0.15em] text-stone-600 dark:text-zinc-400 block">
          Color Palette
        </span>
        <div className="space-y-1">
          {COLOR_PALETTES.map((pal) => (
            <button
              key={pal.id}
              type="button"
              onClick={() => setColorPalette(pal.id)}
              className={`w-full flex items-center justify-between px-3 py-1.5 rounded-lg border text-xs transition cursor-pointer ${
                colorPalette === pal.id
                  ? "bg-[#800000] border-[#800000] text-white shadow-xs font-semibold dark:bg-white dark:border-white dark:text-zinc-950"
                  : "bg-white border-stone-200 text-stone-700 hover:bg-stone-100 hover:text-stone-950 dark:bg-[#0e1017] dark:border-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
            >
              <span>{pal.label}</span>
              <div className="flex items-center gap-1">
                {pal.preview.map((c) => (
                  <span key={c} className="h-2.5 w-2.5 rounded-full ring-1 ring-black/10 dark:ring-white/20" style={{ backgroundColor: c }} />
                ))}
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="lg:hidden pt-3 border-t border-slate-800/80">
        <ShimmerActionButton
          onClick={onMobileViewChart}
          className="w-full h-10 text-xs justify-center"
          icon={<BarChart3 className="h-3.5 w-3.5 text-white dark:text-zinc-950" />}
        >
          <span>Apply &amp; View Visual Chart</span>
        </ShimmerActionButton>
      </div>
    </div>
  );
}
