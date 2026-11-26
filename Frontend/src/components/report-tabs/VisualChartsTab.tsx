import { memo, useMemo, useState } from "react";
import { TrendingUp, type LucideIcon } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { Charts } from "@/types/api";
import { HiPlotParallelCoordinates, type HiPlotPayload } from "../HiPlotParallelCoordinates";

interface ChartTabOption {
  id: string;
  label: string;
  icon: LucideIcon;
}

const SafeChartImage = memo(
  ({ base64Src, alt, className }: { base64Src: string; alt: string; className?: string }) => {
    const src = useMemo(() => {
      if (!base64Src) return "";
      return base64Src.startsWith("data:") ? base64Src : `data:image/png;base64,${base64Src}`;
    }, [base64Src]);

    if (!src) return null;

    return <img src={src} alt={alt} className={className} loading="lazy" />;
  },
);
SafeChartImage.displayName = "SafeChartImage";

function CorrelationChartSection({ correlation }: { correlation: Charts["correlation_heatmap"] }) {
  if (!correlation) return null;
  // SAFETY: correlation_heatmap union type is safely narrowed to image/narrative object or string
  const imgSrc = (correlation as { image?: string })?.image ?? (correlation as string);
  // SAFETY: correlation_heatmap union type is safely narrowed to narrative property or default copy
  const narrativeText =
    (correlation as { narrative?: string })?.narrative ??
    "Pearson correlation matrix. Warm tones indicate strong positive interactions, while dark charcoal represents negative inverse correlations.";

  return (
    <div className="space-y-6">
      <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-3 sm:p-5 max-w-3xl mx-auto shadow-2xs">
        <SafeChartImage
          base64Src={imgSrc}
          alt="Correlation Heatmap"
          className="max-h-[380px] w-full object-contain rounded-xl"
        />
      </div>
      <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
        <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
          Correlation Matrix Narrative
        </div>
        <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">{narrativeText}</p>
      </div>
    </div>
  );
}

function DistributionsChartSection({
  distributions,
  activeDistIndex,
  onSelectIndex,
}: {
  distributions: NonNullable<Charts["distributions"]>;
  activeDistIndex: number;
  onSelectIndex: (idx: number) => void;
}) {
  const current = distributions[activeDistIndex];
  return (
    <div className="space-y-6">
      <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
        {distributions.map((item, idx) => (
          <button
            key={`dist-tab-${item.column}`}
            type="button"
            onClick={() => onSelectIndex(idx)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
              activeDistIndex === idx
                ? "bg-primary text-primary-foreground shadow-2xs"
                : "bg-muted/50 hover:bg-muted text-muted-foreground"
            }`}
          >
            {item.column}
          </button>
        ))}
      </div>

      <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
        <SafeChartImage
          base64Src={current?.image ?? ""}
          alt={`Distribution for ${current?.column}`}
          className="max-h-[340px] w-auto object-contain rounded-xl"
        />
      </div>

      <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
        <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">
          {current?.column} Distribution Analysis
        </div>
        <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">{current?.narrative}</p>
      </div>
    </div>
  );
}

function CompositionChartSection({
  barCharts,
  donutChart,
  activeBarIndex,
  onSelectIndex,
}: {
  barCharts?: Charts["bar_charts"];
  donutChart?: Charts["donut_chart"];
  activeBarIndex: number;
  onSelectIndex: (idx: number) => void;
}) {
  const selectedBar = barCharts?.[activeBarIndex];
  const showDonut = activeBarIndex === -1 && Boolean(donutChart);

  return (
    <div className="space-y-6">
      {barCharts && barCharts.length > 0 && (
        <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
          {barCharts.map((item, idx) => (
            <button
              key={`bar-tab-${item.column}`}
              type="button"
              onClick={() => onSelectIndex(idx)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                activeBarIndex === idx
                  ? "bg-primary text-primary-foreground shadow-2xs"
                  : "bg-muted/50 hover:bg-muted text-muted-foreground"
              }`}
            >
              {item.column}
            </button>
          ))}
          {donutChart && (
            <button
              type="button"
              onClick={() => onSelectIndex(-1)}
              className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
                activeBarIndex === -1
                  ? "bg-primary text-primary-foreground shadow-2xs"
                  : "bg-muted/50 hover:bg-muted text-muted-foreground"
              }`}
            >
              {donutChart.column} (Donut)
            </button>
          )}
        </div>
      )}

      {showDonut && donutChart ? (
        <ChartImagePanel
          src={donutChart.image}
          alt={`Composition of ${donutChart.column}`}
          title="Donut Composition Narrative"
          narrative={donutChart.narrative}
        />
      ) : selectedBar ? (
        <ChartImagePanel
          src={selectedBar.image}
          alt={`Category breakdown of ${selectedBar.column}`}
          title="Categorical Breakdown"
          narrative={selectedBar.narrative}
        />
      ) : (
        <div className="text-center p-8 font-mono text-xs text-muted-foreground">
          No categorical composition charts available.
        </div>
      )}
    </div>
  );
}

function ChartImagePanel({
  src,
  alt,
  title,
  narrative,
}: {
  src: string;
  alt: string;
  title: string;
  narrative: string;
}) {
  return (
    <div className="space-y-6">
      <div className="flex justify-center border border-border/60 bg-background rounded-2xl p-4 sm:p-5 max-w-3xl mx-auto shadow-2xs">
        <SafeChartImage base64Src={src} alt={alt} className="max-h-[340px] w-auto object-contain rounded-xl" />
      </div>
      <div className="bg-muted/20 border border-border/60 p-5 rounded-2xl max-w-3xl mx-auto space-y-2">
        <div className="text-[10px] font-mono font-bold uppercase tracking-wider text-primary">{title}</div>
        <p className="text-xs sm:text-sm text-foreground/80 leading-relaxed font-sans italic">{narrative}</p>
      </div>
    </div>
  );
}

function BivariateChartSection({
  boxplots,
  scatterPlot,
  activeBoxIndex,
  onSelectIndex,
}: {
  boxplots?: Charts["boxplots"];
  scatterPlot?: Charts["scatter_plot"];
  activeBoxIndex: number;
  onSelectIndex: (idx: number) => void;
}) {
  const selectedBox = boxplots?.[activeBoxIndex];
  const showScatter = activeBoxIndex === -1 && Boolean(scatterPlot);

  return (
    <div className="space-y-6">
      <div className="flex overflow-x-auto gap-1.5 justify-start sm:justify-center py-1 scrollbar-none snap-x whitespace-nowrap">
        {boxplots?.map((item, idx) => (
          <button
            key={`box-tab-${item.column}`}
            type="button"
            onClick={() => onSelectIndex(idx)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all shrink-0 ${
              activeBoxIndex === idx
                ? "bg-primary text-primary-foreground shadow-2xs"
                : "bg-muted/50 hover:bg-muted text-muted-foreground"
            }`}
          >
            {item.column}
          </button>
        ))}
        {scatterPlot && (
          <button
            type="button"
            onClick={() => onSelectIndex(-1)}
            className={`px-3 py-1 rounded-lg text-xs font-mono font-semibold transition-all ${
              activeBoxIndex === -1
                ? "bg-primary text-primary-foreground shadow-2xs"
                : "bg-muted/50 hover:bg-muted text-muted-foreground"
            }`}
          >
            {scatterPlot.columns} (Scatter)
          </button>
        )}
      </div>

      {showScatter && scatterPlot ? (
        <ChartImagePanel
          src={scatterPlot.image}
          alt="Bivariate Scatter Plot"
          title="Scatter Regression Narrative"
          narrative={scatterPlot.narrative}
        />
      ) : selectedBox ? (
        <ChartImagePanel
          src={selectedBox.image}
          alt={`Boxplot for ${selectedBox.column}`}
          title="Distribution Spread & Outliers"
          narrative={selectedBox.narrative}
        />
      ) : (
        <div className="text-center p-8 font-mono text-xs text-muted-foreground">
          No bivariate relations charts available.
        </div>
      )}
    </div>
  );
}

export function VisualChartsTab({
  charts,
  chartTabsList,
  hiplotPayload,
  onIsolateSelection,
}: {
  charts: Charts;
  chartTabsList: ChartTabOption[];
  hiplotPayload: HiPlotPayload | null;
  onIsolateSelection: (count: number) => void;
}) {
  const [activeChartTab, setActiveChartTab] = useState<string>("correlation");
  const [activeDistIndex, setActiveDistIndex] = useState<number>(0);
  const [activeBarIndex, setActiveBarIndex] = useState<number>(0);
  const [activeBoxIndex, setActiveBoxIndex] = useState<number>(0);

  if (chartTabsList.length === 0) {
    return (
      <div className="text-center py-12 font-mono text-xs text-muted-foreground">
        No visual charts available for this dataset.
      </div>
    );
  }

  return (
    <Card className="border border-border bg-card shadow-premium rounded-3xl overflow-hidden">
      <CardHeader className="border-b border-border bg-muted/10 pb-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <CardTitle className="text-lg font-display font-bold text-foreground flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Visual Insights Gallery
            </CardTitle>
            <CardDescription className="text-xs">
              Algorithmic visual explorations for bivariate correlations, distribution shapes, and multidimensional trends
            </CardDescription>
          </div>
        </div>

        <div className="flex overflow-x-auto gap-2 mt-4 border-b border-border/40 pb-2 scrollbar-none snap-x whitespace-nowrap">
          {chartTabsList.map((t) => {
            const Icon = t.icon;
            const isActive = activeChartTab === t.id;
            return (
              <button
                key={t.id}
                type="button"
                onClick={() => setActiveChartTab(t.id)}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold tracking-tight transition-all shrink-0 ${
                  isActive
                    ? "bg-primary text-primary-foreground shadow-2xs"
                    : "bg-secondary/40 hover:bg-secondary text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{t.label}</span>
              </button>
            );
          })}
        </div>
      </CardHeader>

      <CardContent className="p-4 sm:p-8">
        {activeChartTab === "correlation" && charts.correlation_heatmap && (
          <CorrelationChartSection correlation={charts.correlation_heatmap} />
        )}

        {activeChartTab === "distributions" && charts.distributions && charts.distributions.length > 0 && (
          <DistributionsChartSection
            distributions={charts.distributions}
            activeDistIndex={activeDistIndex}
            onSelectIndex={setActiveDistIndex}
          />
        )}

        {activeChartTab === "composition" && (
          <CompositionChartSection
            barCharts={charts.bar_charts}
            donutChart={charts.donut_chart}
            activeBarIndex={activeBarIndex}
            onSelectIndex={setActiveBarIndex}
          />
        )}

        {activeChartTab === "bivariate" && (
          <BivariateChartSection
            boxplots={charts.boxplots}
            scatterPlot={charts.scatter_plot}
            activeBoxIndex={activeBoxIndex}
            onSelectIndex={setActiveBoxIndex}
          />
        )}

        {activeChartTab === "hiplot" && hiplotPayload && (
          <div className="space-y-4">
            <div className="bg-muted/10 border border-border/80 rounded-2xl p-4">
              <HiPlotParallelCoordinates
                payload={hiplotPayload}
                onIsolateSelection={(selectedRows) => onIsolateSelection(selectedRows.length)}
              />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
