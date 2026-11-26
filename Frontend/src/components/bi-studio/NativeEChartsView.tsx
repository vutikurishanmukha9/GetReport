import React, { useEffect, useRef, useState } from "react";
import * as echarts from "echarts";
import { useTheme } from "@/hooks/useTheme";

interface EChartsStyleRecord {
  color?: string;
  fontSize?: number;
  fontFamily?: string;
  fontWeight?: number;
  width?: number;
  type?: string;
  borderRadius?: number;
  borderColor?: string;
  borderWidth?: number;
  textBorderColor?: string;
  textBorderWidth?: number;
}

interface EChartsAxisDraft {
  type?: string;
  name?: string;
  axisLabel?: EChartsStyleRecord;
  nameTextStyle?: EChartsStyleRecord;
  axisLine?: { lineStyle?: EChartsStyleRecord };
  splitLine?: { show?: boolean; lineStyle?: EChartsStyleRecord };
}

interface EChartsPieDatum {
  name?: string;
  value?: number | string | null;
}

interface EChartsSeriesDraft {
  type?: string;
  label?: EChartsStyleRecord;
  data?: Array<EChartsPieDatum | number | string | null>;
  radius?: string | string[];
  avoidLabelOverlap?: boolean;
  labelLine?: { show?: boolean; length?: number; length2?: number; smooth?: number; lineStyle?: EChartsStyleRecord };
  itemStyle?: EChartsStyleRecord;
}

interface EChartsDataZoomDraft {
  textStyle?: EChartsStyleRecord;
  borderColor?: string;
  handleStyle?: EChartsStyleRecord;
}

interface EChartsTitleDraft {
  textStyle?: EChartsStyleRecord;
}

interface EChartsTooltipDraft {
  confine?: boolean;
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  padding?: number[];
  extraCssText?: string;
  textStyle?: EChartsStyleRecord;
}

interface EChartsLegendDraft {
  show?: boolean;
  orient?: string;
  left?: string;
  right?: string;
  top?: string | number;
  bottom?: string | number;
  type?: string;
  pageButtonPosition?: string;
  pageIconColor?: string;
  pageIconInactiveColor?: string;
  pageIconSize?: number;
  pageTextStyle?: EChartsStyleRecord;
  textStyle?: EChartsStyleRecord;
}

interface EChartsGridDraft {
  top?: string | number;
}

interface EChartsOptionDraft {
  xAxis?: EChartsAxisDraft | EChartsAxisDraft[];
  yAxis?: EChartsAxisDraft | EChartsAxisDraft[];
  tooltip?: EChartsTooltipDraft;
  legend?: EChartsLegendDraft;
  grid?: EChartsGridDraft;
  series?: EChartsSeriesDraft[];
  title?: EChartsTitleDraft;
  dataZoom?: EChartsDataZoomDraft[];
}

export interface ChartClickPayload {
  name?: string | number | boolean | null | undefined;
}

interface NativeEChartsViewProps {
  option: echarts.EChartsCoreOption;
  onChartClick?: (params: ChartClickPayload) => void;
  chartRef?: { current: echarts.ECharts | null };
  height?: string | number;
  className?: string;
}

function sanitizeEChartsOption(opt: echarts.EChartsCoreOption, isDark: boolean = true): echarts.EChartsCoreOption {
  if (!(opt instanceof Object)) return opt;
  // SAFETY: EChartsCoreOption passed from backend explore responses is a JSON-serializable option object.
  const clone = structuredClone(opt) as EChartsOptionDraft;

  const primaryTextColor = isDark ? "#f4f4f5" : "#18181b";
  const secondaryTextColor = isDark ? "#a1a1aa" : "#44403c";
  const axisLabelColor = isDark ? "#d4d4d8" : "#292524";
  const axisLineColor = isDark ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.15)";
  const splitLineColor = isDark ? "rgba(255, 255, 255, 0.06)" : "rgba(0, 0, 0, 0.08)";
  const labelHaloColor = isDark ? "rgba(9, 13, 22, 0.95)" : "rgba(255, 255, 255, 0.95)";

  const sanitizeAxis = (axis: EChartsAxisDraft): EChartsAxisDraft => {
    if (!(axis instanceof Object)) return axis;
    const clean: EChartsAxisDraft = { ...axis };
    if (clean.type === "linear") clean.type = "value";
    clean.axisLabel = { fontSize: 11, fontFamily: "inherit", fontWeight: 500, ...clean.axisLabel, color: axisLabelColor };
    if (clean.name) {
      clean.nameTextStyle = { fontSize: 11, fontFamily: "inherit", fontWeight: 600, ...clean.nameTextStyle, color: primaryTextColor };
    }
    clean.axisLine = { ...clean.axisLine, lineStyle: { ...clean.axisLine?.lineStyle, color: axisLineColor } };
    if (clean.splitLine && clean.splitLine.show !== false) {
      clean.splitLine = { ...clean.splitLine, lineStyle: { type: "dashed", ...clean.splitLine?.lineStyle, color: splitLineColor } };
    }
    return clean;
  };

  if (clone.yAxis) {
    clone.yAxis = Array.isArray(clone.yAxis) ? clone.yAxis.map(sanitizeAxis) : sanitizeAxis(clone.yAxis);
  }
  if (clone.xAxis) {
    clone.xAxis = Array.isArray(clone.xAxis) ? clone.xAxis.map(sanitizeAxis) : sanitizeAxis(clone.xAxis);
  }

  const origTooltip = clone.tooltip || {};
  clone.tooltip = {
    confine: true,
    ...origTooltip,
    backgroundColor: isDark ? "#18181b" : "#ffffff",
    borderColor: isDark ? "rgba(255, 255, 255, 0.15)" : "rgba(0, 0, 0, 0.12)",
    borderWidth: 1,
    padding: [8, 12],
    textStyle: { fontSize: 12, fontWeight: 500, fontFamily: "inherit", ...origTooltip.textStyle, color: isDark ? "#f4f4f5" : "#18181b" },
    extraCssText: isDark
      ? "box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5); border-radius: 8px; backdrop-filter: blur(8px);"
      : "box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.12); border-radius: 8px;",
  };

  if (clone.legend && clone.legend.show !== false) {
    const origLegend = clone.legend;
    clone.legend = {
      ...origLegend,
      orient: "horizontal",
      left: "center",
      right: "4%",
      top: "0%",
      type: "scroll",
      pageButtonPosition: "end",
      pageIconColor: isDark ? "#38bdf8" : "#800000",
      pageIconInactiveColor: isDark ? "#52525b" : "#a8a29e",
      pageIconSize: 11,
      pageTextStyle: { fontSize: 10, fontWeight: 600, ...origLegend.pageTextStyle, color: secondaryTextColor },
      textStyle: { fontSize: 11, fontWeight: 500, fontFamily: "inherit", ...origLegend.textStyle, color: primaryTextColor },
    };
    delete clone.legend.bottom;
    if (clone.grid) {
      const gridTop = clone.grid.top;
      const numericTop =
        gridTop !== null && gridTop !== undefined && gridTop.constructor === Number && Number.isFinite(gridTop)
          ? Number(gridTop)
          : parseInt(String(gridTop ?? "8"), 10);
      clone.grid = { ...clone.grid, top: Math.max(numericTop, 45) };
    }
  }

  if (Array.isArray(clone.series)) {
    const datumValue = (item: EChartsPieDatum | number | string | null): number =>
      item instanceof Object ? Number(item.value) || 0 : Number(item) || 0;

    clone.series = clone.series.map((s: EChartsSeriesDraft) => {
      if (!(s instanceof Object)) return s;
      const cleanSeries: EChartsSeriesDraft = { ...s };
      if (cleanSeries.label) {
        cleanSeries.label = {
          fontSize: 11.5,
          fontFamily: "inherit",
          fontWeight: 600,
          ...cleanSeries.label,
          color: primaryTextColor,
          textBorderColor: labelHaloColor,
          textBorderWidth: 3,
        };
      }
      if (cleanSeries.type === "pie") {
        if (Array.isArray(cleanSeries.data) && cleanSeries.data.length > 7) {
          const sorted = [...cleanSeries.data].sort((a, b) => datumValue(b) - datumValue(a));
          const restVal = sorted.slice(6).reduce<number>((acc, cur) => acc + datumValue(cur), 0);
          cleanSeries.data = [...sorted.slice(0, 6), { name: `Other (${sorted.length - 6} items)`, value: Math.round(restVal * 100) / 100 }];
        }
        cleanSeries.radius = ["50%", "72%"];
        cleanSeries.avoidLabelOverlap = true;
        cleanSeries.labelLine = {
          show: true,
          length: 15,
          length2: 12,
          smooth: 0.2,
          ...cleanSeries.labelLine,
          lineStyle: { width: 1.2, ...cleanSeries.labelLine?.lineStyle, color: isDark ? "rgba(255, 255, 255, 0.45)" : "rgba(15, 23, 42, 0.45)" },
        };
        cleanSeries.itemStyle = { ...cleanSeries.itemStyle, borderRadius: 3, borderColor: isDark ? "#07080a" : "#ffffff", borderWidth: 2 };
      }
      if (cleanSeries.type === "bar") {
        cleanSeries.itemStyle = { ...cleanSeries.itemStyle, borderColor: "transparent", borderWidth: 0 };
      }
      return cleanSeries;
    });
  }

  if (clone.title) {
    clone.title = {
      ...clone.title,
      textStyle: { fontSize: 13, fontWeight: 600, ...clone.title.textStyle, color: primaryTextColor },
    };
  }

  if (Array.isArray(clone.dataZoom)) {
    clone.dataZoom = clone.dataZoom.map((dz: EChartsDataZoomDraft) => ({
      ...dz,
      textStyle: { fontSize: 10, fontWeight: 500, ...dz.textStyle, color: secondaryTextColor },
      borderColor: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(0, 0, 0, 0.12)",
      handleStyle: { color: isDark ? "#38bdf8" : "#800000", borderColor: isDark ? "#0284c7" : "#550000", ...dz.handleStyle },
    }));
  }

  // SAFETY: clone matches EChartsCoreOption structure with hardened contrast and theme tokens.
  return clone as echarts.EChartsCoreOption;
}

export const NativeEChartsView: React.FC<NativeEChartsViewProps> = ({
  option,
  onChartClick,
  chartRef,
  height = "480px",
  className = "",
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<echarts.ECharts | null>(null);
  const onChartClickRef = useRef(onChartClick);
  const externalChartRef = useRef(chartRef);
  const { theme } = useTheme();

  const [effectiveDark, setEffectiveDark] = useState<boolean>(() => {
    if ("document" in globalThis) {
      return document.documentElement.classList.contains("dark");
    }
    return theme === "dark";
  });

  useEffect(() => {
    const updateThemeState = () => {
      setEffectiveDark("document" in globalThis ? document.documentElement.classList.contains("dark") : theme === "dark");
    };
    updateThemeState();
    const observer = new MutationObserver((mutations) => {
      for (const mut of mutations) {
        if (mut.attributeName === "class") updateThemeState();
      }
    });
    if ("document" in globalThis) {
      observer.observe(document.documentElement, { attributes: true });
    }
    return () => observer.disconnect();
  }, [theme]);

  useEffect(() => {
    onChartClickRef.current = onChartClick;
    externalChartRef.current = chartRef;
  });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return () => {};

    let instance = echarts.getInstanceByDom(container);
    if (!instance) {
      instance = echarts.init(container, undefined, { renderer: "canvas" });
      chartInstanceRef.current = instance;
    }
    if (externalChartRef.current) externalChartRef.current.current = instance;

    const clickHandler = (params: ChartClickPayload) => onChartClickRef.current?.(params);
    instance.off("click");
    instance.on("click", clickHandler);

    const resizeChart = () => {
      if (!instance || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) instance.resize({ width: rect.width, height: rect.height });
      else instance.resize();
    };

    let timer1: ReturnType<typeof setTimeout> | undefined;
    let timer2: ReturnType<typeof setTimeout> | undefined;

    if (option && Object.keys(option).length > 0) {
      const cleanOption = sanitizeEChartsOption(option, effectiveDark);
      try {
        instance.clear();
        instance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
        resizeChart();
        timer1 = setTimeout(resizeChart, 60);
        timer2 = setTimeout(resizeChart, 250);
      } catch (err) {
        console.error("ECharts setOption error, recreating instance:", err);
        try {
          instance.dispose();
          instance = echarts.init(container, undefined, { renderer: "canvas" });
          chartInstanceRef.current = instance;
          if (externalChartRef.current) externalChartRef.current.current = instance;
          instance.on("click", clickHandler);
          instance.setOption(cleanOption, { notMerge: true, lazyUpdate: false });
          instance.resize();
        } catch (recoverErr) {
          console.error("Failed to recover ECharts instance:", recoverErr);
        }
      }
    }

    const resizeObserver = new ResizeObserver(() => resizeChart());
    resizeObserver.observe(container);
    window.addEventListener("resize", resizeChart);

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      resizeObserver.disconnect();
      window.removeEventListener("resize", resizeChart);
      instance.off("click", clickHandler);
    };
  }, [option, effectiveDark]);

  useEffect(() => {
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
      if (externalChartRef.current) {
        externalChartRef.current.current = null;
      }
    };
  }, []);

  return <div ref={containerRef} style={{ width: "100%", height, minHeight: "420px" }} className={`w-full relative ${className}`} />;
};
