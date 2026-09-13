import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Sliders, RefreshCw, Filter, Eye, Layers } from 'lucide-react';

export interface HiPlotDimension {
  name: string;
  display_name: string;
  type: string;
  is_numeric: boolean;
  min?: number;
  max?: number;
  mean?: number;
  p5?: number;
  p95?: number;
  categories?: string[];
}

export interface HiPlotPayload {
  total_rows: number;
  sampled_rows: number;
  is_sampled: boolean;
  dimensions: HiPlotDimension[];
  datapoints: Record<string, any>[];
}

interface Props {
  payload: HiPlotPayload;
  onIsolateSelection?: (selectedRows: Record<string, any>[]) => void;
}

export const HiPlotParallelCoordinates: React.FC<Props> = ({ payload, onIsolateSelection }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const { dimensions = [], datapoints = [] } = payload;

  // Active brush ranges: { [dimName]: [minVal, maxVal] }
  const [brushes, setBrushes] = useState<Record<string, [number, number]>>({});
  const [colorColumn, setColorColumn] = useState<string>(
    () => dimensions.find(d => d.is_numeric)?.name || (dimensions[0]?.name ?? '')
  );

  // Dragging state for canvas brushing
  const [draggingDim, setDraggingDim] = useState<string | null>(null);
  const [dragStartY, setDragStartY] = useState<number | null>(null);
  const [dragCurrentY, setDragCurrentY] = useState<number | null>(null);

  // Available numeric columns for coloring
  const numericDims = useMemo(() => dimensions.filter(d => d.is_numeric), [dimensions]);

  // Determine active/filtered datapoints
  const filteredIndices = useMemo(() => {
    const activeBrushEntries = Object.entries(brushes);
    if (activeBrushEntries.length === 0) {
      return new Set(datapoints.map((_, i) => i));
    }

    const matched = new Set<number>();
    datapoints.forEach((row, idx) => {
      let satisfiesAll = true;
      for (const [dimName, [bMin, bMax]] of activeBrushEntries) {
        const val = Number(row[dimName]);
        if (isNaN(val) || val < bMin || val > bMax) {
          satisfiesAll = false;
          break;
        }
      }
      if (satisfiesAll) {
        matched.add(idx);
      }
    });
    return matched;
  }, [datapoints, brushes]);

  // Color generator based on normalized value [0, 1]
  const getColor = (ratio: number, alpha: number = 0.6) => {
    // Viridis / Teal-Emerald-Blue-Violet gradient
    const r = Math.round(16 + ratio * 200);
    const g = Math.round(185 - ratio * 60);
    const b = Math.round(129 + ratio * 120);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  // Render parallel coordinates on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.length < 2) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;
    const topMargin = 50;
    const bottomMargin = 40;
    const leftMargin = 60;
    const rightMargin = 60;
    const plotHeight = height - topMargin - bottomMargin;
    const plotWidth = width - leftMargin - rightMargin;

    ctx.clearRect(0, 0, width, height);

    // X position for each dimension axis
    const axisSpacing = plotWidth / (dimensions.length - 1);
    const axisPositions = dimensions.map((_, i) => leftMargin + i * axisSpacing);

    // Compute Y coordinate for a given value on a dimension
    const getY = (dim: HiPlotDimension, val: any): number => {
      if (val === null || val === undefined) return height - bottomMargin;
      if (dim.is_numeric) {
        const min = dim.min ?? 0;
        const max = dim.max ?? 1;
        const span = max - min || 1;
        const ratio = (Number(val) - min) / span;
        return height - bottomMargin - ratio * plotHeight;
      } else if (dim.categories && dim.categories.length > 0) {
        const idx = dim.categories.indexOf(String(val));
        const span = dim.categories.length - 1 || 1;
        const ratio = idx >= 0 ? idx / span : 0;
        return height - bottomMargin - ratio * plotHeight;
      }
      return height - bottomMargin;
    };

    // Color scaling dimension bounds
    const colorDim = dimensions.find(d => d.name === colorColumn);
    const cMin = colorDim?.min ?? 0;
    const cMax = colorDim?.max ?? 1;
    const cSpan = cMax - cMin || 1;

    // 1. Draw inactive lines first (low opacity background)
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.07)';
    datapoints.forEach((row, i) => {
      if (filteredIndices.has(i)) return;
      ctx.beginPath();
      dimensions.forEach((dim, dimIdx) => {
        const x = axisPositions[dimIdx];
        const y = getY(dim, row[dim.name]);
        if (dimIdx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // 2. Draw active (brushed) lines with vibrant color
    datapoints.forEach((row, i) => {
      if (!filteredIndices.has(i)) return;
      const cVal = Number(row[colorColumn]);
      const ratio = !isNaN(cVal) ? Math.max(0, Math.min(1, (cVal - cMin) / cSpan)) : 0.5;

      ctx.lineWidth = filteredIndices.size < 500 ? 1.5 : 1;
      ctx.strokeStyle = getColor(ratio, 0.55);
      ctx.beginPath();
      dimensions.forEach((dim, dimIdx) => {
        const x = axisPositions[dimIdx];
        const y = getY(dim, row[dim.name]);
        if (dimIdx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });

    // 3. Draw vertical axes, ticks, and labels
    dimensions.forEach((dim, dimIdx) => {
      const x = axisPositions[dimIdx];

      // Axis vertical spine
      ctx.strokeStyle = 'rgba(203, 213, 225, 0.3)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, topMargin);
      ctx.lineTo(x, height - bottomMargin);
      ctx.stroke();

      // Axis header label
      ctx.fillStyle = '#f8fafc';
      ctx.font = '600 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      const truncatedTitle = dim.display_name.length > 14
        ? dim.display_name.substring(0, 12) + '...'
        : dim.display_name;
      ctx.fillText(truncatedTitle, x, topMargin - 20);

      // Max and Min labels
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px Inter, sans-serif';
      if (dim.is_numeric) {
        ctx.fillText(`${dim.max ?? ''}`, x, topMargin - 6);
        ctx.fillText(`${dim.min ?? ''}`, x, height - bottomMargin + 16);
      } else if (dim.categories && dim.categories.length > 0) {
        ctx.fillText(`${dim.categories[dim.categories.length - 1]}`, x, topMargin - 6);
        ctx.fillText(`${dim.categories[0]}`, x, height - bottomMargin + 16);
      }

      // 4. Draw brush overlay if active on this axis
      const brush = brushes[dim.name];
      if (brush && dim.is_numeric) {
        const y1 = getY(dim, brush[1]);
        const y2 = getY(dim, brush[0]);
        const brushTop = Math.min(y1, y2);
        const brushHeight = Math.abs(y2 - y1);

        ctx.fillStyle = 'rgba(59, 130, 246, 0.25)';
        ctx.strokeStyle = '#3b82f6';
        ctx.lineWidth = 2;
        ctx.fillRect(x - 8, brushTop, 16, brushHeight);
        ctx.strokeRect(x - 8, brushTop, 16, brushHeight);
      }
    });

  }, [dimensions, datapoints, brushes, colorColumn, filteredIndices]);

  // Handle axis drag brush selection
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const width = rect.width;
    const leftMargin = 60;
    const rightMargin = 60;
    const axisSpacing = (width - leftMargin - rightMargin) / (dimensions.length - 1);

    // Find closest axis within 20px
    let closestDim: HiPlotDimension | null = null;
    dimensions.forEach((d, i) => {
      const axisX = leftMargin + i * axisSpacing;
      if (Math.abs(x - axisX) <= 24 && d.is_numeric) {
        closestDim = d;
      }
    });

    if (closestDim) {
      setDraggingDim((closestDim as HiPlotDimension).name);
      setDragStartY(y);
      setDragCurrentY(y);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggingDim) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const y = e.clientY - rect.top;
    setDragCurrentY(y);
  };

  const handleMouseUp = () => {
    if (!draggingDim || dragStartY === null || dragCurrentY === null) {
      setDraggingDim(null);
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const height = rect.height;
    const topMargin = 50;
    const bottomMargin = 40;
    const plotHeight = height - topMargin - bottomMargin;

    const dim = dimensions.find(d => d.name === draggingDim);
    if (dim && dim.is_numeric) {
      const min = dim.min ?? 0;
      const max = dim.max ?? 1;
      const span = max - min || 1;

      const y1 = Math.min(dragStartY, dragCurrentY);
      const y2 = Math.max(dragStartY, dragCurrentY);

      if (Math.abs(y2 - y1) > 8) {
        const valMax = max - ((y1 - topMargin) / plotHeight) * span;
        const valMin = max - ((y2 - topMargin) / plotHeight) * span;

        setBrushes(prev => ({
          ...prev,
          [dim.name]: [Math.max(min, valMin), Math.min(max, valMax)]
        }));
      }
    }

    setDraggingDim(null);
    setDragStartY(null);
    setDragCurrentY(null);
  };

  const resetAllBrushes = () => {
    setBrushes({});
  };

  const handleIsolateClick = () => {
    if (onIsolateSelection) {
      const selected = datapoints.filter((_, i) => filteredIndices.has(i));
      onIsolateSelection(selected);
    }
  };

  const percentBrushed = Math.round((filteredIndices.size / Math.max(datapoints.length, 1)) * 100);

  return (
    <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-2xl space-y-4">
      {/* Top Header & HUD Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded-lg text-blue-400">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white tracking-wide flex items-center gap-2">
              HiPlot Multidimensional Explorer
              <span className="text-xs font-normal text-slate-400 bg-slate-800 px-2 py-0.5 rounded-full border border-slate-700">
                Parallel Coordinates
              </span>
            </h3>
            <p className="text-xs text-slate-400">
              Drag on any vertical axis to brush and isolate multidimensional sub-populations in real-time.
            </p>
          </div>
        </div>

        {/* HUD Statistics & Actions */}
        <div className="flex items-center gap-3 text-xs">
          <div className="bg-slate-800/80 border border-slate-700/80 px-3 py-1.5 rounded-lg text-slate-300 flex items-center gap-2">
            <Eye className="w-3.5 h-3.5 text-emerald-400" />
            <span>
              Brushed: <strong className="text-white font-mono">{filteredIndices.size}</strong> / {datapoints.length} ({percentBrushed}%)
            </span>
          </div>

          {/* Color By Selector */}
          {numericDims.length > 0 && (
            <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700/80 px-2.5 py-1 rounded-lg">
              <Layers className="w-3.5 h-3.5 text-blue-400" />
              <span className="text-slate-400">Color:</span>
              <select
                id="hiplot-color-column-select"
                aria-label="Color lines by dimension"
                value={colorColumn}
                onChange={e => setColorColumn(e.target.value)}
                className="bg-transparent text-slate-200 border-none outline-none cursor-pointer text-xs"
              >
                {numericDims.map(d => (
                  <option key={d.name} value={d.name} className="bg-slate-900 text-white">
                    {d.display_name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Reset Filters */}
          {Object.keys(brushes).length > 0 && (
            <button
              onClick={resetAllBrushes}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg border border-slate-700 transition"
            >
              <RefreshCw className="w-3 h-3 text-amber-400" />
              Reset Brushes
            </button>
          )}

          {/* Isolate Population Button */}
          {onIsolateSelection && (
            <button
              onClick={handleIsolateClick}
              disabled={filteredIndices.size === 0}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium rounded-lg shadow-lg shadow-blue-500/20 transition"
            >
              <Filter className="w-3 h-3" />
              Isolate Selection
            </button>
          )}
        </div>
      </div>

      {/* Parallel Coordinates Canvas */}
      <div ref={containerRef} className="w-full relative h-[420px] bg-slate-950/50 rounded-lg overflow-hidden border border-slate-800/50">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          className="w-full h-full cursor-crosshair"
        />
      </div>
    </div>
  );
};
