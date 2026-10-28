"""
Apache Superset Native Query & Visual Analytics Engine.

Directly adapted from Apache Superset codebase (superset/superset/dataframe.py,
superset/superset/explore/, and superset/superset/utils/core.py).
Executes high-speed in-process analytical aggregations, metric calculations,
and ECharts data series transformations directly on GetReport's uploaded datasets.
"""

from enum import IntEnum
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─── Superset Generic Data Types (from @superset-ui/core/src/types.ts) ────────
class GenericDataType(IntEnum):
    STRING = 0
    NUMERIC = 1
    TEMPORAL = 2
    BOOLEAN = 3


def map_polars_to_superset_dtype(dtype: pl.DataType) -> GenericDataType:
    """Map Polars DataType to Superset GenericDataType."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64, pl.Decimal):
        return GenericDataType.NUMERIC
    elif dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
        return GenericDataType.TEMPORAL
    elif dtype == pl.Boolean:
        return GenericDataType.BOOLEAN
    else:
        return GenericDataType.STRING


# ─── Superset Categorical Palettes (from @superset-ui/core colorSchemes) ───────
SUPERSET_COLOR_PALETTES = {
    "supersetColors": [
        "#1FA8C9", "#454E7C", "#5AC189", "#FF7F44", "#666666",
        "#E04355", "#FCC700", "#A868B7", "#3CCCCB", "#A38F79",
        "#8FD3E4", "#A1A6BD", "#ACE1C4", "#FEC0A1", "#B2B2B2",
    ],
    "airbnb": [
        "#FF5A5F", "#00A699", "#7B0051", "#007A87", "#FFAA91",
        "#B4E9E2", "#D47292", "#61A9AE", "#FCBCA0", "#00D1C1",
    ],
    "d3Category10": [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ],
    "modernDark": [
        "#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6",
        "#a78bfa", "#4ade80", "#f87171", "#2dd4bf", "#c084fc",
    ],
    "emerald": [
        "#10b981", "#059669", "#34d399", "#6ee7b7", "#047857",
        "#065f46", "#a7f3d0", "#022c22", "#3b82f6", "#6366f1",
    ]
}


# ─── Safe Type Conversions (adapted from superset/dataframe.py) ───────────────
def _sanitize_for_json(val: Any) -> Any:
    """Ensure all numerical and missing values serialize cleanly to JSON."""
    if val is None:
        return None
    if isinstance(val, (float, int)):
        if math.isnan(val) or math.isinf(val):
            return None
        if isinstance(val, float):
            return round(val, 2)
        return val
    return str(val)


# ─── Superset Explore Query Request ──────────────────────────────────────────
class SupersetMetricConfig(BaseModel):
    label: str = Field(..., description="Display label for the metric")
    column: Optional[str] = Field(None, description="Dataset column name")
    aggregate: str = Field("COUNT", description="COUNT, SUM, AVG, MIN, MAX, MEDIAN, COUNT_DISTINCT")


class SupersetExploreQuery(BaseModel):
    task_id: str = Field(..., description="GetReport task ID identifying the uploaded dataset")
    chart_type: str = Field("echarts_bar", description="echarts_bar, echarts_line, echarts_area, echarts_pie, echarts_scatter, echarts_treemap, table")
    dimension: str = Field(..., description="Primary categorical/date dimension for X-axis / GroupBy")
    secondary_dimension: Optional[str] = Field(None, description="Optional secondary dimension for stacked series or breakdown")
    metric: Optional[str] = Field(None, description="Numerical metric column to aggregate")
    aggregate: str = Field("COUNT", description="Aggregation function: COUNT, SUM, AVG, MIN, MAX, MEDIAN, COUNT_DISTINCT")
    row_limit: int = Field(15, ge=1, le=500, description="Top N limit")
    sort_order: str = Field("desc", description="desc or asc")
    color_palette: str = Field("supersetColors", description="Color palette key from SUPERSET_COLOR_PALETTES")


class SupersetEngine:
    """
    Executes in-process visual analytics on GetReport's uploaded datasets
    using Apache Superset's Explore architecture.
    """

    def resolve_dataset_df(self, task_id: str) -> pl.DataFrame:
        """Resolve and load the audited Polars DataFrame for a task ID."""
        from app.services.task_manager import title_task_manager
        from app.services.storage import get_storage_provider

        job = title_task_manager.get_job(task_id)
        if not job or not job.result:
            raise ValueError(f"Task '{task_id}' not found or results missing.")

        file_ref = job.result.get("cleaned_file_ref") or job.result.get("_file_ref")
        if not file_ref:
            raise ValueError(f"No source file associated with task '{task_id}'.")

        storage = get_storage_provider()
        file_path = storage.get_absolute_path(file_ref)
        if not os.path.exists(file_path):
            raise ValueError(f"Dataset file does not exist on disk: {file_path}")

        if file_path.endswith(".parquet"):
            return pl.read_parquet(file_path)
        return pl.read_csv(file_path)

    def introspect_dataset(self, task_id: str) -> Dict[str, Any]:
        """
        Returns Superset Explore column definitions, types, and summary metrics.
        """
        df = self.resolve_dataset_df(task_id)
        columns_meta = []
        numeric_cols = []
        categorical_cols = []
        temporal_cols = []

        for col, dtype in zip(df.columns, df.dtypes):
            sup_type = map_polars_to_superset_dtype(dtype)
            is_num = sup_type == GenericDataType.NUMERIC
            is_temp = sup_type == GenericDataType.TEMPORAL

            col_meta = {
                "name": col,
                "type": sup_type,
                "type_name": sup_type.name,
                "is_numeric": is_num,
                "is_temporal": is_temp,
                "null_count": int(df[col].is_null().sum()),
                "unique_values": int(df[col].n_unique()),
            }
            columns_meta.append(col_meta)

            if is_num:
                numeric_cols.append(col)
            elif is_temp:
                temporal_cols.append(col)
            else:
                categorical_cols.append(col)

        return {
            "task_id": task_id,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": columns_meta,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "temporal_columns": temporal_cols,
            "color_palettes": list(SUPERSET_COLOR_PALETTES.keys()),
        }

    def execute_explore_query(self, query: SupersetExploreQuery) -> Dict[str, Any]:
        """
        Executes a Superset Explore query against the dataset and builds
        Apache ECharts configuration payload matching Superset's plugin-chart-echarts.
        """
        df = self.resolve_dataset_df(query.task_id)

        if query.dimension not in df.columns:
            raise ValueError(f"Dimension column '{query.dimension}' does not exist in dataset.")

        # Metric aggregation expression
        agg_op = query.aggregate.upper()
        metric_col = query.metric

        if metric_col and metric_col in df.columns:
            if agg_op == "SUM":
                val_expr = pl.col(metric_col).sum().alias("value")
                metric_label = f"SUM({metric_col})"
            elif agg_op in ("AVG", "MEAN"):
                val_expr = pl.col(metric_col).mean().alias("value")
                metric_label = f"AVG({metric_col})"
            elif agg_op == "MIN":
                val_expr = pl.col(metric_col).min().alias("value")
                metric_label = f"MIN({metric_col})"
            elif agg_op == "MAX":
                val_expr = pl.col(metric_col).max().alias("value")
                metric_label = f"MAX({metric_col})"
            elif agg_op == "MEDIAN":
                val_expr = pl.col(metric_col).median().alias("value")
                metric_label = f"MEDIAN({metric_col})"
            elif agg_op == "COUNT_DISTINCT":
                val_expr = pl.col(metric_col).n_unique().alias("value")
                metric_label = f"COUNT_DISTINCT({metric_col})"
            else:
                val_expr = pl.col(metric_col).count().alias("value")
                metric_label = f"COUNT({metric_col})"
        else:
            val_expr = pl.len().alias("value")
            metric_label = "Record Count"
            metric_col = "records"

        # Grouping columns
        group_cols = [query.dimension]
        has_secondary = bool(query.secondary_dimension and query.secondary_dimension in df.columns)
        if has_secondary and query.secondary_dimension != query.dimension:
            group_cols.append(query.secondary_dimension)

        # Execute Polars Aggregation
        filtered = df.filter(pl.col(query.dimension).is_not_null())
        if has_secondary:
            filtered = filtered.filter(pl.col(query.secondary_dimension).is_not_null())

        grouped = filtered.group_by(group_cols).agg(
            val_expr,
            pl.len().alias("count")
        )

        descending = query.sort_order.lower() == "desc"
        sorted_df = grouped.sort("value", descending=descending).head(query.row_limit)

        # Convert to records using Superset sanitization
        raw_records = sorted_df.to_dicts()
        cleaned_records = []
        for r in raw_records:
            cleaned_row = {k: _sanitize_for_json(v) for k, v in r.items()}
            cleaned_records.append(cleaned_row)

        # Calculate high-level KPIs for BigNumber card
        total_rows = len(df)
        if metric_col != "records" and metric_col in df.columns:
            m_col = df[metric_col]
            kpis = {
                "metric_label": metric_label,
                "dimension_label": query.dimension,
                "total_rows": total_rows,
                "sum": _sanitize_for_json(float(m_col.sum())) if m_col.dtype.is_numeric() else None,
                "mean": _sanitize_for_json(round(float(m_col.mean()), 2)) if m_col.dtype.is_numeric() else None,
                "max": _sanitize_for_json(float(m_col.max())) if m_col.dtype.is_numeric() else None,
                "min": _sanitize_for_json(float(m_col.min())) if m_col.dtype.is_numeric() else None,
                "unique_categories": df[query.dimension].n_unique(),
            }

        else:
            kpis = {
                "metric_label": "Record Count",
                "dimension_label": query.dimension,
                "total_rows": total_rows,
                "sum": total_rows,
                "mean": None,
                "max": None,
                "min": None,
                "unique_categories": df[query.dimension].n_unique(),
            }

        palette = SUPERSET_COLOR_PALETTES.get(query.color_palette, SUPERSET_COLOR_PALETTES["supersetColors"])

        # Build ECharts options payload matching Superset plugins
        echarts_option = self._build_echarts_options(
            chart_type=query.chart_type,
            records=cleaned_records,
            dimension=query.dimension,
            secondary_dimension=query.secondary_dimension if has_secondary else None,
            metric_label=metric_label,
            palette=palette,
        )

        return {
            "task_id": query.task_id,
            "chart_type": query.chart_type,
            "dimension": query.dimension,
            "secondary_dimension": query.secondary_dimension if has_secondary else None,
            "metric": query.metric,
            "aggregate": query.aggregate,
            "metric_label": metric_label,
            "row_limit": query.row_limit,
            "kpis": kpis,
            "records": cleaned_records,
            "total_groups": len(grouped),
            "echarts_option": echarts_option,
        }

    def _build_echarts_options(
        self,
        chart_type: str,
        records: List[Dict[str, Any]],
        dimension: str,
        secondary_dimension: Optional[str],
        metric_label: str,
        palette: List[str],
    ) -> Dict[str, Any]:
        """
        Builds Superset-standard ECharts option dictionary for direct rendering.
        """
        if not records:
            return {"title": {"text": "No data available", "left": "center", "textStyle": {"color": "#94a3b8"}}}

        # ── 1. Pie / Donut Chart ──────────────────────────────────────────
        if chart_type in ("echarts_pie", "pie"):
            pie_data = [{"name": str(r[dimension]), "value": r.get("value", 0)} for r in records]
            return {
                "backgroundColor": "transparent",
                "color": palette,
                "tooltip": {
                    "trigger": "item",
                    "formatter": "{b}: <b>{c}</b> ({d}%)",
                    "backgroundColor": "#181f30",
                    "borderColor": "#334155",
                    "textStyle": {"color": "#f8fafc", "fontSize": 12},
                },
                "legend": {
                    "orient": "horizontal",
                    "bottom": 0,
                    "textStyle": {"color": "#94a3b8", "fontSize": 11},
                },
                "series": [
                    {
                        "name": metric_label,
                        "type": "pie",
                        "radius": ["45%", "72%"],
                        "avoidLabelOverlap": True,
                        "itemStyle": {
                            "borderRadius": 6,
                            "borderColor": "#090d16",
                            "borderWidth": 2,
                        },
                        "label": {
                            "show": True,
                            "formatter": "{b}: {d}%",
                            "color": "#cbd5e1",
                            "fontSize": 11,
                        },
                        "emphasis": {
                            "label": {"show": True, "fontSize": 13, "fontWeight": "bold"},
                        },
                        "data": pie_data,
                    }
                ],
            }

        # ── 2. Treemap Chart ──────────────────────────────────────────────
        elif chart_type in ("echarts_treemap", "treemap"):
            tree_data = [{"name": str(r[dimension]), "value": r.get("value", 0)} for r in records]
            return {
                "backgroundColor": "transparent",
                "color": palette,
                "tooltip": {
                    "formatter": "{b}: <b>{c}</b>",
                    "backgroundColor": "#181f30",
                    "borderColor": "#334155",
                    "textStyle": {"color": "#f8fafc"},
                },
                "series": [
                    {
                        "type": "treemap",
                        "data": tree_data,
                        "roam": False,
                        "label": {"show": True, "formatter": "{b}\n{c}", "color": "#ffffff"},
                        "itemStyle": {"borderColor": "#090d16", "borderWidth": 2, "gapWidth": 2},
                    }
                ],
            }

        # ── 3. Categorical Bar / Column Chart (With Breakdown support) ─────
        categories = list(dict.fromkeys(str(r[dimension]) for r in records))

        if secondary_dimension:
            # Multi-series grouped/stacked bar
            sec_values = list(dict.fromkeys(str(r.get(secondary_dimension, "N/A")) for r in records))
            lookup = {(str(r[dimension]), str(r.get(secondary_dimension, "N/A"))): r.get("value", 0) for r in records}
            series_list = []
            for idx, sec in enumerate(sec_values):
                s_data = [lookup.get((cat, sec), 0) for cat in categories]
                series_list.append({
                    "name": sec,
                    "type": "bar",
                    "stack": "total",
                    "data": s_data,
                    "emphasis": {"focus": "series"},
                    "itemStyle": {"borderRadius": [2, 2, 0, 0]},
                })
            legend = {"show": True, "bottom": 0, "textStyle": {"color": "#94a3b8", "fontSize": 11}}
        else:
            s_data = [r.get("value", 0) for r in records]
            chart_kind = "line" if chart_type in ("echarts_line", "line", "echarts_area", "area") else "bar"
            series_item: Dict[str, Any] = {
                "name": metric_label,
                "type": chart_kind,
                "data": s_data,
                "itemStyle": {
                    "color": palette[0],
                    "borderRadius": [4, 4, 0, 0] if chart_kind == "bar" else 0,
                },
            }
            if chart_type in ("echarts_area", "area"):
                series_item["areaStyle"] = {"opacity": 0.25, "color": palette[0]}
                series_item["smooth"] = True
            elif chart_kind == "line":
                series_item["smooth"] = True

            series_list = [series_item]
            legend = {"show": False}

        return {
            "backgroundColor": "transparent",
            "color": palette,
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {"type": "shadow" if chart_type == "echarts_bar" else "line"},
                "backgroundColor": "#181f30",
                "borderColor": "#334155",
                "textStyle": {"color": "#f8fafc", "fontSize": 12},
            },
            "legend": legend,
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "14%" if secondary_dimension or len(categories) > 8 else "8%",
                "top": "8%",
                "containLabel": True,
            },
            "xAxis": {
                "type": "category",
                "data": categories,
                "axisLabel": {
                    "color": "#94a3b8",
                    "fontSize": 11,
                    "rotate": 30 if len(categories) > 6 else 0,
                    "interval": 0,
                },
                "axisLine": {"lineStyle": {"color": "#334155"}},
            },
            "yAxis": {
                "type": "value",
                "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                "splitLine": {"lineStyle": {"color": "#1e293b", "type": "dashed"}},
            },
            "series": series_list,
        }


superset_engine = SupersetEngine()
