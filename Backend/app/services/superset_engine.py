"""
Apache Superset Native Query & Visual Analytics Engine.

Directly adapted from Apache Superset codebase (superset/superset/dataframe.py,
superset/superset/explore/, and superset/superset/utils/core.py).
Executes high-speed in-process analytical aggregations, metric calculations,
temporal bucketing, ad-hoc multi-dimensional filters, and ECharts data series
transformations directly on GetReport's uploaded datasets.
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
    if dtype in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64, pl.Decimal
    ):
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


# ─── Filter Model ─────────────────────────────────────────────────────────────
class ExploreFilter(BaseModel):
    column: str = Field(..., description="Dataset column name to filter on")
    operator: str = Field(..., description="Comparison: eq, neq, gt, gte, lt, lte, in, contains, is_null, is_not_null")
    value: Optional[Any] = Field(None, description="Comparison value or list of values")


# ─── Superset Explore Query Request ──────────────────────────────────────────
class SupersetMetricConfig(BaseModel):
    label: str = Field(..., description="Display label for the metric")
    column: Optional[str] = Field(None, description="Dataset column name")
    aggregate: str = Field("COUNT", description="COUNT, SUM, AVG, MIN, MAX, MEDIAN, COUNT_DISTINCT")


class SupersetExploreQuery(BaseModel):
    task_id: str = Field(..., description="GetReport task ID identifying the uploaded dataset")
    chart_type: str = Field(
        "echarts_bar",
        description="echarts_bar, echarts_line, echarts_area, echarts_pie, echarts_scatter, echarts_heatmap, echarts_boxplot, echarts_treemap, table"
    )
    dimension: str = Field(..., description="Primary categorical/date dimension for X-axis / GroupBy")
    secondary_dimension: Optional[str] = Field(None, description="Optional secondary dimension for stacked series or breakdown")
    metric: Optional[str] = Field(None, description="Numerical metric column to aggregate")
    aggregate: str = Field("COUNT", description="Aggregation function: COUNT, SUM, AVG, MIN, MAX, MEDIAN, COUNT_DISTINCT")
    time_grain: Optional[str] = Field(None, description="Temporal bucketing: day, week, month, quarter, year")
    filters: Optional[List[ExploreFilter]] = Field(None, description="Optional list of ad-hoc filters to apply prior to aggregation")
    row_limit: int = Field(25, ge=1, le=1000, description="Top N limit")
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

            # Detect potential string dates
            if sup_type == GenericDataType.STRING:
                col_sample = df[col].drop_nulls().head(10).to_list()
                if col_sample and any(isinstance(x, str) and re.match(r"^\d{4}[-/]\d{2}[-/]\d{2}", str(x).strip()) for x in col_sample):
                    is_temp = True
                    sup_type = GenericDataType.TEMPORAL

            # Sample values for filter builder autocomplete
            sample_vals = [
                str(x) for x in df[col].drop_nulls().unique().head(10).to_list()
            ]

            col_meta = {
                "name": col,
                "type": sup_type,
                "type_name": sup_type.name,
                "is_numeric": is_num,
                "is_temporal": is_temp,
                "null_count": int(df[col].is_null().sum()),
                "unique_values": int(df[col].n_unique()),
                "sample_values": sample_vals,
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
            "time_grains": ["day", "week", "month", "quarter", "year"],
            "color_palettes": list(SUPERSET_COLOR_PALETTES.keys()),
        }

    def _apply_filters(self, df: pl.DataFrame, filters: Optional[List[ExploreFilter]]) -> pl.DataFrame:
        """Applies ad-hoc user filters prior to metric aggregation."""
        if not filters:
            return df

        for f in filters:
            col = f.column
            if col not in df.columns:
                continue

            op = f.operator.lower().strip()
            val = f.value
            col_type = df[col].dtype

            try:
                if op in ("is_null", "null"):
                    df = df.filter(pl.col(col).is_null())
                elif op in ("is_not_null", "not_null"):
                    df = df.filter(pl.col(col).is_not_null())
                elif val is not None:
                    if op in ("eq", "=="):
                        if col_type.is_numeric():
                            try:
                                df = df.filter(pl.col(col) == float(val))
                            except (ValueError, TypeError):
                                df = df.filter(pl.col(col).cast(pl.String) == str(val))
                        else:
                            df = df.filter(pl.col(col).cast(pl.String) == str(val))
                    elif op in ("neq", "!="):
                        if col_type.is_numeric():
                            try:
                                df = df.filter(pl.col(col) != float(val))
                            except (ValueError, TypeError):
                                df = df.filter(pl.col(col).cast(pl.String) != str(val))
                        else:
                            df = df.filter(pl.col(col).cast(pl.String) != str(val))
                    elif op in ("gt", ">"):
                        try:
                            df = df.filter(pl.col(col) > float(val))
                        except (ValueError, TypeError):
                            pass
                    elif op in ("gte", ">="):
                        try:
                            df = df.filter(pl.col(col) >= float(val))
                        except (ValueError, TypeError):
                            pass
                    elif op in ("lt", "<"):
                        try:
                            df = df.filter(pl.col(col) < float(val))
                        except (ValueError, TypeError):
                            pass
                    elif op in ("lte", "<="):
                        try:
                            df = df.filter(pl.col(col) <= float(val))
                        except (ValueError, TypeError):
                            pass
                    elif op == "in":
                        items = val if isinstance(val, list) else [val]
                        if col_type.is_numeric():
                            try:
                                num_items = [float(x) for x in items]
                                df = df.filter(pl.col(col).is_in(num_items))
                            except (ValueError, TypeError):
                                str_items = [str(x) for x in items]
                                df = df.filter(pl.col(col).cast(pl.String).is_in(str_items))
                        else:
                            str_items = [str(x) for x in items]
                            df = df.filter(pl.col(col).cast(pl.String).is_in(str_items))
                    elif op == "contains":
                        df = df.filter(pl.col(col).cast(pl.String).str.contains(str(val), literal=False))
            except Exception as fe:
                logger.warning(f"Failed to apply filter {f} on column '{col}': {fe}")

        return df

    def _apply_temporal_bucketing(self, df: pl.DataFrame, dimension: str, time_grain: Optional[str]) -> pl.DataFrame:
        """Applies date truncation (time-grain) to temporal dimensions."""
        if not time_grain or dimension not in df.columns:
            return df

        grain_map = {
            "day": "1d",
            "week": "1w",
            "month": "1mo",
            "quarter": "3mo",
            "year": "1y",
        }
        grain = grain_map.get(time_grain.lower().strip())
        if not grain:
            return df

        col_type = df[dimension].dtype
        try:
            if col_type in (pl.Date, pl.Datetime):
                df = df.with_columns(
                    pl.col(dimension).dt.truncate(grain).dt.to_string("%Y-%m-%d").alias(dimension)
                )
            elif col_type == pl.String:
                # Attempt to parse string to date and then truncate
                parsed = pl.col(dimension).str.to_date(strict=False)
                df = df.with_columns(
                    parsed.dt.truncate(grain).dt.to_string("%Y-%m-%d").alias(dimension)
                )
        except Exception as te:
            logger.warning(f"Could not apply temporal bucketing ({time_grain}) to '{dimension}': {te}")

        return df

    def execute_explore_query(self, query: SupersetExploreQuery) -> Dict[str, Any]:
        """
        Executes a Superset Explore query against the dataset and builds
        Apache ECharts configuration payload matching Superset's plugin-chart-echarts.
        """
        df = self.resolve_dataset_df(query.task_id)

        if query.dimension not in df.columns:
            raise ValueError(f"Dimension column '{query.dimension}' does not exist in dataset.")

        # 1. Apply ad-hoc user filters
        df = self._apply_filters(df, query.filters)
        if df.is_empty():
            return {
                "task_id": query.task_id,
                "chart_type": query.chart_type,
                "dimension": query.dimension,
                "secondary_dimension": query.secondary_dimension,
                "metric": query.metric,
                "aggregate": query.aggregate,
                "metric_label": "No Data (Filtered to 0 rows)",
                "row_limit": query.row_limit,
                "kpis": {
                    "metric_label": "Filtered Records",
                    "dimension_label": query.dimension,
                    "total_rows": 0,
                    "sum": 0,
                    "mean": None,
                    "max": None,
                    "min": None,
                    "unique_categories": 0,
                },
                "records": [],
                "total_groups": 0,
                "echarts_option": {
                    "title": {"text": "No matching records for current filters", "left": "center", "top": "middle", "textStyle": {"color": "#94a3b8", "fontSize": 14}}
                },
            }

        # 2. Apply temporal bucketing if requested
        if query.time_grain:
            df = self._apply_temporal_bucketing(df, query.dimension, query.time_grain)

        # 3. Metric aggregation expression
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

        # 4. Grouping columns
        group_cols = [query.dimension]
        has_secondary = bool(query.secondary_dimension and query.secondary_dimension in df.columns)
        if has_secondary and query.secondary_dimension != query.dimension:
            group_cols.append(query.secondary_dimension)

        # 5. Execute Polars Aggregation
        filtered = df.filter(pl.col(query.dimension).is_not_null())
        if has_secondary:
            filtered = filtered.filter(pl.col(query.secondary_dimension).is_not_null())

        # For boxplots, calculate 5-number summary directly per group
        boxplot_data = None
        if query.chart_type in ("echarts_boxplot", "boxplot") and metric_col != "records" and metric_col in df.columns:
            box_groups = filtered.group_by(query.dimension).all()
            box_records = []
            for row in box_groups.iter_rows(named=True):
                dim_val = str(row[query.dimension])
                vals = [float(v) for v in row[metric_col] if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if vals:
                    vals.sort()
                    n = len(vals)
                    q1 = vals[int(n * 0.25)]
                    med = vals[int(n * 0.50)]
                    q3 = vals[int(n * 0.75)]
                    box_records.append({
                        query.dimension: dim_val,
                        "min": vals[0],
                        "q1": q1,
                        "median": med,
                        "q3": q3,
                        "max": vals[-1],
                        "value": med,
                        "count": n
                    })
            sorted_box = sorted(box_records, key=lambda x: x["value"], reverse=(query.sort_order.lower() == "desc"))[:query.row_limit]
            cleaned_records = sorted_box
            grouped_len = len(box_records)
        else:
            grouped = filtered.group_by(group_cols).agg(
                val_expr,
                pl.len().alias("count")
            )

            descending = query.sort_order.lower() == "desc"
            sorted_df = grouped.sort("value", descending=descending).head(query.row_limit)

            raw_records = sorted_df.to_dicts()
            cleaned_records = []
            for r in raw_records:
                cleaned_row = {k: _sanitize_for_json(v) for k, v in r.items()}
                cleaned_records.append(cleaned_row)
            grouped_len = len(grouped)

        # 6. Calculate high-level KPIs for BigNumber card
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

        # 7. Build ECharts options payload
        echarts_option = self._build_echarts_options(
            chart_type=query.chart_type,
            records=cleaned_records,
            dimension=query.dimension,
            secondary_dimension=query.secondary_dimension if has_secondary else None,
            metric_label=metric_label,
            palette=palette,
            has_time_grain=bool(query.time_grain),
        )

        return {
            "task_id": query.task_id,
            "chart_type": query.chart_type,
            "dimension": query.dimension,
            "secondary_dimension": query.secondary_dimension if has_secondary else None,
            "metric": query.metric,
            "aggregate": query.aggregate,
            "time_grain": query.time_grain,
            "filters": [f.model_dump() for f in (query.filters or [])],
            "metric_label": metric_label,
            "row_limit": query.row_limit,
            "kpis": kpis,
            "records": cleaned_records,
            "total_groups": grouped_len,
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
        has_time_grain: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds Superset-standard ECharts option dictionary for direct rendering.
        Supports bar, line, area, pie, treemap, scatter, heatmap, boxplot, and table.
        """
        if not records:
            return {"title": {"text": "No data available", "left": "center", "top": "middle", "textStyle": {"color": "#94a3b8"}}}

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

        # ── 3. Heatmap Matrix Chart ───────────────────────────────────────
        elif chart_type in ("echarts_heatmap", "heatmap") and secondary_dimension:
            x_cats = list(dict.fromkeys(str(r[dimension]) for r in records))
            y_cats = list(dict.fromkeys(str(r.get(secondary_dimension, "N/A")) for r in records))
            lookup = {(str(r[dimension]), str(r.get(secondary_dimension, "N/A"))): r.get("value", 0) for r in records}

            matrix_data = []
            vals = []
            for xi, xc in enumerate(x_cats):
                for yi, yc in enumerate(y_cats):
                    val = lookup.get((xc, yc), 0)
                    matrix_data.append([xi, yi, val])
                    if val is not None:
                        vals.append(val)

            min_v = min(vals) if vals else 0
            max_v = max(vals) if vals else 100

            return {
                "backgroundColor": "transparent",
                "tooltip": {
                    "position": "top",
                    "backgroundColor": "#181f30",
                    "borderColor": "#334155",
                    "textStyle": {"color": "#f8fafc", "fontSize": 12},
                    "formatter": "{c}",
                },
                "grid": {"height": "65%", "top": "10%", "left": "10%", "right": "5%"},
                "xAxis": {
                    "type": "category",
                    "data": x_cats,
                    "splitArea": {"show": True},
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11, "rotate": 30 if len(x_cats) > 6 else 0},
                },
                "yAxis": {
                    "type": "category",
                    "data": y_cats,
                    "splitArea": {"show": True},
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                },
                "visualMap": {
                    "min": min_v,
                    "max": max_v,
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": "0%",
                    "textStyle": {"color": "#94a3b8", "fontSize": 10},
                    "inRange": {
                        "color": ["#0f172a", "#0369a1", "#0284c7", "#38bdf8", "#34d399", "#fbbf24"]
                    },
                },
                "series": [
                    {
                        "name": metric_label,
                        "type": "heatmap",
                        "data": matrix_data,
                        "label": {"show": len(matrix_data) < 100, "color": "#f8fafc", "fontSize": 10},
                        "emphasis": {
                            "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                        },
                    }
                ],
            }

        # ── 4. Boxplot Chart ──────────────────────────────────────────────
        elif chart_type in ("echarts_boxplot", "boxplot"):
            categories = [str(r[dimension]) for r in records]
            box_data = [
                [r.get("min", 0), r.get("q1", 0), r.get("median", 0), r.get("q3", 0), r.get("max", 0)]
                for r in records
            ]
            return {
                "backgroundColor": "transparent",
                "color": palette,
                "tooltip": {
                    "trigger": "item",
                    "axisPointer": {"type": "shadow"},
                    "backgroundColor": "#181f30",
                    "borderColor": "#334155",
                    "textStyle": {"color": "#f8fafc", "fontSize": 12},
                },
                "grid": {"left": "5%", "right": "5%", "bottom": "15%", "top": "10%", "containLabel": True},
                "xAxis": {
                    "type": "category",
                    "data": categories,
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11, "rotate": 30 if len(categories) > 6 else 0},
                    "axisLine": {"lineStyle": {"color": "#334155"}},
                },
                "yAxis": {
                    "type": "value",
                    "name": metric_label,
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                    "splitLine": {"lineStyle": {"color": "#1e293b", "type": "dashed"}},
                },
                "series": [
                    {
                        "name": metric_label,
                        "type": "boxplot",
                        "data": box_data,
                        "itemStyle": {
                            "color": "rgba(56, 189, 248, 0.3)",
                            "borderColor": palette[0],
                            "borderWidth": 2,
                        },
                    }
                ],
            }

        # ── 5. Scatter / Correlation Chart ────────────────────────────────
        elif chart_type in ("echarts_scatter", "scatter"):
            scatter_data = [[r[dimension], r.get("value", 0)] for r in records]
            return {
                "backgroundColor": "transparent",
                "color": palette,
                "tooltip": {
                    "trigger": "item",
                    "backgroundColor": "#181f30",
                    "borderColor": "#334155",
                    "textStyle": {"color": "#f8fafc", "fontSize": 12},
                    "formatter": f"{dimension}: <b>{{@[0]}}</b><br/>{metric_label}: <b>{{@[1]}}</b>",
                },
                "grid": {"left": "5%", "right": "5%", "bottom": "15%", "top": "10%", "containLabel": True},
                "xAxis": {
                    "type": "category" if any(not isinstance(r[0], (int, float)) for r in scatter_data) else "value",
                    "name": dimension,
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                    "axisLine": {"lineStyle": {"color": "#334155"}},
                },
                "yAxis": {
                    "type": "value",
                    "name": metric_label,
                    "axisLabel": {"color": "#94a3b8", "fontSize": 11},
                    "splitLine": {"lineStyle": {"color": "#1e293b", "type": "dashed"}},
                },
                "series": [
                    {
                        "name": metric_label,
                        "type": "scatter",
                        "symbolSize": 12,
                        "data": scatter_data,
                        "itemStyle": {"color": palette[0], "shadowBlur": 6, "shadowColor": palette[0]},
                    }
                ],
            }

        # ── 6. Categorical Bar, Column, Line, and Area Charts ─────────────
        categories = list(dict.fromkeys(str(r[dimension]) for r in records))
        needs_data_zoom = len(categories) > 15 or has_time_grain

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
            legend = {"show": True, "bottom": "2%" if not needs_data_zoom else "7%", "textStyle": {"color": "#94a3b8", "fontSize": 11}}
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

        echarts_dict: Dict[str, Any] = {
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
                "bottom": "18%" if needs_data_zoom else ("14%" if secondary_dimension or len(categories) > 8 else "8%"),
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
                    "interval": 0 if len(categories) <= 25 else "auto",
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

        # Add interactive dataZoom slider for time series and long categories
        if needs_data_zoom:
            echarts_dict["dataZoom"] = [
                {
                    "type": "slider",
                    "show": True,
                    "bottom": "1%",
                    "height": 18,
                    "borderColor": "#334155",
                    "fillerColor": "rgba(56, 189, 248, 0.15)",
                    "handleStyle": {"color": "#38bdf8", "borderColor": "#0284c7"},
                    "textStyle": {"color": "#94a3b8", "fontSize": 10},
                },
                {"type": "inside"}
            ]

        return echarts_dict


superset_engine = SupersetEngine()
