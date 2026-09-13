"""
Apache Superset-Inspired Declarative Chart Specification Engine for GetReport.
Defines a standard declarative JSON schema for converting tabular SQL / profiling
outputs into rich, interactive visualization specifications (Bar, Line, Area, Scatter, Pie, Treemap).
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {"bar", "line", "area", "scatter", "pie", "treemap"}

COLOR_PALETTES = {
    "modern_dark": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"],
    "emerald": ["#059669", "#10b981", "#34d399", "#6ee7b7", "#a7f3d0"],
    "cyberpunk": ["#ff007f", "#7928ca", "#00dfd8", "#ff4b4b", "#f9cb28"]
}


class DeclarativeChartBuilder:
    """
    Constructs Apache Superset-style declarative chart specifications.
    Decouples query extraction from visual presentation.
    """
    @classmethod
    def build_spec(
        cls,
        chart_type: str,
        title: str,
        data: List[Dict[str, Any]],
        x_axis: str,
        metrics: List[str],
        group_by: Optional[str] = None,
        palette_name: str = "modern_dark",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        c_type = chart_type.lower()
        if c_type not in SUPPORTED_CHART_TYPES:
            logger.warning(f"Unsupported chart type '{chart_type}', defaulting to 'bar'")
            c_type = "bar"

        colors = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["modern_dark"])
        opts = options or {}

        spec = {
            "version": "1.0.0",
            "chart_type": c_type,
            "title": title,
            "encoding": {
                "x": {
                    "field": x_axis,
                    "title": opts.get("x_title", x_axis.replace("_", " ").title()),
                    "type": "temporal" if "time" in x_axis.lower() or "date" in x_axis.lower() else "nominal"
                },
                "y": {
                    "fields": metrics,
                    "title": opts.get("y_title", ", ".join([m.replace("_", " ").title() for m in metrics])),
                    "type": "quantitative"
                },
                "color": {
                    "palette": colors,
                    "group_field": group_by
                },
                "tooltip": [x_axis] + metrics + ([group_by] if group_by else [])
            },
            "ui_config": {
                "responsive": True,
                "show_grid": opts.get("show_grid", True),
                "show_legend": len(metrics) > 1 or bool(group_by),
                "animation": opts.get("animation", True),
                "smooth_curves": c_type in ("line", "area") and opts.get("smooth", True)
            },
            "data": data[: opts.get("max_data_points", 500)]
        }

        return spec

    @classmethod
    def auto_infer_chart_spec(
        cls,
        data: List[Dict[str, Any]],
        title: str = "Query Visualization"
    ) -> Optional[Dict[str, Any]]:
        """
        Auto-infers the best chart type and axis mappings from tabular query results.
        """
        if not data:
            return None

        sample = data[0]
        numeric_fields = []
        categorical_fields = []
        temporal_fields = []

        for k, v in sample.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_fields.append(k)
            elif isinstance(v, str):
                if any(t in k.lower() for t in ("date", "time", "year", "month", "day")):
                    temporal_fields.append(k)
                else:
                    categorical_fields.append(k)

        if not numeric_fields:
            return None

        # 1. Temporal column present -> Line chart
        if temporal_fields:
            return cls.build_spec(
                chart_type="line",
                title=title,
                data=data,
                x_axis=temporal_fields[0],
                metrics=numeric_fields[:3]
            )

        # 2. Categorical + Numeric -> Bar chart
        if categorical_fields:
            return cls.build_spec(
                chart_type="bar",
                title=title,
                data=data,
                x_axis=categorical_fields[0],
                metrics=[numeric_fields[0]]
            )

        # 3. Two numeric columns -> Scatter chart
        if len(numeric_fields) >= 2:
            return cls.build_spec(
                chart_type="scatter",
                title=title,
                data=data,
                x_axis=numeric_fields[0],
                metrics=[numeric_fields[1]]
            )

        # Fallback
        return cls.build_spec(
            chart_type="bar",
            title=title,
            data=data,
            x_axis=list(sample.keys())[0],
            metrics=[numeric_fields[0]]
        )
