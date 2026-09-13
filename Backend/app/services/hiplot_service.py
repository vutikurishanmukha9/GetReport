"""
Facebook Research HiPlot-Inspired High-Dimensional Visual Discovery Engine for GetReport.
Prepares parallel coordinates multi-axis payloads with automatic type inference,
logarithmic scaling, and percentile normalization.
"""

from __future__ import annotations
import math
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


class HiPlotDimensionType:
    NUMERIC = "numeric"
    NUMERIC_LOG = "numeric_log"
    NUMERIC_PERCENTILE = "numeric_percentile"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"


class HiPlotDataService:
    """
    Transforms tabular Polars datasets into optimized HiPlot parallel coordinates payloads.
    """
    @classmethod
    def prepare_hiplot_payload(
        cls,
        df: pl.DataFrame,
        max_rows: int = 2000,
        selected_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extracts dimension definitions, scale normalizations, and sampled datapoints.
        """
        if df.is_empty():
            return {"dimensions": [], "datapoints": [], "total_rows": 0, "sampled_rows": 0}

        # Subsample if dataset exceeds max_rows
        total_rows = df.height
        if total_rows > max_rows:
            sample_df = df.sample(n=max_rows, seed=42)
            sampled = True
        else:
            sample_df = df
            sampled = False

        # Filter columns
        cols = selected_columns or df.columns
        cols = [c for c in cols if c in df.columns][:30]  # Cap at top 30 dimensions for visual clarity

        dimensions = []
        for col in cols:
            s = sample_df[col]
            dtype = s.dtype

            # 1. Temporal column
            if dtype in (pl.Date, pl.Datetime):
                dimensions.append({
                    "name": col,
                    "display_name": col.replace("_", " ").title(),
                    "type": HiPlotDimensionType.TIMESTAMP,
                    "is_numeric": False
                })

            # 2. Numeric column
            elif dtype.is_numeric():
                s_clean = s.drop_nulls()
                if s_clean.is_empty():
                    continue

                arr = s_clean.to_numpy().astype(np.float64)
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                mean_v = float(np.mean(arr))
                std_v = float(np.std(arr))

                # Check for heavy positive skew / dynamic range across orders of magnitude to recommend log scale
                use_log = False
                if min_v > 0 and (max_v / (min_v + 1e-6)) >= 100:
                    use_log = True
                    dim_type = HiPlotDimensionType.NUMERIC_LOG
                else:
                    dim_type = HiPlotDimensionType.NUMERIC

                # Compute 5th and 95th percentiles for robust scaling
                p5 = float(np.percentile(arr, 5))
                p95 = float(np.percentile(arr, 95))

                dimensions.append({
                    "name": col,
                    "display_name": col.replace("_", " ").title(),
                    "type": dim_type,
                    "is_numeric": True,
                    "min": round(min_v, 4),
                    "max": round(max_v, 4),
                    "mean": round(mean_v, 4),
                    "p5": round(p5, 4),
                    "p95": round(p95, 4)
                })

            # 3. Categorical column
            else:
                categories = [str(x) for x in s.drop_nulls().unique().head(25).to_list()]
                dimensions.append({
                    "name": col,
                    "display_name": col.replace("_", " ").title(),
                    "type": HiPlotDimensionType.CATEGORICAL,
                    "is_numeric": False,
                    "categories": sorted(categories)
                })

        # Format datapoints
        datapoints = sample_df.select([d["name"] for d in dimensions]).to_dicts()

        return {
            "total_rows": total_rows,
            "sampled_rows": len(datapoints),
            "is_sampled": sampled,
            "dimensions": dimensions,
            "datapoints": datapoints
        }
