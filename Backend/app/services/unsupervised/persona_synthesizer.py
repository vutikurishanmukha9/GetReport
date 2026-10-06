"""
Contrastive Persona Profile Synthesizer
=======================================
Translates numerical cluster centroids into human-readable business personas
by contrasting cluster averages against the population baseline.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import polars as pl

from .schemas import ClusterFeatureDiff, ClusterPersona


class PersonaSynthesizer:
    """
    Synthesizes actionable segment profiles from cluster memberships.
    """

    @classmethod
    def synthesize(
        cls,
        df: pl.DataFrame,
        labels: np.ndarray,
        k: int,
        numeric_cols: List[str],
    ) -> List[ClusterPersona]:
        """
        Contrasts cluster means against population baseline and generates summaries.
        """
        total_rows = df.height
        personas: List[ClusterPersona] = []

        # Population averages
        pop_means: Dict[str, float] = {}
        for col in numeric_cols:
            m = df[col].cast(pl.Float64, strict=False).mean()
            pop_means[col] = float(m) if m is not None and not math.isnan(m) else 0.0

        for c in range(k):
            c_mask = labels == c
            c_size = int(c_mask.sum())
            if c_size == 0:
                continue

            share_pct = (c_size / total_rows) * 100.0
            sub_df = df.filter(pl.Series(c_mask))

            diffs: List[ClusterFeatureDiff] = []
            for col in numeric_cols:
                pop_m = pop_means.get(col, 0.0)
                sub_m = sub_df[col].cast(pl.Float64, strict=False).mean()
                c_m = float(sub_m) if sub_m is not None and not math.isnan(sub_m) else 0.0

                if abs(pop_m) > 1e-4:
                    index_ratio = c_m / pop_m
                else:
                    index_ratio = 1.0

                direction = "higher" if index_ratio >= 1.05 else ("lower" if index_ratio <= 0.95 else "average")
                if direction != "average":
                    diffs.append(
                        ClusterFeatureDiff(
                            feature=col,
                            cluster_mean=c_m,
                            population_mean=pop_m,
                            index_ratio=index_ratio,
                            direction=direction,
                        )
                    )

            # Sort by deviation from 1.0 (highest over/under indexation)
            diffs.sort(key=lambda d: -abs(d.index_ratio - 1.0))
            top_diffs = diffs[:4]

            if top_diffs:
                lead = top_diffs[0]
                sign = f"{lead.index_ratio:.1f}x higher" if lead.direction == "higher" else f"{abs(1.0 - lead.index_ratio) * 100:.0f}% lower"
                col_clean = lead.feature.replace("_", " ").title()
                name = f"Cluster {c + 1}: {col_clean} ({sign})"

                summary_parts = [
                    f"{d.feature.replace('_', ' ').title()} is {d.index_ratio:.1f}x {d.direction} than average"
                    for d in top_diffs[:2]
                ]
                summary = f"Represents {share_pct:.1f}% of data ({c_size:,} records). Key pattern: " + "; ".join(summary_parts) + "."
            else:
                name = f"Cluster {c + 1}: Balanced Segment"
                summary = f"Represents {share_pct:.1f}% of data ({c_size:,} records) with near-average distribution across all metrics."

            personas.append(
                ClusterPersona(
                    cluster_id=c,
                    name=name,
                    size=c_size,
                    share_pct=share_pct,
                    distinguishing_features=top_diffs,
                    summary=summary,
                )
            )

        return personas
