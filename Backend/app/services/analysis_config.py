from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

DomainType = Literal["general", "finance", "medical", "marketing", "sales"]

class AnalysisConfig(BaseModel):
    """
    Configuration for the analysis engine.
    Controls which analyses are run and their sensitivity thresholds.
    """
    # ─── Toggles ─────────────────────────────────────────────────────────────
    enable_correlation: bool = True
    enable_outliers: bool = True
    enable_time_series: bool = True
    enable_categorical: bool = True
    enable_text_analysis: bool = False  # Expensive/Slow defaults to False
    enable_feature_engineering_recs: bool = True
    enable_smart_schema: bool = True
    
    # ─── Thresholds ──────────────────────────────────────────────────────────
    correlation_strong_threshold: float = 0.7
    outlier_iqr_multiplier: float = 1.5
    min_distinct_for_categorical: int = 2
    max_distinct_for_categorical: int = 50  # Max categories to analyze specifically
    
    # ─── Domain Context ──────────────────────────────────────────────────────
    domain: DomainType = "general"
    
    @classmethod
    def default(cls) -> "AnalysisConfig":
        return cls()

    @classmethod
    def for_domain(cls, domain: str = "general") -> "AnalysisConfig":
        """Factory method to get tuned configs for specific domains."""
        domain = domain.lower()
        
        if domain == "finance":
            return cls(
                domain="finance",
                enable_time_series=True,
                enable_outliers=True,
                outlier_iqr_multiplier=2.0,  # Finance data often has heavy tails, strict 1.5 flags too much
                correlation_strong_threshold=0.8, # Higher bar for correlation
            )
            
        elif domain == "medical":
            return cls(
                domain="medical",
                enable_outliers=True,
                outlier_iqr_multiplier=1.5,
                enable_text_analysis=True, # Notes are common
            )
            
        elif domain == "marketing":
            return cls(
                domain="marketing",
                enable_categorical=True,
                max_distinct_for_categorical=100, # Marketing often has many segments/campaigns
            )

        return cls(domain="general")
