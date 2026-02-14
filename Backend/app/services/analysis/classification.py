from __future__ import annotations
import polars as pl
from app.core.config import settings

# ─── Semantic Column Detection Thresholds ─────────────────────────────────────
ID_UNIQUENESS_THRESHOLD: float = settings.ID_UNIQUENESS_THRESHOLD  # >98% unique values AND name suggests ID = likely ID
EXCEL_DATE_RANGE = (25569, 73050)  # Excel serial dates: 1970-2100
# Only very strong ID patterns - avoid false positives like "student_number" which is valid data
ID_COLUMN_PATTERNS = ['_id', 'uuid', 'guid', 'pk', 'primary_key', 'row_id', 'record_id']
DATE_COLUMN_PATTERNS = ['date', 'time', 'timestamp', 'created_at', 'updated_at', 'modified_at']

def classify_numeric_columns(df: pl.DataFrame, numeric_cols: list[str]) -> dict[str, list[str]]:
    """
    Classify numeric columns into semantic categories:
    - analytical: Real numeric data suitable for correlation/distribution (e.g., sales, price)
    - id_like: Likely identifiers (high uniqueness, sequential) - exclude from analysis
    - date_like: Likely Excel serial dates (values in 25569-73050 range) - exclude from analysis
    - low_variance: Near-constant values - exclude from correlation
    
    Returns dict with 'analytical', 'excluded', 'exclusion_reasons'
    """
    analytical = []
    excluded = []
    exclusion_reasons = {}
    
    for col in numeric_cols:
        col_lower = col.lower()
        reasons = []
        
        # Check 1: Column name patterns
        is_id_name = any(pattern in col_lower for pattern in ID_COLUMN_PATTERNS)
        is_date_name = any(pattern in col_lower for pattern in DATE_COLUMN_PATTERNS)
        
        if is_id_name:
            reasons.append("name_suggests_id")
        if is_date_name:
            reasons.append("name_suggests_date")
        
        # Check 2: High uniqueness = likely ID
        try:
            n_unique = df[col].n_unique()
            uniqueness_ratio = n_unique / df.height if df.height > 0 else 0
            
            if uniqueness_ratio >= ID_UNIQUENESS_THRESHOLD:
                reasons.append(f"high_uniqueness_{round(uniqueness_ratio*100)}%")
        except:
            pass
        
        # Check 3: Excel serial date range detection
        try:
            non_null = df[col].drop_nulls()
            if non_null.len() > 0:
                min_val = non_null.min()
                max_val = non_null.max()
                
                # Excel serial date range check (1970-2100)
                if EXCEL_DATE_RANGE[0] <= min_val <= EXCEL_DATE_RANGE[1] and \
                   EXCEL_DATE_RANGE[0] <= max_val <= EXCEL_DATE_RANGE[1]:
                    # Additional check: values are mostly integers and within date range spread
                    int_ratio = (non_null == non_null.round(0)).sum() / non_null.len()
                    if int_ratio > 0.9:  # 90% integer values
                        reasons.append("excel_serial_date_range")
        except:
            pass
        
        # Check 4: Low variance (near-constant)
        try:
            std = df[col].std()
            mean = df[col].mean()
            if std is not None and mean is not None and mean != 0:
                cv = abs(std / mean)  # Coefficient of variation
                if cv < 0.01:  # <1% variation
                    reasons.append("low_variance")
        except:
            pass
        
        # Decision: Be conservative - only exclude if:
        # 1. Low variance (always exclude near-constants), OR
        # 2. BOTH name pattern AND data evidence (high uniqueness or date range)
        should_exclude = False
        
        if "low_variance" in reasons:
            should_exclude = True  # Low variance is always excluded
        elif (is_id_name or is_date_name):
            # Name suggests ID/date - need additional data evidence
            has_data_evidence = any(
                r for r in reasons 
                if "high_uniqueness" in r or "excel_serial_date" in r
            )
            if has_data_evidence:
                should_exclude = True
        
        if should_exclude:
            excluded.append(col)
            exclusion_reasons[col] = reasons
        else:
            analytical.append(col)
    
    return {
        "analytical": analytical,
        "excluded": excluded,
        "exclusion_reasons": exclusion_reasons
    }
