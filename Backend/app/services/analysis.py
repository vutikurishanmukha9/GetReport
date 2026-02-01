import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Performs comprehensive statistical analysis on the dataset.
    """
    try:
        # Separate numeric and categorical columns
        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(exclude=[np.number])
        
        analysis_result = {
            "summary": {},
            "correlation": {},
            "outliers": {},
            "categorical_distribution": {}
        }
        
        # 1. Descriptive Statistics (Robust)
        if not numeric_df.empty:
            summary = numeric_df.describe().T
            summary['skewness'] = numeric_df.skew()
            summary['kurtosis'] = numeric_df.kurtosis()
            analysis_result["summary"] = summary.to_dict(orient='index')
            
            # 2. Correlation Matrix
            # Handle potential NaNs or constant columns that cause correlation errors
            corr_matrix = numeric_df.corr(method='pearson').fillna(0)
            analysis_result["correlation"] = corr_matrix.to_dict()
            
            # 3. Outlier Detection (Interquartile Range - IQR)
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)][col]
                
                if not outliers.empty:
                    analysis_result["outliers"][col] = {
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(df)) * 100,
                        "min_outlier": outliers.min(),
                        "max_outlier": outliers.max()
                    }

        # 4. Categorical Distribution
        for col in categorical_df.columns:
            # Limit to top 10 categories to avoid massive payloads
            dist = categorical_df[col].value_counts().head(10)
            analysis_result["categorical_distribution"][col] = dist.to_dict()
            
        return analysis_result

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        # Return partial results or re-raise depending on strictness. 
        # Here we re-raise as analysis failure is critical.
        raise ValueError(f"Analysis failed: {str(e)}")
