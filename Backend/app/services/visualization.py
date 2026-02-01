import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64
import logging

logger = logging.getLogger(__name__)

def fig_to_base64(fig) -> str:
    """
    Converts a Matplotlib figure to a Base64 string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) # Critical: Free up memory
    return img_str

def generate_charts(df: pd.DataFrame) -> dict:
    """
    Generates a suite of visualizations for the dataset.
    Returns a dictionary of Base64 strings.
    """
    charts = {}
    
    try:
        numeric_df = df.select_dtypes(include=['number'])
        
        # 1. Correlation Heatmap
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Matrix")
            charts['correlation_heatmap'] = fig_to_base64(plt.gcf())

        # 2. Distribution Plots (Top 5 features by variance)
        if not numeric_df.empty:
            # Select columns with highest variance to show interesting distributions
            variances = numeric_df.var().sort_values(ascending=False)
            top_features = variances.head(5).index.tolist()
            
            distributions = []
            for col in top_features:
                plt.figure(figsize=(8, 5))
                sns.histplot(numeric_df[col], kde=True, color='skyblue')
                plt.title(f"Distribution of {col}")
                distributions.append({
                    "column": col,
                    "image": fig_to_base64(plt.gcf())
                })
            charts['distributions'] = distributions

        # 3. Categorical Counts (Bar Charts - Top 3 categorical columns)
        categorical_df = df.select_dtypes(exclude=['number'])
        if not categorical_df.empty:
            cat_charts = []
            for col in categorical_df.columns[:3]:
                 # Only plot if cardinality is manageable (< 20 unique values)
                if categorical_df[col].nunique() < 20:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(y=df[col], order=df[col].value_counts().index)
                    plt.title(f"Counts: {col}")
                    cat_charts.append({
                        "column": col,
                        "image": fig_to_base64(plt.gcf())
                    })
            charts['categorical'] = cat_charts

        return charts

    except Exception as e:
        logger.error(f"Error generating charts: {str(e)}")
        # Don't fail the entire request if charts fail, just return empty or error image
        return {"error": str(e)}
