import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
from io import BytesIO
import logging

# Configure logic
logger = logging.getLogger(__name__)

async def load_dataframe(file: UploadFile) -> pd.DataFrame:
    """
    Loads a CSV or Excel file into a Pandas DataFrame.
    """
    contents = await file.read()
    buffer = BytesIO(contents)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(buffer)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        return df
    except Exception as e:
        logger.error(f"Error loading file {file.filename}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not parse file: {str(e)}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs production-grade data cleaning:
    1. Removes empty columns/rows.
    2. Infers correct data types.
    3. Standardizes column names.
    4. Handles missing values appropriately.
    """
    # 1. Drop completely empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    
    # 2. Standardize column names (snake_case, strip whitespace)
    df.columns = df.columns.astype(str).str.strip()
    
    # 3. Intelligent Type Inference
    for col in df.columns:
        # Try numeric conversion
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except (ValueError, TypeError):
            pass
        
        # Try datetime conversion using flexible parsing
        try:
            df[col] = pd.to_datetime(df[col], format='mixed')
            continue
        except (ValueError, TypeError):
            pass
            
    # 4. Fill NaNs with appropriate defaults based on type
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Fill numeric NaNs with 0 (or median, depending on strategy)
            # For this report app, 0 is safer than assuming median
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")

    return df

def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    Returns metadata about the dataset for the frontend.
    """
    buffer = BytesIO()
    df.info(buf=buffer)
    
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "summary": df.describe(include='all').fillna("").to_dict(),
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient='records')
    }
