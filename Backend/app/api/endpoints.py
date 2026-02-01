from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.services.data_processing import load_dataframe, clean_data, get_dataset_info
from app.services.analysis import analyze_dataset
from app.services.visualization import generate_charts
from app.services.report_generator import generate_pdf_report
from app.services.llm_insight import generate_insights
import base64
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload CSV or Excel.")
    
    try:
        # 1. Load Data
        df = await load_dataframe(file)
        
        # 2. Clean Data (Production Grade)
        cleaned_df = clean_data(df)
        
        # 3. Basic Info (for Frontend Preview)
        dataset_info = get_dataset_info(cleaned_df)
        
        # 4. Perform Analysis
        analysis_result = analyze_dataset(cleaned_df)
        
        # 5. Generate Charts
        charts = generate_charts(cleaned_df)
        
        # 6. Generate Insights (Async/Background optional)
        insights = await generate_insights(analysis_result.get("summary", {}))
        
        # 7. Prepare Response
        response_data = {
            "filename": file.filename,
            "info": dataset_info,
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        # In production, hide specific error details or map to user-friendly messages
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/generate-report")
async def generate_report_endpoint(data: dict):
    # Placeholder for PDF generation trigger
    return {"message": "Report generation endpoint not yet connected"}
