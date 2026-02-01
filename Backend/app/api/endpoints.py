from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from app.services.data_processing import load_dataframe, clean_data, get_dataset_info, UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError
from app.services.analysis import analyze_dataset, EmptyDatasetError, InsufficientDataError, AnalysisError
from app.services.visualization import generate_charts
from app.services.report_generator import generate_pdf_report
from app.services.llm_insight import generate_insights
import base64
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Original validation removed here as it's now handled in load_dataframe (strict whitelisting)
    # But we keep a check if file object itself is invalid? No, FastAPI handles that.
    
    try:
        # 1. Load Data
        df = await load_dataframe(file)
        
        # 2. Clean Data (Production Grade) - Returns (df, report)
        cleaned_df, cleaning_report = clean_data(df)
        
        # 3. Basic Info (for Frontend Preview) - Accepts cleaned_df
        dataset_info = get_dataset_info(cleaned_df)
        
        # Explicitly convert cleaning_report to dict for JSON response
        cleaning_report_dict = cleaning_report.to_dict()
        
        # 4. Perform Analysis
        analysis_result = analyze_dataset(cleaned_df)
        
        # 5. Generate Charts
        charts = generate_charts(cleaned_df)
        
        # 6. Generate Insights (Async/Background optional)
        # Pass the FULL analysis_result, not just summary
        insights_result = await generate_insights(analysis_result)
        
        # 7. Prepare Response
        response_data = {
            "filename": file.filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report_dict, # Added this field
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights_result.to_dict()
        }
        
        return JSONResponse(content=response_data)

    except (UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError) as e:
        logger.warning(f"Upload validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except (EmptyDatasetError, InsufficientDataError) as e:
        logger.warning(f"Analysis input error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except AnalysisError as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis engine error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/generate-report")
async def generate_report_endpoint(data: dict):
    # Placeholder for PDF generation trigger
    return {"message": "Report generation endpoint not yet connected"}
