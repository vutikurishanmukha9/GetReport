from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, List
import logging

from app.services.data_processing import (
    load_dataframe, clean_data, get_dataset_info, 
    UnsupportedFileTypeError, FileTooLargeError, EmptyFileError, ParseError
)
from app.services.analysis import (
    analyze_dataset, EmptyDatasetError, InsufficientDataError, AnalysisError
)
from app.services.visualization import generate_charts
from app.services.report_generator import generate_pdf_report
from app.services.llm_insight import generate_insights

logger = logging.getLogger(__name__)
router = APIRouter()

# ─── Data Models ─────────────────────────────────────────────────────────────
class ReportRequest(BaseModel):
    filename: str
    analysis: Dict[str, Any]
    charts: Dict[str, Any]

# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file upload, cleaning, analysis, and insight generation.
    Returns a comprehensive JSON object for the frontend dashboard.
    """
    try:
        # 1. Load Data
        df = await load_dataframe(file)
        
        # 2. Clean Data (Returns cleaned_df and CleaningReport object)
        cleaned_df, cleaning_report = clean_data(df)
        
        # 3. Basic Info (for Frontend Preview)
        dataset_info = get_dataset_info(cleaned_df)
        
        # Convert report to dict for JSON serialization
        cleaning_report_dict = cleaning_report.to_dict()
        
        # 4. Perform Analysis
        analysis_result = analyze_dataset(cleaned_df)
        
        # 5. Generate Charts (now returns tuple)
        charts, charts_report = generate_charts(cleaned_df)
        
        # We can add 'charts_report' to the response if desired, for now just use 'charts'
        
        # 6. Generate Insights (Async)
        # Pass the full analysis_result so the LLM sees everything
        insights_result = await generate_insights(analysis_result)
        
        # 7. Prepare Response
        response_data = {
            "filename": file.filename,
            "info": dataset_info,
            "cleaning_report": cleaning_report_dict,
            "analysis": analysis_result,
            "charts": charts,
            "insights": insights_result.to_dict()  # Serialize InsightResult
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
async def generate_report_endpoint(request: ReportRequest):
    """
    Generates a PDF report from the provided analysis data and charts.
    Returns a downloadable PDF file via StreamingResponse.
    
    Expects JSON payload:
    {
        "filename": "...",
        "analysis": { ... },
        "charts": { ... }
    }
    """
    try:
        # Generate the PDF
        # The 'analysis' dict must include 'insights' text if checking against report_generator expectations.
        # Ensure the frontend passes the FULL analysis object or merges insights into it.
        # Based on /upload response, 'insights' is at root level, but analysis is separate.
        # We might need to merge them if report_generator expects insights INSIDE analysis_results.
        
        # Checking report_generator.py:
        # def _build_insights_section(analysis_results...):
        #    if "insights" not in analysis_results...
        
        # So YES, 'insights' must be IN request.analysis for it to appear in the PDF.
        # The frontend needs to ensure this. 
        # For robustness, we can check if it's missing in analysis but present elsewhere?
        # But ReportRequest definition only has 'analysis' and 'charts'.
        # We assume 'request.analysis' is the COMBINED object or the frontend prepares it correctly.
        
        pdf_buffer, metadata = generate_pdf_report(
            request.analysis,
            request.charts,
            request.filename
        )
        
        logger.info(f"Report generated successfully: {metadata}")
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=Report_{request.filename}.pdf",
                "X-Report-Metadata": str(metadata) # Optional debug header
            }
        )

    except Exception as e:
        logger.error(f"Report generation endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
