"""
file_validation.py
~~~~~~~~~~~~~~~~~~
Security hardening for file uploads.
Validates file content using Magic Numbers (signatures) instead of just extensions.
"""
import logging
from fastapi import UploadFile, HTTPException

logger = logging.getLogger(__name__)

# Magic Numbers (File Signatures)
SIGNATURES = {
    # Office Open XML (xlsx, docx, pptx) - technically ZIP archives
    "xlsx": b"\x50\x4B\x03\x04", 
    # Legacy Microsoft Office (xls, doc, ppt) - OLE2 Compound File
    "xls":  b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1",
}

async def validate_file_signature(file: UploadFile) -> None:
    """
    Validate file content matches its extension using magic numbers.
    Raises HTTPException(400) if invalid.
    Resets file pointer to 0 after checking.
    """
    filename = file.filename.lower()
    
    # Read start of file
    await file.seek(0)
    header = await file.read(8)
    await file.seek(0)  # Reset immediately
    
    # 1. Excel (XLSX)
    if filename.endswith(".xlsx"):
        if not header.startswith(SIGNATURES["xlsx"]):
            logger.warning(f"Validation failed: {filename} claims to be XLSX but lacks ZIP signature.")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file content. Extension says .xlsx but content does not match (ZIP signature missing)."
            )

    # 2. Excel 97-2003 (XLS)
    elif filename.endswith(".xls"):
        if not header.startswith(SIGNATURES["xls"]):
            logger.warning(f"Validation failed: {filename} claims to be XLS but lacks OLE2 signature.")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file content. Extension says .xls but content does not match (OLE2 signature missing)."
            )

    # 3. CSV (Text-based)
    elif filename.endswith(".csv"):
        # CSVs don't have magic numbers. We check if the header looks like text.
        # Binary files often likely contain null bytes, which are rare in valid CSVs.
        if b"\x00" in header:
             logger.warning(f"Validation failed: {filename} contains null bytes, likely binary.")
             raise HTTPException(
                status_code=400,
                detail="Invalid file content. CSV file appears to be binary."
            )
            
        # Optional: Try decoding to catch non-text garbage if no nulls
        try:
            header.decode("utf-8")
        except UnicodeDecodeError:
            # If not UTF-8, it might be Latin-1/Windows-1252.
            # But since we already checked for nulls, we can be more lenient here 
            # or just accept it if we want to support legacy encodings.
            pass
