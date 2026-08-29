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
    # Apache Parquet
    "parquet": b"PAR1",
    # Apache Arrow / Feather
    "feather": b"ARROW1",
    # GZIP compressed file
    "gz": b"\x1f\x8b",
}

def verify_header_bytes(header: bytes, filename: str) -> None:
    """
    Synchronously validate raw initial bytes match the extension signature.
    Raises HTTPException(400) on mismatch.
    """
    fn_lower = filename.lower()
    
    # 1. Excel (XLSX)
    if fn_lower.endswith(".xlsx"):
        if not header.startswith(SIGNATURES["xlsx"]):
            logger.warning(f"Validation failed: {filename} claims to be XLSX but lacks ZIP signature.")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file content. Extension says .xlsx but content does not match (ZIP signature missing)."
            )

    # 2. Excel 97-2003 (XLS)
    elif fn_lower.endswith(".xls"):
        if not header.startswith(SIGNATURES["xls"]):
            logger.warning(f"Validation failed: {filename} claims to be XLS but lacks OLE2 signature.")
            raise HTTPException(
                status_code=400, 
                detail="Invalid file content. Extension says .xls but content does not match (OLE2 signature missing)."
            )

    # 3. Parquet
    elif fn_lower.endswith(".parquet"):
        if not header.startswith(SIGNATURES["parquet"]):
            logger.warning(f"Validation failed: {filename} claims to be Parquet but lacks PAR1 magic number.")
            raise HTTPException(
                status_code=400,
                detail="Invalid file content. Parquet header signature missing."
            )

    # 4. Feather / Arrow
    elif fn_lower.endswith((".feather", ".arrow")):
        if not header.startswith(SIGNATURES["feather"]):
            logger.warning(f"Validation failed: {filename} claims to be Feather/Arrow but lacks ARROW1 signature.")
            raise HTTPException(
                status_code=400,
                detail="Invalid file content. Feather/Arrow header signature missing."
            )

    # 5. GZIP
    elif fn_lower.endswith(".gz"):
        if not header.startswith(SIGNATURES["gz"]):
            logger.warning(f"Validation failed: {filename} claims to be GZIP but lacks GZIP magic number.")
            raise HTTPException(
                status_code=400,
                detail="Invalid file content. GZIP compression header signature missing."
            )

    # 6. CSV (Text-based)
    elif fn_lower.endswith(".csv"):
        is_text = False
        for encoding in ["utf-8", "utf-16", "latin-1"]:
            try:
                header.decode(encoding)
                is_text = True
                break
            except UnicodeError:
                continue
                
        if not is_text and b"\x00" in header:
            logger.warning(f"Validation failed: {filename} contains null bytes and failed text decoding.")
            raise HTTPException(
                status_code=400,
                detail="Invalid file content. CSV file appears to be binary."
            )


async def validate_file_signature(file: UploadFile) -> None:
    """
    Validate file content matches its extension using magic numbers.
    Raises HTTPException(400) if invalid.
    Resets file pointer to 0 after checking.
    """
    # Read start of file
    await file.seek(0)
    header = await file.read(8)
    await file.seek(0)  # Reset immediately
    
    verify_header_bytes(header, file.filename)
