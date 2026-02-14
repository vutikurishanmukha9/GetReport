import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pytest
from fastapi import UploadFile, HTTPException

# Add Backend to path
sys.path.append(os.path.join(os.getcwd(), "Backend"))

from app.core.file_validation import validate_file_signature, SIGNATURES

class MockUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        self.position = 0
        
    async def seek(self, pos):
        self.position = pos
        
    async def read(self, size):
        data = self.content[self.position:self.position+size]
        self.position += size
        return data

async def test_validation():
    print("Testing File Validation...")
    
    # 1. Valid XLSX
    xlsx = MockUploadFile("test.xlsx", SIGNATURES["xlsx"] + b"payload")
    await validate_file_signature(xlsx)
    print("✓ Valid XLSX passed")
    
    # 2. Invalid XLSX (Text content)
    bad_xlsx = MockUploadFile("fake.xlsx", b"This is text not zip")
    try:
        await validate_file_signature(bad_xlsx)
        print("✗ Invalid XLSX failed to raise error")
    except HTTPException:
        print("✓ Invalid XLSX correctly raised error")
        
    # 3. Valid XLS
    xls = MockUploadFile("test.xls", SIGNATURES["xls"] + b"payload")
    await validate_file_signature(xls)
    print("✓ Valid XLS passed")
    
    # 4. Valid CSV (UTF-8)
    csv = MockUploadFile("data.csv", b"col1,col2\nval1,val2")
    await validate_file_signature(csv)
    print("✓ Valid CSV passed")
    
    # 5. Invalid CSV (Binary nonsense)
    bad_csv = MockUploadFile("bad.csv", b"\x00\x01\x02\x03")
    try:
        await validate_file_signature(bad_csv)
        print("✗ Binary CSV failed to raise error")
    except HTTPException:
        print("✓ Binary CSV correctly raised error")

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_validation())
    loop.close()
