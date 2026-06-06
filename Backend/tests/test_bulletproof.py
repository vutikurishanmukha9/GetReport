import os
import tempfile
import zipfile
import polars as pl
import pytest

from app.services.data_processing import (
    clean_data,
    load_dataframe,
    _validate_zip_bomb,
    ParseError
)
from app.core.config import settings

def test_winsorization_outliers():
    # Create a dataset with values [10, 12, 11, 10, 12, 110] (110 is outlier)
    # Median is 11.0. Lower/upper bounds for IQR are calculated.
    # q1 = 10.0, q3 = 12.0, iqr = 2.0.
    # Lower bound = 10 - 1.5 * 2 = 7.0
    # Upper bound = 12 + 1.5 * 2 = 15.0
    # The outlier 110 should be capped to 15.0 (Winsorization) instead of median 11.0
    
    df = pl.DataFrame({
        "val": [10.0, 12.0, 11.0, 10.0, 12.0, 110.0]
    })
    
    rules = {
        "val": {
            "action": "replace_outliers_median"
        }
    }
    
    cleaned_df, report, dag = clean_data(df, rules)
    
    # Assert Winsorization capped the outlier at upper_bound (15.0) instead of median (11.0)
    # The columns are converted to snake_case, so "val" remains "val"
    values = cleaned_df["val"].to_list()
    assert 110.0 not in values
    assert 11.0 in values
    # Outlier replaced by upper bound (15.0)
    assert 15.0 in values
    # Median was 11.0. If it were replaced by median, there would be no 15.0 in the output, and 11.0 would appear more times.
    assert values.count(11.0) == 1 # original count of 11.0 was 1, should remain 1

def test_validate_zip_bomb_healthy():
    # Test a healthy zip file passes validation
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        with zipfile.ZipFile(tmp_name, 'w') as zf:
            zf.writestr("sheet1.xml", "<worksheet><row><c>1</c></row></worksheet>")
            zf.writestr("workbook.xml", "<workbook></workbook>")
            
        # Should not raise any exceptions
        _validate_zip_bomb(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_validate_zip_bomb_too_many_files():
    # Test a zip file with more than 5000 files triggers Zip Bomb ParseError
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        with zipfile.ZipFile(tmp_name, 'w') as zf:
            # Add 5001 tiny dummy files
            for i in range(5002):
                zf.writestr(f"file_{i}.xml", "a")
                
        with pytest.raises(ParseError) as exc:
            _validate_zip_bomb(tmp_name)
        assert "Too many files" in str(exc.value)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_validate_zip_bomb_size_exceeded():
    # Test a zip file that decompresses beyond settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        # Temporarily lower the max size to 1MB for testing
        original_max_size = settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB
        settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB = 1
        
        with zipfile.ZipFile(tmp_name, 'w') as zf:
            # Write a 2MB uncompressed file (highly compressed)
            zf.writestr("huge.xml", "a" * (2 * 1024 * 1024))
            
        with pytest.raises(ParseError) as exc:
            _validate_zip_bomb(tmp_name)
        assert "Decompressed size" in str(exc.value)
    finally:
        settings.MAX_EXCEL_DECOMPRESSED_SIZE_MB = original_max_size
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_validate_zip_bomb_compression_ratio():
    # Test high compression ratio triggers Zip Bomb warning if size > 10MB
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_name = tmp.name
        
    try:
        with zipfile.ZipFile(tmp_name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Write a 12MB uncompressed file consisting of zeros (extremely compressible)
            zf.writestr("huge_empty.xml", "0" * (12 * 1024 * 1024))
            
        with pytest.raises(ParseError) as exc:
            # Lower ratio limit to make sure it trips
            _validate_zip_bomb(tmp_name, max_ratio=20.0)
        assert "compression ratio" in str(exc.value).lower()
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
