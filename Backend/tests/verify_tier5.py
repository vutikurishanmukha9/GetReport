
import sys
import os
import polars as pl
from pprint import pprint

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.analysis import analyze_dataset
from app.services.analysis_config import AnalysisConfig

def create_dummy_data():
    return pl.DataFrame({
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # Perfect correlation with A
        "C": [1, 2, 3, 4, 5, 6, 100, 8, 9, 10], # Outlier 100, non-zero IQR
        "D": [None, None, None, 1, 2, 3, 4, 5, 6, 7] # Missing values
    })

def test_default_config():
    print("--- Testing Default Config ---")
    df = create_dummy_data()
    result = analyze_dataset(df)
    
    print(f"Ranked Insights Found: {len(result['ranked_insights'])}")
    for insight in result['ranked_insights']:
        print(f" - [{insight['score']}] {insight['type']}: {insight['description']}")
        
    assert len(result['ranked_insights']) > 0, "No insights found with default config"
    
    # Check for specific expected insights
    types = [i['type'] for i in result['ranked_insights']]
    assert 'correlation' in types, "Missing expected correlation insight"
    assert 'outlier' in types, "Missing expected outlier insight"

def test_custom_config():
    print("\n--- Testing Custom Config (No Correlation) ---")
    df = create_dummy_data()
    
    # Disable correlation
    config = AnalysisConfig(enable_correlation=False)
    result = analyze_dataset(df, config=config)
    
    # Check that correlation is empty
    assert result['correlation'] == {}, "Correlation should be empty"
    assert result['strong_correlations'] == [], "Strong correlations should be empty"
    
    # Check that correlation insights are missing
    types = [i['type'] for i in result['ranked_insights']]
    assert 'correlation' not in types, "Correlation insight found despite being disabled"
    print("Correlation correctly skipped.")

if __name__ == "__main__":
    try:
        test_default_config()
        test_custom_config()
        print("\n[PASS] Tier 5 Verification Passed!")
    except Exception as e:
        print(f"\n[FAIL] Tier 5 Verification Failed: {e}")
        import traceback
        traceback.print_exc()
