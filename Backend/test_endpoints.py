
from fastapi.testclient import TestClient
from app.main import app
from app.services.report_generator import ReportMetadata
import base64

client = TestClient(app)

def test_generate_report_endpoint():
    """
    Tests the /generate-report endpoint by sending a mock payload.
    """
    # 1. Prepare Mock Data
    filename = "test_data.csv"
    
    # Mock Analysis Results (including 'insights' which is key for the PDF)
    analysis_data = {
        "summary": {
            "col1": {"mean": 10.5, "max": 20},
            "col2": {"count": 100}
        },
        "insights": "These are mock AI insights for testing.",
        "metadata": {
            "total_rows": 100,
            "total_columns": 5
        }
    }
    
    # Mock Charts (dummy base64)
    # 1x1 pixel white transparent png
    dummy_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    charts_data = {
        "correlation_heatmap": dummy_b64,
        "distributions": [{"column": "col1", "image": dummy_b64}]
    }
    
    payload = {
        "filename": filename,
        "analysis": analysis_data,
        "charts": charts_data
    }
    
    print(f"Sending payload for {filename}...")
    
    # 2. Call Endpoint
    response = client.post("/api/generate-report", json=payload)
    
    # 3. Assertions
    if response.status_code != 200:
        print(f"Failed: {response.status_code} - {response.text}")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment; filename=" in response.headers["content-disposition"]
    
    # Check PDF content (magic bytes)
    content = response.content
    assert content.startswith(b"%PDF"), "Response is not a valid PDF"
    
    print(f"Success! Received PDF of size {len(content)} bytes.")

if __name__ == "__main__":
    # Manually run the test function if executed as script
    try:
        test_generate_report_endpoint()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
