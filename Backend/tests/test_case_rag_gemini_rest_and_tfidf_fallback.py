import base64
import pytest
from app.core.rag_utils import TFIDFVectorStore, SimpleVectorStore
from app.services.rag_service import EnhancedRAGService
from app.services.report_styles import _decode_image, ReportMetadata
from app.services.report_weasyprint import _JINJA_ENV

def test_tfidf_vector_store_disk_serialization(tmp_path):
    """
    Verify TFIDFVectorStore saves and loads from disk cache without missing _embeddings_matrix error.
    """
    rag = EnhancedRAGService()
    task_id = "test_tfidf_task_123"
    
    store = TFIDFVectorStore()
    texts = [
        "Dataset contains 100 rows and 5 columns.",
        "High correlation between total_sales and operating_profit.",
        "Missing values detected in column price_per_unit."
    ]
    store.add_texts(texts)
    
    # Verify save does not raise AttributeError
    rag._save_local_vector_store(task_id, store)
    
    # Verify load returns valid store with text search capability
    loaded_store = rag._load_local_vector_store(task_id)
    assert loaded_store is not None
    assert len(loaded_store.documents) == 3
    
    results = loaded_store.similarity_search("total_sales profit", k=2)
    assert len(results) > 0
    assert "total_sales" in results[0][0]["content"]


def test_reportlab_decode_image_with_dict_and_svg():
    """
    Verify ReportLab _decode_image handles dictionary chart payloads, data URIs, and SVG bypass cleanly.
    """
    meta = ReportMetadata(filename="test.csv")
    
    # 1. 1x1 transparent PNG base64
    sample_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    # Dict payload (e.g., correlation_heatmap: {"image": "...", "narrative": "..."})
    chart_dict = {
        "image": sample_png_b64,
        "narrative": "Key correlation observed."
    }
    img = _decode_image(chart_dict, 100, 100, "Correlation Heatmap", meta)
    assert img is not None
    assert meta.charts_included == 1
    assert meta.charts_skipped == 0
    
    # 2. Raw SVG XML bypass
    svg_payload = "<svg viewBox='0 0 100 100'><rect width='100' height='100'/></svg>"
    img_svg = _decode_image(svg_payload, 100, 100, "Vector SVG", meta)
    assert img_svg is None
    assert meta.charts_skipped == 1


def test_weasyprint_jinja_domain_label_filter():
    """
    Verify domain_label Jinja2 filter is registered and formats domain titles correctly.
    """
    template = _JINJA_ENV.from_string("Domain: {{ domain | domain_label }}")
    rendered = template.render(domain="ecommerce_sales")
    assert rendered == "Domain: Ecommerce Sales"
    
    rendered_unknown = template.render(domain="Unknown")
    assert rendered_unknown == "Domain: General Business / Generic"
