import matplotlib.pyplot as plt
import jinja2
from pathlib import Path
from app.services.visualization import _fig_to_svg, _fig_to_base64
from app.services.report_weasyprint import _JINJA_ENV

def test_vector_svg_chart_generation_and_template_rendering():
    """
    Verify Phase 3 Vector SVG chart generation:
    1. _fig_to_svg outputs clean inline SVG XML.
    2. Figures are closed with 0 lingering figures.
    3. Jinja2 report.html template properly embeds inline SVGs.
    """
    # 1. Create a sample figure and convert to SVG
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([1, 2, 3], [4, 5, 6], label="Trend Line")
    ax.set_title("Test Vector SVG")
    ax.legend()
    
    svg_xml = _fig_to_svg(fig)
    assert svg_xml.startswith("<svg")
    assert "</svg>" in svg_xml
    assert "Test Vector SVG" in svg_xml
    assert len(plt.get_fignums()) == 0
    
    # 2. Render Jinja2 template with SVG chart
    template = _JINJA_ENV.get_template("report.html")
    rendered_html = template.render(
        filename="test_dataset.csv",
        generated_at="August 29, 2026",
        metadata={"total_rows": 100, "total_columns": 5, "numeric_columns": 3, "categorical_columns": 2},
        analysis={"summary": {}},
        charts={
            "correlation_heatmap": {
                "image": svg_xml,
                "narrative": "Strong positive correlation observed between features."
            }
        }
    )
    
    assert "chart-card__svg-wrapper" in rendered_html
    assert "<svg" in rendered_html
    assert "Strong positive correlation observed" in rendered_html
