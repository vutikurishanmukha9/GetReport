import matplotlib.pyplot as plt
import re
import pytest
from app.services.visualization import _fig_to_svg

def test_vector_svg_responsive_viewbox_and_styles():
    """
    Verify generated vector SVG contains responsive viewBox and valid SVG elements:
    1. viewBox attribute present for scalable PDF layout.
    2. Zero lingering open figures.
    3. Proper XML closing tags and path definitions.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Cat A", "Cat B", "Cat C", "Cat D"]
    values = [45, 88, 23, 67]
    ax.bar(categories, values, color="#6366f1")
    ax.set_title("Category Distribution Test")
    
    svg_str = _fig_to_svg(fig)
    
    # 1. Check root element
    assert svg_str.startswith("<svg")
    assert svg_str.endswith("</svg>")
    
    # 2. Check for viewBox attribute
    assert "viewBox=" in svg_str
    
    # 3. Check for rendered graphic elements
    assert "<path" in svg_str or "<rect" in svg_str
    assert "Category Distribution Test" in svg_str
    
    # 4. Check that no figure leaks in matplotlib
    assert len(plt.get_fignums()) == 0
