import os
import tempfile
import pytest

typst = pytest.importorskip("typst")

UNICODE_TYPST_SNIPPET = """
#set text(font: ("Segoe UI", "Arial Unicode MS", "Segoe UI Symbol", "DejaVu Sans", "Arial"))

Tōkyō • Mária Ivanov • Kwame Nakamura • 東京 • دبي • תל אביב • मुंबई
"""


def test_typst_unicode_rendering():
    """Verify Typst compiles multilingual unicode characters without fatal error."""
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "test.typ")
        with open(p, "w", encoding="utf-8") as f:
            f.write(UNICODE_TYPST_SNIPPET)
        pdf_bytes = typst.compile(p)
        assert pdf_bytes is not None
        assert len(pdf_bytes) > 0
        assert pdf_bytes.startswith(b"%PDF")

        # Optional text extraction verification if PyMuPDF is installed
        try:
            import pymupdf

            doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            text = doc[0].get_text()
            assert len(text.strip()) > 0
            assert "\u25a0" not in text
        except ImportError:
            pass
