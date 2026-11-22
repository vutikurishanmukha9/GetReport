import typst
import tempfile
import os
import pymupdf

code = """
#set text(font: ("Segoe UI", "Arial Unicode MS", "Segoe UI Symbol", "DejaVu Sans", "Arial"))

Tōkyō • Mária Ivanov • Kwame Nakamura • 東京 • دبي • תל אביב • मुंबई
"""

with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "test.typ")
    with open(p, "w", encoding="utf-8") as f:
        f.write(code)
    pdf_bytes = typst.compile(p)
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    text = doc[0].get_text()
    print("Compiled size:", len(pdf_bytes))
    print("Extracted text:", text.strip())
    print("Contains black square ■:", "\u25a0" in text)
