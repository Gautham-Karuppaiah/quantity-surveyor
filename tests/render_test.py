import fitz
from main import render_page_pixmap


def test_render_speed(qtbot):
    doc = fitz.open("test.pdf")
    pixmap = render_page_pixmap(doc, 0)
    doc.close()
    assert not pixmap.isNull()
