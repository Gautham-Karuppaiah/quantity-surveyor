import sqlite3
import time
import fitz
from main import render_page_bytes, pixmap_from_bytes


def test_render_speed(qtbot):
    doc = fitz.open("test.pdf")
    data = render_page_bytes(doc, 0)
    doc.close()
    assert len(data) > 0


def test_pixmap_from_bytes(qtbot):
    conn = sqlite3.connect("tests/test.db")
    (image,) = conn.execute("SELECT image FROM pages LIMIT 1").fetchone()
    conn.close()

    start = time.perf_counter()
    pixmap = pixmap_from_bytes(image)
    elapsed = time.perf_counter() - start
    print(f"from memory: {elapsed * 1000:.1f}ms")
    assert not pixmap.isNull()

    conn = sqlite3.connect("tests/test.db")
    start = time.perf_counter()
    (image,) = conn.execute("SELECT image FROM pages LIMIT 1").fetchone()
    pixmap = pixmap_from_bytes(image)
    elapsed = time.perf_counter() - start
    conn.close()
    print(f"from db: {elapsed * 1000:.1f}ms")
    assert not pixmap.isNull()
