import time
import os
import sys
import fitz
import numpy as np

PDF_PATH = "/home/bleedingdoughnut/Documents/panacar/BINGHATTI SKYRISE ACCESS CONTROL SYSTEM SCHEMATIC DIAGRAM REV-01.pdf"
PAGE_INDEX = 0
DPIS = [150, 300, 450, 600]


def render(doc, page_index, dpi):
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def main():
    doc = fitz.open(PDF_PATH)
    print(f"Page size: {doc[PAGE_INDEX].rect}")
    print(f"{'DPI':<6} {'Size (MB)':<12} {'Dims':<20} {'Time (s)':<10}")
    print("-" * 50)

    for dpi in DPIS:
        t0 = time.perf_counter()
        data = render(doc, PAGE_INDEX, dpi)
        elapsed = time.perf_counter() - t0

        arr = np.frombuffer(data, dtype=np.uint8)
        import cv2

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        raw_mb = (h * w * 3) / 1e6

        out = f"tests/dpi_{dpi}.png"
        with open(out, "wb") as f:
            f.write(data)
        print(f"{dpi:<6} {raw_mb:<12.1f} {f'{w}x{h}':<20} {elapsed:<10.2f}  -> {out}")

    doc.close()


if __name__ == "__main__":
    main()
