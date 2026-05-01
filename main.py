import os
import signal
import sys
from dataclasses import dataclass

import cv2
import fitz
import numpy as np
from img2table.document import Image as TableImage
from img2table.ocr import EasyOCR
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QDockWidget, QListWidget, QListWidgetItem,
)
from PyQt6.QtGui import QAction, QPixmap, QImage, QPen, QColor, QIcon
from PyQt6.QtCore import Qt, QRectF, QSize

os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

DPI = 300
WHITE_THRESHOLD = 240


@dataclass
class LegendEntry:
    label: str
    image: np.ndarray
    mask: np.ndarray
    pixmap: QPixmap


def remove_border(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    binary = (img.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8) * 255
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if flood[0, x] == 255: cv2.floodFill(flood, flood_mask, (x, 0), 128, flags=8)
        if flood[h - 1, x] == 255: cv2.floodFill(flood, flood_mask, (x, h - 1), 128, flags=8)
    for y in range(h):
        if flood[y, 0] == 255: cv2.floodFill(flood, flood_mask, (0, y), 128, flags=8)
        if flood[y, w - 1] == 255: cv2.floodFill(flood, flood_mask, (w - 1, y), 128, flags=8)
    result = img.copy()
    result[flood == 128] = 255
    return result


def tight_crop(img: np.ndarray, pad: int = 4) -> np.ndarray:
    mask = img.min(axis=2) < WHITE_THRESHOLD
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = img.shape[:2]
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)
    return img[rmin:rmax + 1, cmin:cmax + 1]


def bgr_to_qpixmap(arr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def render_page(path: str, page_index: int = 0) -> QPixmap:
    doc = fitz.open(path)
    page = doc[page_index]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888).copy()
    doc.close()
    return QPixmap.fromImage(img)


class RectSelectTool:
    def __init__(self, viewer, callback):
        self._viewer = viewer
        self._callback = callback
        self._anchor = None
        self._rect_item = None

    def activate(self):
        self._viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._viewer.setCursor(Qt.CursorShape.CrossCursor)

    def deactivate(self):
        self._viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._viewer.unsetCursor()
        if self._rect_item:
            self._viewer.remove_scene_item(self._rect_item)
            self._rect_item = None
        self._anchor = None

    def handle(self, event_type, event):
        match event_type:
            case "press":
                self._on_press(event)
            case "move":
                self._on_move(event)
            case "release":
                self._on_release(event)

    def _on_press(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._anchor = self._viewer.mapToScene(event.pos())
        pen = QPen(QColor(0, 120, 255))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        self._rect_item = QGraphicsRectItem()
        self._rect_item.setPen(pen)
        self._rect_item.setBrush(QColor(0, 120, 255, 30))
        self._viewer.add_scene_item(self._rect_item)

    def _on_move(self, event):
        if not self._anchor or not self._rect_item:
            return
        current = self._viewer.mapToScene(event.pos())
        self._rect_item.setRect(QRectF(self._anchor, current).normalized())

    def _on_release(self, event):
        if not self._anchor or event.button() != Qt.MouseButton.LeftButton:
            return
        current = self._viewer.mapToScene(event.pos())
        rect = QRectF(self._anchor, current).normalized()
        callback = self._callback
        self._viewer.set_tool(None)
        callback(rect)


class PDFViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._item = None
        self._tool = None

    def set_tool(self, tool):
        if self._tool:
            self._tool.deactivate()
        self._tool = tool
        if tool:
            tool.activate()

    def add_scene_item(self, item):
        self._scene.addItem(item)

    def remove_scene_item(self, item):
        self._scene.removeItem(item)

    def load_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._item)
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self._tool:
            self.set_tool(None)
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if self._tool:
            self._tool.handle("press", event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._tool:
            self._tool.handle("move", event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._tool:
            self._tool.handle("release", event)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._tool:
            self._tool.handle("wheel", event)
        else:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)


class LegendList(QListWidget):
    BASE_SIZE = 80

    def __init__(self):
        super().__init__()
        self._icon_size = self.BASE_SIZE
        self.setIconSize(QSize(self._icon_size, self._icon_size))
        self.setWordWrap(True)
        font = self.font()
        font.setPointSize(15)
        self.setFont(font)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.1 if event.angleDelta().y() > 0 else 1 / 1.1
            self._icon_size = max(40, min(600, round(self._icon_size * factor)))
            self.setIconSize(QSize(self._icon_size, self._icon_size))
        else:
            super().wheelEvent(event)


class LegendPanel(QDockWidget):
    def __init__(self):
        super().__init__("Legend")
        self._list = LegendList()
        self.setWidget(self._list)

    def set_entries(self, entries: list[LegendEntry]):
        self._list.clear()
        for entry in entries:
            item = QListWidgetItem(QIcon(entry.pixmap), entry.label)
            self._list.addItem(item)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QS Automation")
        self.resize(1200, 900)

        self.viewer = PDFViewer()
        self.setCentralWidget(self.viewer)

        self.legend_panel = LegendPanel()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.legend_panel)

        self._legend_entries: list[LegendEntry] = []

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.legend_panel.toggleViewAction())

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self.open_pdf)
        toolbar.addAction(open_action)

        legend_action = QAction("Load Legend", self)
        legend_action.triggered.connect(self.load_legend)
        toolbar.addAction(legend_action)

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path:
            return
        pixmap = render_page(path)
        self.viewer.load_pixmap(pixmap)
        self.setWindowTitle(f"QS Automation — {path}")

    def load_legend(self):
        self.viewer.set_tool(RectSelectTool(self.viewer, self.on_legend_rect))

    def on_legend_rect(self, rect: QRectF):
        if not self.viewer._item:
            return

        crop = self.viewer._item.pixmap().copy(rect.toRect())
        qimg = crop.toImage().convertToFormat(QImage.Format.Format_RGB888)
        cw, ch = qimg.width(), qimg.height()
        stride = qimg.bytesPerLine()
        ptr = qimg.bits()
        ptr.setsize(ch * stride)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((ch, stride))[:, :cw * 3].reshape((ch, cw, 3)).copy()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        _, buf = cv2.imencode(".png", bgr)
        ocr = EasyOCR(lang=["en"])
        doc = TableImage(src=buf.tobytes())
        tables = doc.extract_tables(ocr=ocr)

        entries = []
        for table in tables:
            for row in table.content.values():
                if len(row) < 2:
                    continue
                symbol_cell, label_cell = row[0], row[1]
                label = " ".join((label_cell.value or "").split())
                bx1, by1 = symbol_cell.bbox.x1, symbol_cell.bbox.y1
                bx2, by2 = symbol_cell.bbox.x2, symbol_cell.bbox.y2
                cell_bgr = bgr[by1:by2, bx1:bx2]
                if cell_bgr.size == 0:
                    continue
                clean = tight_crop(remove_border(cell_bgr))
                mask = (clean.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8) * 255
                pixmap = bgr_to_qpixmap(clean)
                entries.append(LegendEntry(label=label, image=clean, mask=mask, pixmap=pixmap))

        self._legend_entries.extend(entries)
        self.legend_panel.set_entries(self._legend_entries)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
