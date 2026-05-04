import os
import signal
import sys
from collections import deque
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
from PyQt6.QtCore import Qt, QRectF, QSize, QObject, QPointF, pyqtSignal

os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

DPI = 300
WHITE_THRESHOLD = 240


class Page:
    def __init__(self, id: int, pixmap: QPixmap | None = None):
        self.id = id
        self.pixmap = pixmap
        self.undo_stack = deque(maxlen=100)
        self.redo_stack = deque()


class Drawing:
    def __init__(self, id: int, filename: str, folder_id: int | None = None):
        self.id = id
        self.filename = filename
        self.folder_id = folder_id
        self.pages: list[Page] = []
        self.last_page_index: int = 0


@dataclass
class LegendEntry:
    label: str
    image: np.ndarray
    mask: np.ndarray
    pixmap: QPixmap
    auto_count: bool = True


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


def qpixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    stride = qimg.bytesPerLine()
    ptr = qimg.bits()
    ptr.setsize(h * stride)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, stride))[:, :w * 3].reshape((h, w, 3)).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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


def extract_legend(pixmap: QPixmap) -> list[LegendEntry]:
    bgr = qpixmap_to_bgr(pixmap)
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
            entry_pixmap = bgr_to_qpixmap(clean)
            entries.append(LegendEntry(label=label, image=clean, mask=mask, pixmap=entry_pixmap))
    return entries


class Task:
    def execute(self, project): raise NotImplementedError


class Command(Task):
    def undo(self, project): raise NotImplementedError


class AddLegendEntries(Task):
    def __init__(self, entries: list[LegendEntry]):
        self.entries = entries

    def execute(self, project):
        project.add_legend_entries(self.entries)



class Project(QObject):
    legend_entries_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._legend_entries: list[LegendEntry] = []

    @property
    def legend_entries(self):
        return self._legend_entries

    def add_legend_entries(self, entries: list[LegendEntry]):
        self._legend_entries.extend(entries)
        self.legend_entries_changed.emit()



class Tool(QObject):
    task_ready = pyqtSignal(object)
    done = pyqtSignal()
    cursor: Qt.CursorShape | None = None
    persistent: bool = False

    def __init__(self):
        super().__init__()
        self._canvas = None

    def activate(self, canvas):
        self._canvas = canvas

    def deactivate(self):
        self._canvas = None

    def on_task_emitted(self):
        if not self.persistent:
            self.done.emit()

    def on_press(self, pos: QPointF): pass
    def on_move(self, pos: QPointF): pass
    def on_release(self, pos: QPointF): pass



class RectSelectTool(Tool):
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self):
        super().__init__()
        self._anchor: QPointF | None = None

    def deactivate(self):
        self._anchor = None
        if self._canvas:
            self._canvas.clear_preview()
        super().deactivate()

    def on_press(self, pos: QPointF):
        self._anchor = pos

    def on_move(self, pos: QPointF):
        if self._anchor is None:
            return
        self._canvas.show_rect_preview(QRectF(self._anchor, pos).normalized())

    def on_release(self, pos: QPointF):
        if self._anchor is None:
            return
        rect = QRectF(self._anchor, pos).normalized()
        self._anchor = None
        self._canvas.clear_preview()
        self.on_complete(rect)

    def on_complete(self, rect: QRectF):
        pass


class LegendSelectTool(RectSelectTool):
    def on_complete(self, rect: QRectF):
        crop = self._canvas.crop(rect)
        if not crop.isNull():
            self.task_ready.emit(AddLegendEntries(extract_legend(crop)))
            self.on_task_emitted()


class AppController(QObject):
    def __init__(self, project: Project):
        super().__init__()
        self._project = project
        self._undo: list[Command] = []
        self._redo: list[Command] = []
        self._canvas = None

    def set_canvas(self, canvas):
        self._canvas = canvas

    def set_tool(self, tool: Tool | None):
        self._canvas.set_tool(tool)
        if tool:
            tool.task_ready.connect(self._on_task_ready)
            tool.done.connect(lambda: self.set_tool(None))

    def execute(self, cmd: Command):
        cmd.execute(self._project)
        self._undo.append(cmd)
        self._redo.clear()

    def execute_no_undo(self, action: Task):
        action.execute(self._project)

    def undo(self):
        if self._undo:
            cmd = self._undo.pop()
            cmd.undo(self._project)
            self._redo.append(cmd)

    def redo(self):
        if self._redo:
            cmd = self._redo.pop()
            cmd.execute(self._project)
            self._undo.append(cmd)

    def cancel_tool(self):
        self.set_tool(None)

    def open_pdf(self, path: str):
        self._canvas.load_pixmap(render_page(path))

    def _on_task_ready(self, action: Task):
        if isinstance(action, Command):
            self.execute(action)
        else:
            self.execute_no_undo(action)


class PDFViewer(QGraphicsView):
    escape_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._page_item: QGraphicsPixmapItem | None = None
        self._rect_item: QGraphicsRectItem | None = None
        self._active_tool: Tool | None = None

    def set_tool(self, tool: Tool | None):
        if self._active_tool:
            self._active_tool.deactivate()
        self._active_tool = tool
        if tool:
            tool.activate(self)
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if tool.cursor is not None:
                self.setCursor(tool.cursor)
            else:
                self.unsetCursor()
        else:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.unsetCursor()

    def load_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._rect_item = None
        self._page_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._page_item)
        self.fitInView(self._page_item, Qt.AspectRatioMode.KeepAspectRatio)

    def crop(self, rect: QRectF) -> QPixmap:
        if self._page_item is None:
            return QPixmap()
        return self._page_item.pixmap().copy(rect.toRect())

    def show_rect_preview(self, rect: QRectF):
        if self._rect_item is None:
            pen = QPen(QColor(0, 120, 255))
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(pen)
            self._rect_item.setBrush(QColor(0, 120, 255, 30))
            self._scene.addItem(self._rect_item)
        self._rect_item.setRect(rect)

    def clear_preview(self):
        if self._rect_item:
            self._scene.removeItem(self._rect_item)
            self._rect_item = None

    def zoom(self, delta: int):
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            self._active_tool.on_press(self.mapToScene(event.pos()))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._active_tool:
            self._active_tool.on_move(self.mapToScene(event.pos()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            self._active_tool.on_release(self.mapToScene(event.pos()))
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.zoom(event.angleDelta().y())


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
        self._project = Project()
        self._controller = AppController(self._project)

        self.setWindowTitle("QS Automation")
        self.resize(1200, 900)

        self.viewer = PDFViewer()
        self.setCentralWidget(self.viewer)
        self._controller.set_canvas(self.viewer)

        self.legend_panel = LegendPanel()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.legend_panel)

        self._project.legend_entries_changed.connect(
            lambda: self.legend_panel.set_entries(self._project.legend_entries)
        )
        self.viewer.escape_pressed.connect(self._controller.cancel_tool)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.legend_panel.toggleViewAction())

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        open_action = QAction("Open PDF", self)
        open_action.triggered.connect(self._open_pdf)
        toolbar.addAction(open_action)

        legend_action = QAction("Load Legend", self)
        legend_action.triggered.connect(lambda: self._controller.set_tool(LegendSelectTool()))
        toolbar.addAction(legend_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._controller.undo)
        self.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._controller.redo)
        self.addAction(redo_action)

    def _open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF Files (*.pdf)")
        if not path:
            return
        self._controller.open_pdf(path)
        self.setWindowTitle(f"QS Automation — {path}")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
