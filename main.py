import os
import signal
import sys
from collections import deque

import cv2
import fitz
import numpy as np
from img2table.document import Image as TableImage
from img2table.ocr import EasyOCR
from sqlalchemy import create_engine, select, func, LargeBinary, ForeignKey, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, Session
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
)
from PyQt6.QtGui import QAction, QPixmap, QImage, QPen, QColor, QIcon, QImageReader
from PyQt6.QtCore import Qt, QRectF, QSize, QObject, QPointF, pyqtSignal, QTimer

os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

DPI = 200
WHITE_THRESHOLD = 240


class Base(DeclarativeBase):
    pass


class Folder(Base):
    __tablename__ = "folders"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]


class Drawing(Base):
    __tablename__ = "drawings"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    last_page: Mapped[int] = mapped_column(default=0)
    folder_id: Mapped[int | None] = mapped_column(ForeignKey("folders.id"), nullable=True)
    pages: Mapped[list["Page"]] = relationship(
        back_populates="drawing",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_page_index: int = 0


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(primary_key=True)
    drawing_id: Mapped[int] = mapped_column(ForeignKey("drawings.id"))
    page_number: Mapped[int]
    image_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    drawing: Mapped["Drawing"] = relationship(back_populates="pages")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pixmap: QPixmap | None = None
        self.undo_stack = deque(maxlen=100)
        self.redo_stack = deque()


class LegendEntry(Base):
    __tablename__ = "legend_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str]
    image_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    mask_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    auto_count: Mapped[bool] = mapped_column(default=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.pixmap: QPixmap | None = None


class ProjectState(Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True)
    last_opened_drawing_id: Mapped[int | None] = mapped_column(
        ForeignKey("drawings.id"), nullable=True
    )


def _make_engine(path: str):
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def set_pragmas(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys = ON")
        dbapi_conn.execute("PRAGMA journal_mode = WAL")

    return engine


def _init_project_state(engine):
    with Session(engine) as s:
        if not s.scalar(select(func.count()).select_from(ProjectState)):
            s.add(ProjectState())
            s.commit()


def remove_border(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    binary = (img.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8) * 255
    flood = binary.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        if flood[0, x] == 255:
            cv2.floodFill(flood, flood_mask, (x, 0), 128, flags=8)
        if flood[h - 1, x] == 255:
            cv2.floodFill(flood, flood_mask, (x, h - 1), 128, flags=8)
    for y in range(h):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, flood_mask, (0, y), 128, flags=8)
        if flood[y, w - 1] == 255:
            cv2.floodFill(flood, flood_mask, (w - 1, y), 128, flags=8)
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
    return img[rmin : rmax + 1, cmin : cmax + 1]


def qpixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    stride = qimg.bytesPerLine()
    ptr = qimg.bits()
    ptr.setsize(h * stride)
    arr = (
        np.frombuffer(ptr, dtype=np.uint8)
        .reshape((h, stride))[:, : w * 3]
        .reshape((h, w, 3))
        .copy()
    )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_qpixmap(arr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def render_page_bytes(doc: fitz.Document, page_index: int) -> bytes:
    page = doc[page_index]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def pixmap_from_bytes(data: bytes) -> QPixmap:
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap


def load_page(page: Page, session: Session) -> None:
    session.refresh(page, ["image_data"])
    page.pixmap = pixmap_from_bytes(page.image_data)


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
            entry = LegendEntry(label=label)
            entry.image = clean
            entry.mask = mask
            entry.pixmap = bgr_to_qpixmap(clean)
            entries.append(entry)
    return entries


class Task:
    def execute(self, project, session):
        raise NotImplementedError


class Command(Task):
    def undo(self, project, session):
        raise NotImplementedError


class AddLegendEntries(Task):
    def __init__(self, entries: list[LegendEntry]):
        self.entries = entries

    def execute(self, project, session):
        for entry in self.entries:
            _, img_buf = cv2.imencode(".png", entry.image)
            _, mask_buf = cv2.imencode(".png", entry.mask)
            entry.image_data = img_buf.tobytes()
            entry.mask_data = mask_buf.tobytes()
            session.add(entry)
        project.add_legend_entries(self.entries)


class LoadProject(Task):
    def execute(self, project, session):
        drawings = list(session.scalars(select(Drawing)))
        for drawing in drawings:
            drawing.last_page_index = drawing.last_page
            project.add_drawing(drawing)

        legend_entries = list(session.scalars(select(LegendEntry)))
        for entry in legend_entries:
            entry.image = cv2.imdecode(
                np.frombuffer(entry.image_data, np.uint8), cv2.IMREAD_COLOR
            )
            entry.mask = cv2.imdecode(
                np.frombuffer(entry.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            entry.pixmap = bgr_to_qpixmap(entry.image)
        if legend_entries:
            project.add_legend_entries(legend_entries)

        state = session.scalar(select(ProjectState))
        if not state or state.last_opened_drawing_id is None:
            return

        active_drawing = session.get(Drawing, state.last_opened_drawing_id)
        if active_drawing is None:
            return

        project.active_drawing = active_drawing
        last_page_index = min(active_drawing.last_page_index, len(active_drawing.pages) - 1)
        page = active_drawing.pages[last_page_index]
        load_page(page, session)
        project.active_page = page


class ImportDrawing(Task):
    def __init__(self, path: str):
        self.path = path

    def execute(self, project, session):
        doc = fitz.open(self.path)
        drawing = Drawing(name=os.path.basename(self.path))
        session.add(drawing)
        for i in range(len(doc)):
            page = Page(page_number=i, image_data=render_page_bytes(doc, i))
            drawing.pages.append(page)
        session.flush()
        doc.close()

        project.add_drawing(drawing)
        project.active_drawing = drawing
        if drawing.pages:
            first_page = drawing.pages[0]
            load_page(first_page, session)
            project.active_page = first_page


class Project(QObject):
    legend_entries_changed = pyqtSignal()
    active_page_changed = pyqtSignal()
    active_drawing_changed = pyqtSignal()
    drawings_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._legend_entries: list[LegendEntry] = []
        self._active_page: Page | None = None
        self._active_drawing: Drawing | None = None
        self.drawings: list[Drawing] = []

    @property
    def active_drawing(self):
        return self._active_drawing

    @active_drawing.setter
    def active_drawing(self, drawing: Drawing | None):
        self._active_drawing = drawing
        self.active_drawing_changed.emit()

    @property
    def legend_entries(self):
        return self._legend_entries

    @property
    def active_page(self):
        return self._active_page

    @active_page.setter
    def active_page(self, page: Page | None):
        self._active_page = page
        self.active_page_changed.emit()

    def add_drawing(self, drawing: Drawing):
        self.drawings.append(drawing)
        self.drawings_changed.emit()

    def add_legend_entries(self, entries: list[LegendEntry]):
        self._legend_entries.extend(entries)
        self.legend_entries_changed.emit()


class RectGesture:
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, on_complete):
        self._on_complete = on_complete
        self._anchor: QPointF | None = None
        self._canvas = None

    def activate(self, canvas):
        self._canvas = canvas

    def deactivate(self):
        self._anchor = None
        if self._canvas:
            self._canvas.clear_preview()
        self._canvas = None

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
        self._on_complete(rect)


def start_legend_select(controller):
    def on_complete(rect):
        crop = controller.canvas.crop(rect)
        if not crop.isNull():
            controller.dispatch(AddLegendEntries(extract_legend(crop)))
        controller.set_tool(None)

    controller.set_tool(RectGesture(on_complete=on_complete))


class AppController(QObject):
    def __init__(self, project: Project, session: Session | None = None):
        super().__init__()
        self._project = project
        self._session = session
        self._canvas = None
        self._autosave_timer = QTimer()
        self._autosave_timer.setInterval(30_000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        project.active_drawing_changed.connect(self._on_active_drawing_changed)

    @property
    def project(self) -> Project:
        return self._project

    @property
    def canvas(self):
        return self._canvas

    def set_canvas(self, canvas):
        self._canvas = canvas

    def set_session(self, session: Session):
        self._session = session

    def set_tool(self, tool):
        self._canvas.set_tool(tool)

    def dispatch(self, task: Task):
        if isinstance(task, Command):
            page = self._project.active_page
            task.execute(self._project, self._session)
            if page:
                page.undo_stack.append(task)
                page.redo_stack.clear()
        else:
            task.execute(self._project, self._session)

    def undo(self):
        page = self._project.active_page
        if page and page.undo_stack:
            cmd = page.undo_stack.pop()
            cmd.undo(self._project, self._session)
            page.redo_stack.append(cmd)

    def redo(self):
        page = self._project.active_page
        if page and page.redo_stack:
            cmd = page.redo_stack.pop()
            cmd.execute(self._project, self._session)
            page.undo_stack.append(cmd)

    def cancel_tool(self):
        self.set_tool(None)

    def shutdown(self):
        if self._session:
            self._session.commit()
            self._session.close()

    def _autosave(self):
        if self._session:
            self._session.commit()

    def _on_active_drawing_changed(self):
        if not self._session or not self._project.active_drawing:
            return
        state = self._session.scalar(select(ProjectState))
        if state:
            state.last_opened_drawing_id = self._project.active_drawing.id


class PDFViewer(QGraphicsView):
    escape_pressed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._page_item: QGraphicsPixmapItem | None = None
        self._rect_item: QGraphicsRectItem | None = None
        self._active_tool = None

    def set_tool(self, tool):
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


class LandingWindow(QMainWindow):
    def __init__(self, controller: "AppController"):
        super().__init__()
        self._controller = controller
        self.setWindowTitle("QS Automation")
        self.resize(400, 300)

        central = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("QS Automation")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        new_btn = QPushButton("New Project")
        open_btn = QPushButton("Open Project")
        new_btn.clicked.connect(self._new_project)
        open_btn.clicked.connect(self._open_project)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(new_btn)
        layout.addWidget(open_btn)
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _new_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "New Project", "", "QS Project (*.qsproj)"
        )
        if not path:
            return
        if not path.endswith(".qsproj"):
            os.remove(path)
            path = os.path.splitext(path)[0] + ".qsproj"
        engine = _make_engine(path)
        Base.metadata.create_all(engine)
        _init_project_state(engine)
        self._launch(Session(engine), path)

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "QS Project (*.qsproj)"
        )
        if not path:
            return
        engine = _make_engine(path)
        Base.metadata.create_all(engine)
        _init_project_state(engine)
        self._launch(Session(engine), path)

    def _launch(self, session, path):
        self._controller.set_session(session)
        self._main = MainWindow(self._controller, path)
        self._main.show()
        self._controller.dispatch(LoadProject())
        self.close()


class MainWindow(QMainWindow):
    def __init__(self, controller: "AppController", project_path: str):
        super().__init__()
        self._controller = controller
        self._project = controller.project

        self.setWindowTitle(f"QS Automation — {project_path}")
        self.resize(1200, 900)

        self.viewer = PDFViewer()
        self.setCentralWidget(self.viewer)
        self._controller.set_canvas(self.viewer)

        self.legend_panel = LegendPanel()
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.legend_panel)

        self._project.legend_entries_changed.connect(
            lambda: self.legend_panel.set_entries(self._project.legend_entries)
        )
        self._project.active_page_changed.connect(
            lambda: (
                self.viewer.load_pixmap(self._project.active_page.pixmap)
                if self._project.active_page
                else None
            )
        )
        self.viewer.escape_pressed.connect(self._controller.cancel_tool)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.legend_panel.toggleViewAction())

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        import_action = QAction("Import Drawing", self)
        import_action.triggered.connect(self._import_drawing)
        toolbar.addAction(import_action)

        legend_action = QAction("Load Legend", self)
        legend_action.triggered.connect(lambda: start_legend_select(self._controller))
        toolbar.addAction(legend_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._controller.undo)
        self.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._controller.redo)
        self.addAction(redo_action)

    def _import_drawing(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Drawing", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            self._controller.dispatch(ImportDrawing(path))
        except MemoryError:
            QMessageBox.critical(
                self, "Import Failed", "Not enough memory to import this drawing."
            )
        except OSError as e:
            QMessageBox.critical(self, "Import Failed", f"Disk error: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    QImageReader.setAllocationLimit(0)
    controller = AppController(Project())
    app.aboutToQuit.connect(controller.shutdown)
    window = LandingWindow(controller)
    window.show()
    sys.exit(app.exec())
