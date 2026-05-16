import json
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
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
    make_transient,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QToolBar,
    QFileDialog,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsPolygonItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QProgressBar,
    QInputDialog,
)
from PyQt6.QtGui import (
    QAction,
    QPixmap,
    QImage,
    QPen,
    QColor,
    QIcon,
    QImageReader,
    QPolygonF,
    QPainterPath,
)
from PyQt6.QtCore import (
    Qt,
    QRectF,
    QSize,
    QObject,
    QPointF,
    pyqtSignal,
    QTimer,
    QThread,
)

os.environ["QT_QPA_PLATFORMTHEME"] = "xdgdesktopportal"

DPI = 600
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
    folder_id: Mapped[int | None] = mapped_column(
        ForeignKey("folders.id"), nullable=True
    )
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
    markers: Mapped[list["Marker"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    zones: Mapped[list["Zone"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )

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
    markers: Mapped[list["Marker"]] = relationship(
        back_populates="legend_entry", cascade="all"
    )
    samples: Mapped[list["Sample"]] = relationship(
        back_populates="legend_entry", cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.pixmap: QPixmap | None = None
        self.dirty: bool = True


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(primary_key=True)
    legend_entry_id: Mapped[int] = mapped_column(ForeignKey("legend_entries.id"))
    image_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    mask_data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    legend_entry: Mapped["LegendEntry"] = relationship(back_populates="samples")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.pixmap: QPixmap | None = None


class Marker(Base):
    __tablename__ = "markers"

    id: Mapped[int] = mapped_column(primary_key=True)
    legend_entry_id: Mapped[int] = mapped_column(ForeignKey("legend_entries.id"))
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"))
    x: Mapped[int]
    y: Mapped[int]
    w: Mapped[int]
    h: Mapped[int]
    score: Mapped[float | None] = mapped_column(nullable=True)
    source: Mapped[str] = mapped_column(default="manual")
    page: Mapped["Page"] = relationship(back_populates="markers")
    legend_entry: Mapped["LegendEntry"] = relationship(back_populates="markers")


class ProjectState(Base):
    __tablename__ = "project"

    id: Mapped[int] = mapped_column(primary_key=True)
    last_opened_drawing_id: Mapped[int | None] = mapped_column(
        ForeignKey("drawings.id"), nullable=True
    )


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id"))
    name: Mapped[str]
    geometry: Mapped[str]
    page: Mapped["Page"] = relationship(back_populates="zones")

    @property
    def points(self) -> list:
        return json.loads(self.geometry)

    @points.setter
    def points(self, value: list):
        self.geometry = json.dumps(value)


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
    fitz.TOOLS.set_aa_level(8)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = (
        np.frombuffer(pix.samples, dtype=np.uint8)
        .reshape(pix.height, pix.width, 3)
        .copy()
    )
    img = np.clip(255 - (255 - img.astype(np.int32)) * 3, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def pixmap_from_bytes(data: bytes) -> QPixmap:
    pixmap = QPixmap()
    pixmap.loadFromData(data)
    return pixmap


def load_page(page: Page, session: Session) -> None:
    session.refresh(page, ["image_data"])
    page.pixmap = pixmap_from_bytes(page.image_data)
    _ = page.zones  # eagerly load zones into the session identity map


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


class SetLegendSample(Task):
    def __init__(self, entry: LegendEntry, image_bgr: np.ndarray):
        self._entry = entry
        self._image_bgr = image_bgr

    def execute(self, project, session):
        clean = tight_crop(self._image_bgr)
        mask = (clean.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8) * 255
        _, img_buf = cv2.imencode(".png", clean)
        _, mask_buf = cv2.imencode(".png", mask)
        sample = Sample(
            legend_entry_id=self._entry.id,
            image_data=img_buf.tobytes(),
            mask_data=mask_buf.tobytes(),
        )
        sample.image = clean
        sample.mask = mask
        sample.pixmap = bgr_to_qpixmap(clean)
        session.add(sample)
        session.flush()
        self._entry.samples.append(sample)
        self._entry.dirty = True
        project.set_legend_entry_sample(self._entry)


class AddLegendEntryFromSample(Task):
    def __init__(self, label: str, image_bgr: np.ndarray):
        self._label = label
        self._image_bgr = image_bgr

    def execute(self, project, session):
        clean = tight_crop(self._image_bgr)
        mask = (clean.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8) * 255
        _, img_buf = cv2.imencode(".png", clean)
        _, mask_buf = cv2.imencode(".png", mask)
        entry = LegendEntry(
            label=self._label,
            image_data=img_buf.tobytes(),
            mask_data=mask_buf.tobytes(),
            auto_count=True,
        )
        entry.image = clean
        entry.mask = mask
        entry.pixmap = bgr_to_qpixmap(clean)
        entry.dirty = True
        sample = Sample(image_data=img_buf.tobytes(), mask_data=mask_buf.tobytes())
        sample.image = clean
        sample.mask = mask
        sample.pixmap = bgr_to_qpixmap(clean)
        entry.samples.append(sample)
        session.add(entry)
        session.flush()
        project.add_legend_entries([entry])


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
            entry.dirty = True
            for sample in entry.samples:
                sample.image = cv2.imdecode(
                    np.frombuffer(sample.image_data, np.uint8), cv2.IMREAD_COLOR
                )
                sample.mask = cv2.imdecode(
                    np.frombuffer(sample.mask_data, np.uint8), cv2.IMREAD_GRAYSCALE
                )
                sample.pixmap = bgr_to_qpixmap(sample.image)
        if legend_entries:
            project.add_legend_entries(legend_entries)

        state = session.scalar(select(ProjectState))
        if not state or state.last_opened_drawing_id is None:
            return

        active_drawing = session.get(Drawing, state.last_opened_drawing_id)
        if active_drawing is None:
            return

        project.active_drawing = active_drawing
        last_page_index = min(
            active_drawing.last_page_index, len(active_drawing.pages) - 1
        )
        page = active_drawing.pages[last_page_index]
        load_page(page, session)
        project.active_page = page


class ImportDrawing(Task):
    def __init__(self, path: str):
        self.path = path
        self._progress_cb = None

    def execute(self, project, session):
        doc = fitz.open(self.path)
        total = len(doc)
        drawing = Drawing(name=os.path.basename(self.path))
        session.add(drawing)
        for i in range(total):
            page = Page(page_number=i, image_data=render_page_bytes(doc, i))
            drawing.pages.append(page)
            if self._progress_cb:
                self._progress_cb(i + 1, total)
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
    markers_changed = pyqtSignal()
    zones_changed = pyqtSignal()

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

    def set_legend_entry_sample(self, entry: LegendEntry):
        self.legend_entries_changed.emit()

    def notify_markers_changed(self):
        self.markers_changed.emit()

    def notify_zones_changed(self):
        self.zones_changed.emit()


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


def _entry_color(legend_entry_id: int) -> QColor:
    hue = (legend_entry_id * 0.618033988749895) % 1.0
    return QColor.fromHsvF(hue, 0.9, 0.85)


class AddMarker(Command):
    def __init__(self, entry: LegendEntry, page: Page, x: int, y: int, w: int, h: int):
        self._entry = entry
        self._page = page
        self._marker = Marker(
            legend_entry_id=entry.id, page_id=page.id, x=x, y=y, w=w, h=h
        )

    def execute(self, project, session):
        self._page.markers.append(self._marker)
        session.add(self._marker)
        session.flush()
        project.notify_markers_changed()

    def undo(self, project, session):
        self._page.markers.remove(self._marker)
        session.delete(self._marker)
        project.notify_markers_changed()


class DeleteMarker(Command):
    def __init__(self, marker: Marker, page: Page):
        self._marker = marker
        self._page = page

    def execute(self, project, session):
        self._page.markers.remove(self._marker)
        session.delete(self._marker)
        project.notify_markers_changed()

    def undo(self, project, session):
        make_transient(self._marker)
        self._marker.id = None
        self._page.markers.append(self._marker)
        session.add(self._marker)
        session.flush()
        project.notify_markers_changed()


class AddZone(Command):
    def __init__(self, page: Page, name: str, points: list):
        self._page = page
        self._zone = Zone(page_id=page.id, name=name, geometry=json.dumps(points))

    def execute(self, project, session):
        self._page.zones.append(self._zone)
        session.add(self._zone)
        session.flush()
        project.notify_zones_changed()

    def undo(self, project, session):
        self._page.zones.remove(self._zone)
        session.delete(self._zone)
        project.notify_zones_changed()


class DeleteZone(Command):
    def __init__(self, zone: Zone, page: Page):
        self._zone = zone
        self._page = page

    def execute(self, project, session):
        self._page.zones.remove(self._zone)
        session.delete(self._zone)
        project.notify_zones_changed()

    def undo(self, project, session):
        make_transient(self._zone)
        self._zone.id = None
        self._page.zones.append(self._zone)
        session.add(self._zone)
        session.flush()
        project.notify_zones_changed()


class PolygonGesture:
    cursor = Qt.CursorShape.CrossCursor

    def __init__(self, on_complete):
        self._on_complete = on_complete
        self._points: list[QPointF] = []
        self._canvas = None

    @property
    def is_started(self):
        return len(self._points) > 0

    def activate(self, canvas):
        self._canvas = canvas
        self._points = []

    def deactivate(self):
        if self._canvas:
            self._canvas.clear_polygon_preview()
        self._canvas = None

    def on_press(self, pos: QPointF):
        self._points.append(pos)
        self._canvas.update_polygon_preview(self._points, pos)

    def on_move(self, pos: QPointF):
        if self._points:
            self._canvas.update_polygon_preview(self._points, pos)

    def on_release(self, pos: QPointF):
        pass

    def on_double_click(self, pos: QPointF):
        if self._points:
            self._points.pop()  # remove duplicate from preceding press event
        if len(self._points) >= 3:
            self._canvas.clear_polygon_preview()
            self._on_complete([[p.x(), p.y()] for p in self._points])

    def on_key_return(self):
        if len(self._points) >= 3:
            self._canvas.clear_polygon_preview()
            self._on_complete([[p.x(), p.y()] for p in self._points])


def _prompt_zone_name(parent) -> str | None:
    name, ok = QInputDialog.getText(parent, "Zone Name", "Name:")
    return name.strip() if ok and name.strip() else None


def start_draw_zone_polygon(controller, parent):
    def on_complete(points):
        page = controller.project.active_page
        if not page:
            return
        name = _prompt_zone_name(parent)
        if name:
            controller.dispatch(AddZone(page, name, points))
        controller.set_tool(None)

    controller.set_tool(PolygonGesture(on_complete=on_complete))


def start_draw_zone_rect(controller, parent):
    def on_complete(rect: QRectF):
        page = controller.project.active_page
        if not page:
            return
        name = _prompt_zone_name(parent)
        if name:
            points = [
                [rect.left(), rect.top()],
                [rect.right(), rect.top()],
                [rect.right(), rect.bottom()],
                [rect.left(), rect.bottom()],
            ]
            controller.dispatch(AddZone(page, name, points))
        controller.set_tool(None)

    controller.set_tool(RectGesture(on_complete=on_complete))


def start_delete_zone(controller):
    def on_click(pos: QPointF):
        page = controller.project.active_page
        if not page:
            return
        zone_id = controller.canvas.zone_id_at(pos)
        if zone_id is None:
            return
        zone = next((z for z in page.zones if z.id == zone_id), None)
        if zone:
            controller.dispatch(DeleteZone(zone, page))

    controller.set_tool(PointGesture(on_click))


class TaskWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, task, project, engine):
        super().__init__()
        self._task = task
        self._project = project
        self._engine = engine

    def run(self):
        session = Session(self._engine)
        try:
            if hasattr(self._task, "_progress_cb"):
                self._task._progress_cb = self.progress.emit
            self._task.execute(self._project, session)
            session.commit()
        except Exception as e:
            session.rollback()
            self.error.emit(str(e))
        finally:
            session.close()
        self.finished.emit()

    def _color_patches(self, samples, page_bgr, pw, ph):
        dominant_hue, dominant_sat = self._dominant_color(samples)
        if dominant_sat < self.COLOR_SAT_MIN:
            return [(0, 0, page_bgr)]
        tw = max(s.image.shape[1] for s in samples)
        th = max(s.image.shape[0] for s in samples)
        page_hsv = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2HSV)
        hue = dominant_hue
        lo1 = np.array(
            [max(hue - self.HUE_TOLERANCE, 0), self.COLOR_SAT_MIN, 30], dtype=np.uint8
        )
        hi1 = np.array([min(hue + self.HUE_TOLERANCE, 179), 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(page_hsv, lo1, hi1)
        if hue - self.HUE_TOLERANCE < 0:
            lo2 = np.array(
                [179 + hue - self.HUE_TOLERANCE, self.COLOR_SAT_MIN, 30], dtype=np.uint8
            )
            color_mask = cv2.bitwise_or(
                color_mask,
                cv2.inRange(page_hsv, lo2, np.array([179, 255, 255], dtype=np.uint8)),
            )
        elif hue + self.HUE_TOLERANCE > 179:
            hi2 = np.array([hue + self.HUE_TOLERANCE - 179, 255, 255], dtype=np.uint8)
            color_mask = cv2.bitwise_or(
                color_mask,
                cv2.inRange(
                    page_hsv, np.array([0, self.COLOR_SAT_MIN, 30], dtype=np.uint8), hi2
                ),
            )
        cnts, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = [
            [x, y, x + w, y + h]
            for cnt in cnts
            for x, y, w, h in [cv2.boundingRect(cnt)]
            if w * h >= self.MIN_BLOB_AREA
        ]
        if not rects:
            return [(0, 0, page_bgr)]
        merged = self._merge_rects(rects, tw, th)
        return [
            (
                max(x1 - tw, 0),
                max(y1 - th, 0),
                page_bgr[
                    max(y1 - th, 0) : min(y2 + th, ph),
                    max(x1 - tw, 0) : min(x2 + tw, pw),
                ],
            )
            for x1, y1, x2, y2 in merged
        ]

    def _dominant_color(self, samples):
        all_fg = []
        for s in samples:
            hsv = cv2.cvtColor(s.image, cv2.COLOR_BGR2HSV)
            fg = (
                hsv[s.mask > 0]
                if s.mask is not None and s.mask.shape == s.image.shape[:2]
                else hsv.reshape(-1, 3)
            )
            all_fg.append(fg)
        fg = np.concatenate(all_fg) if all_fg else np.empty((0, 3), dtype=np.uint8)
        if len(fg) == 0:
            return 0, 0
        sat = fg[:, 1].astype(float)
        weights = sat / (sat.sum() + 1e-6)
        return int(np.average(fg[:, 0], weights=weights)), int(fg[:, 1].mean())

    def _merge_rects(self, rects, tw, th):
        changed = True
        while changed:
            changed = False
            merged = []
            used = [False] * len(rects)
            for i, a in enumerate(rects):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = a
                for j, b in enumerate(rects):
                    if i == j or used[j]:
                        continue
                    bx1, by1, bx2, by2 = b
                    if (
                        ax2 + tw >= bx1
                        and bx2 + tw >= ax1
                        and ay2 + th >= by1
                        and by2 + th >= ay1
                    ):
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        used[j] = True
                        changed = True
                merged.append([ax1, ay1, ax2, ay2])
                used[i] = True
            rects = merged
        return rects

    def _nms(self, boxes, overlap=0.3):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            for kb in kept:
                kx1, ky1, kx2, ky2 = kb[:4]
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if (
                    ix2 > ix1
                    and iy2 > iy1
                    and (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1)) > overlap
                ):
                    break
            else:
                kept.append(box)
        return kept


class CountSymbols(Task):
    THRESHOLD = 0.6
    COLOR_SAT_MIN = 30
    HUE_TOLERANCE = 12
    MIN_BLOB_AREA = 16

    def __init__(self, page_id: int, page_bgr: np.ndarray, entries: list[LegendEntry]):
        self._page_id = page_id
        self._page_bgr = page_bgr
        self._entries = entries
        self._progress_cb = None

    def execute(self, project, session):
        entry_ids = {e.id for e in self._entries}
        for m in (
            session.execute(
                select(Marker).where(
                    Marker.page_id == self._page_id,
                    Marker.source == "auto",
                    Marker.legend_entry_id.in_(entry_ids),
                )
            )
            .scalars()
            .all()
        ):
            session.delete(m)
        session.flush()

        manual = (
            session.execute(
                select(Marker).where(
                    Marker.page_id == self._page_id,
                    Marker.source == "manual",
                    Marker.legend_entry_id.in_(entry_ids),
                )
            )
            .scalars()
            .all()
        )

        ph, pw = self._page_bgr.shape[:2]
        total = len(self._entries)
        for i, entry in enumerate(self._entries):
            entry_manual = [m for m in manual if m.legend_entry_id == entry.id]
            patches = self._color_patches(entry.samples, self._page_bgr, pw, ph)
            all_boxes = []
            for sample in entry.samples:
                rotations = [sample.image] + [
                    cv2.rotate(sample.image, f)
                    for f in (
                        cv2.ROTATE_90_CLOCKWISE,
                        cv2.ROTATE_180,
                        cv2.ROTATE_90_COUNTERCLOCKWISE,
                    )
                ]
                for rot in rotations:
                    rh, rw = rot.shape[:2]
                    for px, py, patch in patches:
                        if patch.shape[0] < rh or patch.shape[1] < rw:
                            continue
                        result = cv2.matchTemplate(patch, rot, cv2.TM_CCOEFF_NORMED)
                        _, result_bin = cv2.threshold(
                            result, self.THRESHOLD, 255, cv2.THRESH_BINARY
                        )
                        cnts, _ = cv2.findContours(
                            result_bin.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        for cnt in cnts:
                            x, y, w, h = cv2.boundingRect(cnt)
                            cx = min(x + w // 2, result.shape[1] - 1)
                            cy = min(y + h // 2, result.shape[0] - 1)
                            all_boxes.append(
                                (
                                    px + x,
                                    py + y,
                                    px + x + rw,
                                    py + y + rh,
                                    float(result[cy, cx]),
                                )
                            )
            for x1, y1, x2, y2, score in self._nms(all_boxes):
                if any(
                    x1 < m.x + m.w and x2 > m.x and y1 < m.y + m.h and y2 > m.y
                    for m in entry_manual
                ):
                    continue
                session.add(
                    Marker(
                        legend_entry_id=entry.id,
                        page_id=self._page_id,
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        score=score,
                        source="auto",
                    )
                )
            if self._progress_cb:
                self._progress_cb(i + 1, total)

    def complete(self, project, session):
        page = project.active_page
        if page and page.id == self._page_id:
            session.expire(page, ["markers"])
            _ = page.markers
        for e in self._entries:
            e.dirty = False
        project.notify_markers_changed()

    def _color_patches(self, samples, page_bgr, pw, ph):
        dominant_hue, dominant_sat = self._dominant_color(samples)
        if dominant_sat < self.COLOR_SAT_MIN:
            return [(0, 0, page_bgr)]
        tw = max(s.image.shape[1] for s in samples)
        th = max(s.image.shape[0] for s in samples)
        page_hsv = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2HSV)
        hue = dominant_hue
        lo1 = np.array(
            [max(hue - self.HUE_TOLERANCE, 0), self.COLOR_SAT_MIN, 30], dtype=np.uint8
        )
        hi1 = np.array([min(hue + self.HUE_TOLERANCE, 179), 255, 255], dtype=np.uint8)
        color_mask = cv2.inRange(page_hsv, lo1, hi1)
        if hue - self.HUE_TOLERANCE < 0:
            lo2 = np.array(
                [179 + hue - self.HUE_TOLERANCE, self.COLOR_SAT_MIN, 30], dtype=np.uint8
            )
            color_mask = cv2.bitwise_or(
                color_mask,
                cv2.inRange(page_hsv, lo2, np.array([179, 255, 255], dtype=np.uint8)),
            )
        elif hue + self.HUE_TOLERANCE > 179:
            hi2 = np.array([hue + self.HUE_TOLERANCE - 179, 255, 255], dtype=np.uint8)
            color_mask = cv2.bitwise_or(
                color_mask,
                cv2.inRange(
                    page_hsv, np.array([0, self.COLOR_SAT_MIN, 30], dtype=np.uint8), hi2
                ),
            )
        cnts, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        rects = [
            [x, y, x + w, y + h]
            for cnt in cnts
            for x, y, w, h in [cv2.boundingRect(cnt)]
            if w * h >= self.MIN_BLOB_AREA
        ]
        if not rects:
            return [(0, 0, page_bgr)]
        merged = self._merge_rects(rects, tw, th)
        return [
            (
                max(x1 - tw, 0),
                max(y1 - th, 0),
                page_bgr[
                    max(y1 - th, 0) : min(y2 + th, ph),
                    max(x1 - tw, 0) : min(x2 + tw, pw),
                ],
            )
            for x1, y1, x2, y2 in merged
        ]

    def _dominant_color(self, samples):
        all_fg = []
        for s in samples:
            hsv = cv2.cvtColor(s.image, cv2.COLOR_BGR2HSV)
            fg = (
                hsv[s.mask > 0]
                if s.mask is not None and s.mask.shape == s.image.shape[:2]
                else hsv.reshape(-1, 3)
            )
            all_fg.append(fg)
        fg = np.concatenate(all_fg) if all_fg else np.empty((0, 3), dtype=np.uint8)
        if len(fg) == 0:
            return 0, 0
        sat = fg[:, 1].astype(float)
        weights = sat / (sat.sum() + 1e-6)
        return int(np.average(fg[:, 0], weights=weights)), int(fg[:, 1].mean())

    def _merge_rects(self, rects, tw, th):
        changed = True
        while changed:
            changed = False
            merged = []
            used = [False] * len(rects)
            for i, a in enumerate(rects):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = a
                for j, b in enumerate(rects):
                    if i == j or used[j]:
                        continue
                    bx1, by1, bx2, by2 = b
                    if (
                        ax2 + tw >= bx1
                        and bx2 + tw >= ax1
                        and ay2 + th >= by1
                        and by2 + th >= ay1
                    ):
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        used[j] = True
                        changed = True
                merged.append([ax1, ay1, ax2, ay2])
                used[i] = True
            rects = merged
        return rects

    def _nms(self, boxes, overlap=0.3):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            for kb in kept:
                kx1, ky1, kx2, ky2 = kb[:4]
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                if (
                    ix2 > ix1
                    and iy2 > iy1
                    and (ix2 - ix1) * (iy2 - iy1) / ((x2 - x1) * (y2 - y1)) > overlap
                ):
                    break
            else:
                kept.append(box)
        return kept


class PointGesture:
    cursor = Qt.CursorShape.CrossCursor
    is_started = False

    def __init__(self, on_click):
        self._on_click = on_click

    def activate(self, canvas):
        pass

    def deactivate(self):
        pass

    def on_press(self, pos: QPointF):
        self._on_click(pos)

    def on_move(self, pos: QPointF):
        pass

    def on_release(self, pos: QPointF):
        pass


def start_add_marker(controller, entry: LegendEntry):
    def on_click(pos: QPointF):
        page = controller.project.active_page
        if page is None:
            return
        img = entry.samples[0].image if entry.samples else entry.image
        th, tw = img.shape[:2]
        x = round(pos.x() - tw / 2)
        y = round(pos.y() - th / 2)
        controller.dispatch(AddMarker(entry, page, x, y, tw, th))

    controller.set_tool(PointGesture(on_click))


def start_delete_marker(controller):
    def on_click(pos: QPointF):
        page = controller.project.active_page
        if page is None:
            return
        marker_id = controller.canvas.marker_id_at(pos)
        if marker_id is None:
            return
        marker = next((m for m in page.markers if m.id == marker_id), None)
        if marker:
            controller.dispatch(DeleteMarker(marker, page))

    controller.set_tool(PointGesture(on_click))


def start_set_symbol(controller, entry: LegendEntry):
    def on_complete(rect):
        crop = controller.canvas.crop(rect)
        if not crop.isNull():
            controller.dispatch(SetLegendSample(entry, qpixmap_to_bgr(crop)))
        controller.set_tool(None)

    controller.set_tool(RectGesture(on_complete=on_complete))


def start_add_legend_entry(controller, parent):
    def on_complete(rect):
        crop = controller.canvas.crop(rect)
        if crop.isNull():
            controller.set_tool(None)
            return
        label, ok = QInputDialog.getText(parent, "New Entry", "Label:")
        if ok and label.strip():
            controller.dispatch(
                AddLegendEntryFromSample(label.strip(), qpixmap_to_bgr(crop))
            )
        controller.set_tool(None)

    controller.set_tool(RectGesture(on_complete=on_complete))


class AppController(QObject):
    task_started = pyqtSignal(str)
    task_finished = pyqtSignal()
    task_progress = pyqtSignal(int, int)

    def __init__(self, project: Project, session: Session | None = None):
        super().__init__()
        self._project = project
        self._session = session
        self._engine = None
        self._canvas = None
        self._thread: QThread | None = None
        self._worker: QObject | None = None
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

    def set_session(self, session: Session, engine):
        self._session = session
        self._engine = engine

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

    def dispatch_async(self, task: Task, label: str):
        self._session.commit()
        worker = TaskWorker(task, self._project, self._engine)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.task_progress)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._on_async_done(task))
        self._thread = thread
        self._worker = worker
        self.task_started.emit(label)
        thread.start()

    def _on_async_done(self, task: Task):
        if hasattr(task, "complete"):
            task.complete(self._project, self._session)
        self.task_finished.emit()
        self._thread = None

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
        if self._session and self._thread is None:
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
        self._overlay_items: list[QGraphicsRectItem] = []
        self._zone_items: list = []
        self._poly_preview_items: list = []
        self._pan_origin = None
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

    def show_page(self, page):
        self.load_pixmap(page.pixmap)
        self.set_markers(page.markers)
        self.set_zones(page.zones)

    def load_pixmap(self, pixmap: QPixmap):
        self._scene.clear()
        self._rect_item = None
        self._overlay_items = []
        self._zone_items = []
        self._poly_preview_items = []
        self._page_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._page_item)
        self.fitInView(self._page_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_markers(self, markers: list):
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        for marker in markers:
            color = _entry_color(marker.legend_entry_id)
            pen = QPen(color)
            pen.setWidth(3)
            pen.setCosmetic(True)
            fill = QColor(color)
            fill.setAlpha(40)
            item = QGraphicsRectItem(QRectF(marker.x, marker.y, marker.w, marker.h))
            item.setPen(pen)
            item.setBrush(fill)
            item.setData(0, marker.id)
            self._scene.addItem(item)
            self._overlay_items.append(item)

    def set_zones(self, zones: list):
        for item in self._zone_items:
            self._scene.removeItem(item)
        self._zone_items = []
        color = QColor(80, 200, 120)
        pen = QPen(color)
        pen.setWidth(3)
        pen.setCosmetic(True)
        fill = QColor(color)
        fill.setAlpha(25)
        for zone in zones:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in zone.points])
            item = QGraphicsPolygonItem(poly)
            item.setPen(pen)
            item.setBrush(fill)
            item.setData(0, zone.id)
            self._scene.addItem(item)
            self._zone_items.append(item)
            cx = sum(p[0] for p in zone.points) / len(zone.points)
            cy = sum(p[1] for p in zone.points) / len(zone.points)
            label = QGraphicsTextItem(zone.name)
            label.setDefaultTextColor(color)
            label.setPos(cx, cy)
            self._scene.addItem(label)
            self._zone_items.append(label)

    def zone_id_at(self, pos: QPointF) -> int | None:
        for item in self._scene.items(pos):
            zone_id = item.data(0)
            if zone_id is not None:
                return zone_id
        return None

    def update_polygon_preview(self, points: list[QPointF], cursor: QPointF):
        for item in self._poly_preview_items:
            self._scene.removeItem(item)
        self._poly_preview_items = []
        pen = QPen(QColor(255, 160, 0))
        pen.setWidth(2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        for i in range(len(points) - 1):
            path = QPainterPath()
            path.moveTo(points[i])
            path.lineTo(points[i + 1])
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            self._scene.addItem(item)
            self._poly_preview_items.append(item)
        if points:
            path = QPainterPath()
            path.moveTo(points[-1])
            path.lineTo(cursor)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            self._scene.addItem(item)
            self._poly_preview_items.append(item)

    def clear_polygon_preview(self):
        for item in self._poly_preview_items:
            self._scene.removeItem(item)
        self._poly_preview_items = []

    def marker_id_at(self, pos: QPointF) -> int | None:
        for item in self._scene.items(pos):
            marker_id = item.data(0)
            if marker_id is not None:
                return marker_id
        return None

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
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._active_tool and hasattr(self._active_tool, "on_key_return"):
                self._active_tool.on_key_return()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            if hasattr(self._active_tool, "on_double_click"):
                self._active_tool.on_double_click(self.mapToScene(event.pos()))
        else:
            super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            if not (self._active_tool and self._active_tool.is_started):
                self._pan_origin = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool:
            self._active_tool.on_press(self.mapToScene(event.pos()))
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_origin is not None:
            delta = event.pos() - self._pan_origin
            self._pan_origin = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            return
        if self._active_tool:
            self._active_tool.on_move(self.mapToScene(event.pos()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.MouseButton.MiddleButton
            and self._pan_origin is not None
        ):
            self._pan_origin = None
            if self._active_tool and self._active_tool.cursor is not None:
                self.setCursor(self._active_tool.cursor)
            else:
                self.unsetCursor()
            return
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
        self._entries = entries
        self._list.clear()
        for entry in entries:
            item = QListWidgetItem(QIcon(entry.pixmap), entry.label)
            self._list.addItem(item)

    def selected_entry(self) -> LegendEntry | None:
        row = self._list.currentRow()
        if row < 0 or not hasattr(self, "_entries"):
            return None
        return self._entries[row]


def _zone_counts(page, entries: list) -> dict:
    if not page or not page.zones or not page.markers:
        return {}
    contours = {
        zone.id: np.array(zone.points, dtype=np.float32).reshape(-1, 1, 2)
        for zone in page.zones
    }
    counts = {}
    for marker in page.markers:
        cx = marker.x + marker.w / 2
        cy = marker.y + marker.h / 2
        for zone in page.zones:
            key = (zone.id, marker.legend_entry_id)
            counts.setdefault(key, 0)
            if cv2.pointPolygonTest(contours[zone.id], (cx, cy), False) >= 0:
                counts[key] += 1
    return counts


class ZoneCountsPanel(QDockWidget):
    def __init__(self, controller: "AppController"):
        super().__init__("Zone Counts")
        self._project = controller.project
        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.setWidget(self._table)
        self._project.active_page_changed.connect(self._rebuild)
        self._project.markers_changed.connect(self._rebuild)
        self._project.zones_changed.connect(self._rebuild)
        self._project.legend_entries_changed.connect(self._rebuild)

    def _rebuild(self):
        page = self._project.active_page
        entries = self._project.legend_entries
        zones = page.zones if page else []
        self._table.clear()
        self._table.setRowCount(len(zones))
        self._table.setColumnCount(len(entries))
        self._table.setHorizontalHeaderLabels([e.label for e in entries])
        self._table.setVerticalHeaderLabels([z.name for z in zones])
        if not zones or not entries or not page:
            return
        counts = _zone_counts(page, entries)
        for r, zone in enumerate(zones):
            for c, entry in enumerate(entries):
                val = counts.get((zone.id, entry.id), 0)
                self._table.setItem(r, c, QTableWidgetItem(str(val)))


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
        self._launch(Session(engine), engine, path)

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "QS Project (*.qsproj)"
        )
        if not path:
            return
        engine = _make_engine(path)
        Base.metadata.create_all(engine)
        _init_project_state(engine)
        self._launch(Session(engine), engine, path)

    def _launch(self, session, engine, path):
        self._controller.set_session(session, engine)
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

        self.zone_counts_panel = ZoneCountsPanel(controller)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.zone_counts_panel
        )

        self._project.legend_entries_changed.connect(
            lambda: self.legend_panel.set_entries(self._project.legend_entries)
        )
        self._project.active_page_changed.connect(self._on_active_page_changed)
        self._project.markers_changed.connect(
            lambda: self.viewer.set_markers(
                self._project.active_page.markers if self._project.active_page else []
            )
        )
        self._project.zones_changed.connect(
            lambda: self.viewer.set_zones(
                self._project.active_page.zones if self._project.active_page else []
            )
        )
        self.viewer.escape_pressed.connect(self._controller.cancel_tool)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.hide()
        self.statusBar().addPermanentWidget(self._progress_bar)
        controller.task_started.connect(self._on_task_started)
        controller.task_finished.connect(self._on_task_finished)
        controller.task_progress.connect(self._on_task_progress)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.legend_panel.toggleViewAction())
        view_menu.addAction(self.zone_counts_panel.toggleViewAction())

        self._toolbar = QToolBar()
        self.addToolBar(self._toolbar)

        import_action = QAction("Import Drawing", self)
        import_action.triggered.connect(self._import_drawing)
        self._toolbar.addAction(import_action)

        legend_action = QAction("Load Legend", self)
        legend_action.triggered.connect(lambda: start_legend_select(self._controller))
        self._toolbar.addAction(legend_action)

        add_entry_action = QAction("Add Entry", self)
        add_entry_action.triggered.connect(
            lambda: start_add_legend_entry(self._controller, self)
        )
        self._toolbar.addAction(add_entry_action)

        set_symbol_action = QAction("Set Symbol", self)
        set_symbol_action.triggered.connect(self._set_symbol)
        self._toolbar.addAction(set_symbol_action)

        count_action = QAction("Count", self)
        count_action.triggered.connect(self._dispatch_count)
        self._toolbar.addAction(count_action)

        add_marker_action = QAction("Add Marker", self)
        add_marker_action.triggered.connect(self._add_marker_mode)
        self._toolbar.addAction(add_marker_action)

        delete_marker_action = QAction("Delete Marker", self)
        delete_marker_action.triggered.connect(
            lambda: start_delete_marker(self._controller)
        )
        self._toolbar.addAction(delete_marker_action)

        self._toolbar.addSeparator()

        draw_zone_poly_action = QAction("Draw Zone (Polygon)", self)
        draw_zone_poly_action.triggered.connect(
            lambda: start_draw_zone_polygon(self._controller, self)
        )
        self._toolbar.addAction(draw_zone_poly_action)

        draw_zone_rect_action = QAction("Draw Zone (Rect)", self)
        draw_zone_rect_action.triggered.connect(
            lambda: start_draw_zone_rect(self._controller, self)
        )
        self._toolbar.addAction(draw_zone_rect_action)

        delete_zone_action = QAction("Delete Zone", self)
        delete_zone_action.triggered.connect(
            lambda: start_delete_zone(self._controller)
        )
        self._toolbar.addAction(delete_zone_action)

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self._controller.undo)
        self.addAction(undo_action)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Ctrl+Y")
        redo_action.triggered.connect(self._controller.redo)
        self.addAction(redo_action)

    def _on_task_started(self, label: str):
        self.statusBar().showMessage(label)
        self._progress_bar.setValue(0)
        self._progress_bar.show()
        self._toolbar.setEnabled(False)

    def _on_task_progress(self, current: int, total: int):
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(current)

    def _on_task_finished(self):
        self.statusBar().clearMessage()
        self._progress_bar.hide()
        self._toolbar.setEnabled(True)

    def _on_active_page_changed(self):
        page = self._project.active_page
        if page:
            self.viewer.show_page(page)

    def _dispatch_count(self):
        page = self._project.active_page
        if page is None or page.pixmap is None:
            return
        entries = [
            e
            for e in self._project.legend_entries
            if e.auto_count and e.samples and e.dirty
        ]
        if not entries:
            return
        page_bgr = qpixmap_to_bgr(page.pixmap)
        self._controller.dispatch_async(
            CountSymbols(page.id, page_bgr, entries), "Counting symbols..."
        )

    def _add_marker_mode(self):
        entry = self.legend_panel.selected_entry()
        if entry is None:
            QMessageBox.information(
                self, "No Selection", "Select a legend entry first."
            )
            return
        start_add_marker(self._controller, entry)

    def _set_symbol(self):
        entry = self.legend_panel.selected_entry()
        if entry is None:
            QMessageBox.information(
                self, "No Selection", "Select a legend entry first."
            )
            return
        start_set_symbol(self._controller, entry)

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
