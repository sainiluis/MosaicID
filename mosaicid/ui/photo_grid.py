"""Grilla fotográfica justificada (filas de altura pareja, 4 px de separación, sin cards).

- Solo pinta lo visible y pide thumbnails en segundo plano (QThreadPool) a medida que aparecen.
- Nunca carga la imagen original: usa cache/thumbnails/.
- Hover discreto: fecha + similitud.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QThreadPool, QTimer, Signal
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..services.workers import ThumbLoader, thumb_emitter
from .theme import C, font, qc

C_OK = C["ok"]


@dataclass
class GridItem:
    key: str
    thumb: Path
    aspect: float
    caption: str  # fecha
    metric: str = ""  # similitud
    payload: object = None
    boxes: list = field(default_factory=list)  # rostros coincidentes, normalizados (x, y, w, h) sobre la foto


@dataclass
class GridSection:
    title: str
    subtitle: str
    items: list[GridItem] = field(default_factory=list)


class ThumbCache:
    def __init__(self, limit: int = 600):
        self.images: dict[str, QImage] = {}
        self.loaded_at: dict[str, float] = {}
        self.pending: set[str] = set()
        self.limit = limit
        thumb_emitter().loaded.connect(self._on_loaded)
        self.listeners: list = []

    def get(self, key: str, path: Path) -> QImage | None:
        img = self.images.get(key)
        if img is None and key not in self.pending:
            self.pending.add(key)
            QThreadPool.globalInstance().start(ThumbLoader(key, path, 640))
        return img

    def _on_loaded(self, key: str, img: QImage) -> None:
        self.pending.discard(key)
        if len(self.images) > self.limit:
            for k in list(self.images)[: self.limit // 4]:
                self.images.pop(k, None)
                self.loaded_at.pop(k, None)
        self.images[key] = img
        self.loaded_at[key] = time.perf_counter()
        for fn in list(self.listeners):
            fn(key)


_CACHE: ThumbCache | None = None


def thumb_cache() -> ThumbCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ThumbCache()
    return _CACHE


class PhotoGrid(QWidget):
    itemClicked = Signal(object)

    def __init__(self, row_height: int = 190, gap: int = 4):
        super().__init__()
        self.row_height = row_height
        self.gap = gap
        self.sections: list[GridSection] = []
        self.tiles: list[tuple[QRectF, GridItem]] = []
        self.headers: list[tuple[QRectF, GridSection]] = []
        self.hover: GridItem | None = None
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.cache = thumb_cache()
        self.cache.listeners.append(self._loaded)
        self._fade = QTimer(self, interval=30, timeout=self.update)
        self._title_font = font(15, 600)
        self._sub_font = font(12)
        self._cap_font = font(11.5)
        self._cap_bold = font(11.5, 600)
        self._layout_w = -1
        self._shown_at = 0.0

    def set_sections(self, sections: list[GridSection]) -> None:
        self.sections = sections
        self.hover = None
        self._shown_at = time.perf_counter()  # los recuadros de coincidencia se animan cada vez que se muestran
        self._relayout(force=True)
        if any(it.boxes for sec in sections for it in sec.items):
            self._fade.start()
        self.update()

    def _loaded(self, key: str) -> None:
        if any(it.key == key for _, it in self.tiles):
            self._fade.start()
            self.update()

    def resizeEvent(self, e):
        self._relayout()

    def _relayout(self, force: bool = False) -> None:
        W = self.width()
        if W <= 0 or (W == self._layout_w and not force):
            return
        self._layout_w = W
        self.tiles, self.headers = [], []
        y = 0.0
        show_headers = len(self.sections) > 1 or (self.sections and self.sections[0].title)
        for sec in self.sections:
            if show_headers:
                self.headers.append((QRectF(0, y + 20, W, 28), sec))
                y += 20 + 28 + 10
            row: list[GridItem] = []
            for it in sec.items + [None]:
                if it is not None:
                    row.append(it)
                    width = sum(max(0.4, min(2.5, r.aspect)) * self.row_height for r in row) + self.gap * (len(row) - 1)
                    if width < W:
                        continue
                    scale = (W - self.gap * (len(row) - 1)) / (width - self.gap * (len(row) - 1))
                elif row:
                    scale = 1.0
                else:
                    break
                h = self.row_height * scale
                x = 0.0
                for r in row:
                    w = max(0.4, min(2.5, r.aspect)) * h
                    self.tiles.append((QRectF(x, y, w, h), r))
                    x += w + self.gap
                y += h + self.gap
                row = []
        self.setFixedHeight(int(y + 8))

    # ── interacción ──────────────────────────────────────────────────────
    def _at(self, pos) -> GridItem | None:
        for r, it in self.tiles:
            if r.contains(pos):
                return it
        return None

    def mouseMoveEvent(self, e):
        it = self._at(e.position())
        if it is not self.hover:
            self.hover = it
            self.setCursor(Qt.PointingHandCursor if it else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, e):
        self.hover = None
        self.update()

    def mouseReleaseEvent(self, e):
        it = self._at(e.position())
        if it is not None and e.button() == Qt.LeftButton:
            self.itemClicked.emit(it.payload)

    # ── pintura ──────────────────────────────────────────────────────────
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        vis = QRectF(ev.rect())
        for r, sec in self.headers:
            if r.intersects(vis):
                p.setFont(self._title_font)
                p.setPen(qc("ink"))
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, sec.title)
                tw = p.fontMetrics().horizontalAdvance(sec.title)
                p.setFont(self._sub_font)
                p.setPen(qc("ink3"))
                p.drawText(r.adjusted(tw + 12, 2, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, sec.subtitle)
        now = time.perf_counter()
        animating = False
        for r, it in self.tiles:
            if not r.intersects(vis.adjusted(0, -400, 0, 400)):
                continue
            img = self.cache.get(it.key, it.thumb)
            if not r.intersects(vis):
                continue
            path = QPainterPath()
            path.addRoundedRect(r, 3, 3)
            p.save()
            p.setClipPath(path)
            p.fillRect(r, qc("sunken"))
            if img is not None and not img.isNull():
                age = now - self.cache.loaded_at.get(it.key, 0)
                alpha = min(1.0, age / 0.3)
                animating |= alpha < 1.0
                p.setOpacity(alpha)
                iw, ih = img.width(), img.height()
                s = max(r.width() / iw, r.height() / ih)
                ir = QRectF(r.center().x() - iw * s / 2, r.center().y() - ih * s / 2, iw * s, ih * s)
                p.drawImage(ir, img)
                p.setOpacity(1.0)
                if it.boxes:
                    animating |= self._paint_match(p, r, ir, it, now - max(self._shown_at, self.cache.loaded_at.get(it.key, 0)))
            if it is self.hover:
                p.fillRect(r, QColor(0, 0, 0, 20))
                g = QLinearGradient(0, r.bottom() - 44, 0, r.bottom())
                g.setColorAt(0, QColor(0, 0, 0, 0))
                g.setColorAt(1, QColor(0, 0, 0, 150))
                p.fillRect(QRectF(r.x(), r.bottom() - 44, r.width(), 44), g)
                p.setPen(QColor("white"))
                p.setFont(self._cap_font)
                p.drawText(r.adjusted(10, 0, -10, -8), Qt.AlignLeft | Qt.AlignBottom, it.caption)
                if it.metric:
                    p.setFont(self._cap_bold)
                    p.drawText(r.adjusted(10, 0, -10, -8), Qt.AlignRight | Qt.AlignBottom, it.metric)
            p.restore()
        if not animating:
            self._fade.stop()

    def _paint_match(self, p: QPainter, tile: QRectF, ir: QRectF, it: GridItem, age: float) -> bool:
        """Recuadro verde que se «engancha» sobre la cara (se achica y aparece) y tilde en la esquina."""
        t = max(0.0, min(1.0, (age - 0.15) / 0.45))
        ease = 1 - (1 - t) ** 3
        green = QColor(C_OK)
        for nx, ny, nw, nh in it.boxes:
            b = QRectF(ir.x() + nx * ir.width(), ir.y() + ny * ir.height(), nw * ir.width(), nh * ir.height())
            if b.width() < 16:  # en fotos grupales la cara puede medir pocos píxeles: mínimo visible
                k = (16 - b.width()) / 2
                b = b.adjusted(-k, -k, k, k)
            grow = 0.35 * (1 - ease)
            b = b.adjusted(-b.width() * grow, -b.height() * grow, b.width() * grow, b.height() * grow)
            halo = QColor(255, 255, 255, int(150 * ease))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(halo, 4.5))
            p.drawRoundedRect(b, 5, 5)
            green.setAlpha(int(255 * ease))
            p.setPen(QPen(green, 2.4))
            p.drawRoundedRect(b, 5, 5)
        # tilde
        d = 22
        c = QPointF(tile.right() - d / 2 - 7, tile.top() + d / 2 + 7)
        k = 0.6 + 0.4 * ease
        p.setPen(QPen(QColor(255, 255, 255, int(255 * ease)), 2))
        green.setAlpha(int(255 * ease))
        p.setBrush(green)
        p.drawEllipse(c, d / 2 * k, d / 2 * k)
        pen = QPen(QColor(255, 255, 255, int(255 * ease)), 2.2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        u = d / 2 * k
        path = QPainterPath(QPointF(c.x() - u * 0.45, c.y() + u * 0.02))
        path.lineTo(c.x() - u * 0.1, c.y() + u * 0.38)
        path.lineTo(c.x() + u * 0.5, c.y() - u * 0.35)
        p.drawPath(path)
        return t < 1.0
