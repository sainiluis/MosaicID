"""Componentes base de la interfaz (equivalentes a los del prototipo React)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import (QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer,
                            Signal, Property)
from PySide6.QtGui import QBrush, QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QPushButton, QScrollArea,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import theme
from .theme import C, LINE, font, qc

ICONS_DIR: Path | None = None


# ── iconos (Lucide, SVG locales) ─────────────────────────────────────────
@lru_cache(maxsize=512)
def _svg_bytes(name: str, color: str, stroke: float) -> bytes:
    p = ICONS_DIR / f"{name}.svg"
    txt = p.read_text(encoding="utf-8")
    txt = txt.replace("currentColor", color).replace('stroke-width="2"', f'stroke-width="{stroke}"')
    return txt.encode()


def pixmap(name: str, color: str = C["ink3"], size: int = 16, stroke: float = 1.75, dpr: float = 2.0) -> QPixmap:
    r = QSvgRenderer(_svg_bytes(name, color, stroke))
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    r.render(p)
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def icon(name: str, color: str = C["ink3"], size: int = 16, stroke: float = 1.75) -> QIcon:
    return QIcon(pixmap(name, color, size, stroke))


def icon_label(name: str, color: str = C["ink3"], size: int = 16) -> QLabel:
    l = QLabel()
    l.setPixmap(pixmap(name, color, size))
    l.setFixedSize(size, size)
    return l


# ── conversiones ─────────────────────────────────────────────────────────
def bgr_to_qimage(a: np.ndarray) -> QImage:
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, a.strides[0], QImage.Format_BGR888)
    return img.copy()


# ── tipografía ───────────────────────────────────────────────────────────
def label(text: str = "", role: str = "body", tone: str | None = None, wrap: bool = False,
          weight: int | None = None) -> QLabel:
    l = QLabel(text)
    l.setProperty("role", role)
    if tone:
        l.setProperty("tone", tone)
    if wrap:
        l.setWordWrap(True)
    if weight:
        f = l.font()
        f.setWeight(QFont_weight(weight))
        l.setFont(f)
        l.setStyleSheet(f"font-weight: {weight};")
    return l


def QFont_weight(w: int):
    from PySide6.QtGui import QFont
    return QFont.Weight(w)


def restyle(w: QWidget) -> None:
    w.style().unpolish(w)
    w.style().polish(w)


def set_tone(l: QLabel, tone: str | None) -> None:
    l.setProperty("tone", tone or "")
    restyle(l)


# ── botones ──────────────────────────────────────────────────────────────
class Button(QPushButton):
    def __init__(self, text: str = "", variant: str = "secondary", size: str = "md", icon_name: str | None = None,
                 icon_right: str | None = None, parent=None):
        super().__init__(text, parent)
        self.setProperty("variant", variant)
        self.setProperty("size", size)
        self.setCursor(Qt.PointingHandCursor)
        self._variant = variant
        self._icon_name = icon_name
        self._icon_right = icon_right
        self._text = text
        self._loading = False
        self._angle = 0
        self._spin = QTimer(self, interval=60, timeout=self._tick)
        if icon_right:
            self.setLayoutDirection(Qt.RightToLeft)
        self._apply_icon()

    def _color(self) -> str:
        if not self.isEnabled():
            return "#FFFFFF" if self._variant == "primary" else C["ink4"]
        return {"primary": "#FFFFFF", "quiet": C["accent"], "ghost": C["ink3"], "danger": C["err"],
                "dark": "#C9CDD1"}.get(self._variant, C["ink3"])

    def _apply_icon(self) -> None:
        name = self._icon_name or self._icon_right
        sz = 14 if self.property("size") == "sm" else 15
        if self._loading:
            pm = pixmap("loader-circle", self._color(), sz, 2)
            t = QPixmap(pm.size())
            t.setDevicePixelRatio(pm.devicePixelRatio())
            t.fill(Qt.transparent)
            p = QPainter(t)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            c = QPointF(sz / 2, sz / 2)
            p.translate(c)
            p.rotate(self._angle)
            p.translate(-c)
            p.drawPixmap(0, 0, pm)
            p.end()
            self.setIcon(QIcon(t))
        elif name:
            self.setIcon(icon(name, self._color(), sz))
        else:
            self.setIcon(QIcon())
        self.setIconSize(QSize(sz, sz))

    def set_icon(self, name: str | None) -> None:
        self._icon_name = name
        self._apply_icon()

    def _tick(self) -> None:
        self._angle = (self._angle + 30) % 360
        self._apply_icon()

    def set_loading(self, on: bool, text: str | None = None) -> None:
        self._loading = on
        self.setEnabled(not on)
        if text is not None:
            self.setText(text)
        (self._spin.start if on else self._spin.stop)()
        self._apply_icon()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.EnabledChange:
            self._apply_icon()


# ── indicadores ──────────────────────────────────────────────────────────
TONE_COLOR = {"ok": C["ok"], "warn": C["warn"], "err": C["err"], "idle": C["ink4"], "accent": C["accent"]}
TONE_SOFT = {"ok": C["ok_soft"], "warn": C["warn_soft"], "err": C["err_soft"], "idle": C["sunken"], "accent": C["accent_soft"]}


class Dot(QWidget):
    def __init__(self, tone: str = "ok", size: int = 6, parent=None):
        super().__init__(parent)
        self.tone = tone
        self.setFixedSize(size, size)

    def set_tone(self, tone: str) -> None:
        self.tone = tone
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(TONE_COLOR.get(self.tone, C["ink4"])))
        p.drawEllipse(self.rect())


class Badge(QLabel):
    def __init__(self, text: str = "", tone: str = "idle", dot: bool = True):
        super().__init__()
        self._dot = dot
        self.set(text, tone)

    def set(self, text: str, tone: str) -> None:
        self.tone = tone
        self.setText(text)
        self.setStyleSheet(
            f"background: {TONE_SOFT[tone]}; color: {TONE_COLOR[tone] if tone != 'idle' else C['ink3']};"
            f" border-radius: 4px; padding: 2px 6px 2px {14 if self._dot else 6}px; font-size: 12px; font-weight: 500;")
        self.setFixedHeight(20)
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._dot:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(TONE_COLOR[self.tone]))
            p.drawEllipse(QRectF(6, self.height() / 2 - 3, 6, 6))


def status_line(text: str, tone: str = "ok", role: str = "small2") -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(6)
    d = Dot(tone)
    l = label(text, role)
    h.addWidget(d, 0, Qt.AlignVCenter)
    h.addWidget(l)
    w.dot, w.text = d, l
    return w


class Divider(QFrame):
    def __init__(self, vertical: bool = False, margin: int = 0):
        super().__init__()
        if vertical:
            self.setFixedWidth(1)
        else:
            self.setFixedHeight(1)
        self.setStyleSheet(f"background: {C['line']}; border: none;")


class Meter(QWidget):
    """Barra fina (similitud / probabilidad / progreso) con marcador de umbral opcional."""

    def __init__(self, value: float = 0.0, tone: str = "accent", marker: float | None = None, height: int = 9):
        super().__init__()
        self.value, self.tone, self.marker = value, tone, marker
        self.setFixedHeight(height)
        self.setMinimumWidth(40)

    def set(self, value: float, tone: str | None = None, marker: float | None = None) -> None:
        self.value = max(0.0, min(1.0, float(value)))
        if tone:
            self.tone = tone
        if marker is not None:
            self.marker = marker
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        y = self.height() / 2 - 1.5
        w = self.width()
        p.setPen(Qt.NoPen)
        p.setBrush(qc("sunken"))
        p.drawRoundedRect(QRectF(0, y, w, 3), 1.5, 1.5)
        p.setBrush(QColor(TONE_COLOR.get(self.tone, C["accent"])))
        p.drawRoundedRect(QRectF(0, y, w * self.value, 3), 1.5, 1.5)
        if self.marker is not None:
            x = w * self.marker
            p.setBrush(qc("ink2"))
            p.drawRect(QRectF(x, y - 3, 1, 9))


class Spinner(QWidget):
    def __init__(self, size: int = 16, color: str = C["ink3"]):
        super().__init__()
        self.setFixedSize(size, size)
        self._a = 0
        self._color = QColor(color)
        self._t = QTimer(self, interval=40, timeout=self._tick)
        self._t.start()

    def _tick(self):
        self._a = (self._a + 24) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._color, 1.8)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        r = QRectF(2, 2, self.width() - 4, self.height() - 4)
        p.drawArc(r, -self._a * 16, 270 * 16)


# ── layout de página ─────────────────────────────────────────────────────
def hbox(*items, spacing: int = 8, margins=(0, 0, 0, 0), align=None) -> QHBoxLayout:
    h = QHBoxLayout()
    h.setContentsMargins(*margins)
    h.setSpacing(spacing)
    for it in items:
        if it is None:
            h.addStretch(1)
        elif isinstance(it, int):
            h.addSpacing(it)
        elif isinstance(it, QWidget):
            h.addWidget(it, 0, align) if align is not None else h.addWidget(it)
        else:
            h.addLayout(it)
    return h


def vbox(*items, spacing: int = 0, margins=(0, 0, 0, 0)) -> QVBoxLayout:
    v = QVBoxLayout()
    v.setContentsMargins(*margins)
    v.setSpacing(spacing)
    for it in items:
        if it is None:
            v.addStretch(1)
        elif isinstance(it, int):
            v.addSpacing(it)
        elif isinstance(it, QWidget):
            v.addWidget(it)
        else:
            v.addLayout(it)
    return v


def wrap(layout) -> QWidget:
    w = QWidget()
    w.setLayout(layout)
    return w


class Page(QScrollArea):
    """Página con scroll y márgenes consistentes (32 px laterales, 28 px arriba)."""

    def __init__(self, max_width: int = 1360, margins=(32, 28, 32, 40), scroll: bool = True):
        super().__init__()
        self.setObjectName("Page")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if not scroll:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer = QWidget()
        outer.setObjectName("PageBody")
        oh = QHBoxLayout(outer)
        oh.setContentsMargins(0, 0, 0, 0)
        self.body = QWidget()
        self.body.setMaximumWidth(max_width)
        self.layout_ = QVBoxLayout(self.body)
        self.layout_.setContentsMargins(*margins)
        self.layout_.setSpacing(0)
        oh.addWidget(self.body)
        self.setWidget(outer)

    def add(self, w, stretch: int = 0) -> None:
        if isinstance(w, QWidget):
            self.layout_.addWidget(w, stretch)
        elif isinstance(w, int):
            self.layout_.addSpacing(w)
        else:
            self.layout_.addLayout(w, stretch)

    def stretch(self) -> None:
        self.layout_.addStretch(1)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", aside: list[QWidget] | None = None, back: QWidget | None = None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 20)
        v.setSpacing(0)
        if back:
            v.addWidget(back, 0, Qt.AlignLeft)
            v.addSpacing(16)
        row = QHBoxLayout()
        row.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(4)
        self.title = label(title, "title")
        self.subtitle = label(subtitle, "lead")
        left.addWidget(self.title)
        left.addWidget(self.subtitle)  # siempre con padre: sin él, show() lo abre como ventana suelta
        self.subtitle.setVisible(bool(subtitle))
        row.addLayout(left, 1)
        self.aside = QHBoxLayout()
        self.aside.setSpacing(8)
        for a in aside or []:
            self.aside.addWidget(a)
        row.addLayout(self.aside)
        row.setAlignment(self.aside, Qt.AlignBottom)
        v.addLayout(row)


class LinkLabel(QLabel):
    clicked = Signal()

    def __init__(self, text: str, role: str = "small", icon_left: str | None = None, color: str | None = None):
        super().__init__(text)
        self.setProperty("role", role)
        self.setCursor(Qt.PointingHandCursor)
        self._color = color

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()


class BackLink(QWidget):
    clicked = Signal()

    def __init__(self, text: str):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.ic = icon_label("arrow-left", C["ink3"], 14)
        self.l = label(text, "small")
        h.addWidget(self.ic)
        h.addWidget(self.l)

    def enterEvent(self, e):
        self.l.setStyleSheet(f"color: {C['ink']};")

    def leaveEvent(self, e):
        self.l.setStyleSheet("")

    def mouseReleaseEvent(self, e):
        self.clicked.emit()


def overline(text: str) -> QLabel:
    return label(text.upper(), "over")


class PropRow(QWidget):
    def __init__(self, key: str, value: str | QWidget = ""):
        super().__init__()
        self.setFixedHeight(28)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(label(key, "muted"))
        h.addStretch(1)
        if isinstance(value, QWidget):
            self.value = value
        else:
            self.value = label(value, "body")
            self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        h.addWidget(self.value)

    def set(self, text: str) -> None:
        self.value.setText(text)


class Row(QWidget):
    """Fila de lista sin contenedor: hover sutil, separador inferior, clic."""

    clicked = Signal()

    def __init__(self, height: int = 44, clickable: bool = True, separator: bool = True, radius: int = 0):
        super().__init__()
        self.setFixedHeight(height)
        self._hover = False
        self._selected = False
        self._clickable = clickable
        self._sep = separator
        self._radius = radius
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover, True)
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(0, 0, 0, 0)
        self.h.setSpacing(12)

    def set_selected(self, on: bool) -> None:
        self._selected = on
        self.update()

    def enterEvent(self, e):
        self._hover = True
        self.update()

    def leaveEvent(self, e):
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        if self._clickable and e.button() == Qt.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self._selected or (self._hover and self._clickable):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 12 if self._selected else 7))
            p.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        if self._sep:
            p.setPen(LINE)
            p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


class FaceThumb(QWidget):
    """Miniatura de rostro: recorte real (np BGR) o QImage; circular o cuadrada."""

    def __init__(self, size: int = 28, round_: bool = True, radius: int = 4):
        super().__init__()
        self._img: QImage | None = None
        self.round = round_
        self.radius = radius
        if size:
            self.setFixedSize(size, size)

    def set_array(self, a: np.ndarray | None) -> None:
        self._img = bgr_to_qimage(a) if a is not None else None
        self.update()

    def set_image(self, img: QImage | None) -> None:
        self._img = img
        self.update()

    def set_path(self, path: Path | str | None) -> None:
        self._img = QImage(str(path)) if path and Path(path).exists() else None
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        path = QPainterPath()
        if self.round:
            path.addEllipse(r)
        else:
            path.addRoundedRect(r, self.radius, self.radius)
        p.setClipPath(path)
        p.fillRect(r, qc("sunken"))
        if self._img is not None and not self._img.isNull():
            iw, ih = self._img.width(), self._img.height()
            s = max(r.width() / iw, r.height() / ih)
            tw, th = iw * s, ih * s
            p.drawImage(QRectF((r.width() - tw) / 2, (r.height() - th) / 2, tw, th), self._img)
        p.setClipping(False)
        if self.round:
            p.setPen(QPen(QColor(20, 32, 43, 20), 1))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(r.adjusted(0.5, 0.5, -0.5, -0.5))


class Segmented(QWidget):
    """Control segmentado: pista hundida + indicador blanco deslizante (animado 180 ms)."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], value: str | None = None, small: bool = False):
        super().__init__()
        self.options = options
        self.value = value or options[0][0]
        self.small = small
        self.setFixedHeight(28 if small else 32)
        self._fm_font = font(12 if small else 13, 500)
        self._x = 0.0
        self._anim = QPropertyAnimation(self, b"indicator", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.setCursor(Qt.PointingHandCursor)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self._fm_font)
        self._widths = [fm.horizontalAdvance(t) + (20 if small else 26) for _, t in options]
        self.setMinimumWidth(sum(self._widths) + 4)
        self._x = self._seg_rect(self._index()).x()

    def _index(self) -> int:
        return next((i for i, (k, _) in enumerate(self.options) if k == self.value), 0)

    def _seg_rects(self) -> list[QRectF]:
        total = sum(self._widths)
        extra = max(0, self.width() - 4 - total) / len(self._widths)
        x, out = 2.0, []
        for w in self._widths:
            out.append(QRectF(x, 2, w + extra, self.height() - 4))
            x += w + extra
        return out

    def _seg_rect(self, i: int) -> QRectF:
        return self._seg_rects()[i]

    def get_ind(self):
        return self._x

    def set_ind(self, v):
        self._x = v
        self.update()

    indicator = Property(float, get_ind, set_ind)

    def set_value(self, v: str, animate: bool = True, emit: bool = False) -> None:
        if v == self.value:
            return
        self.value = v
        target = self._seg_rect(self._index()).x()
        if animate:
            self._anim.stop()
            self._anim.setStartValue(self._x)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._x = target
        self.update()
        if emit:
            self.changed.emit(v)

    def resizeEvent(self, e):
        self._x = self._seg_rect(self._index()).x()

    def mouseReleaseEvent(self, e):
        for i, r in enumerate(self._seg_rects()):
            if r.contains(e.position()):
                self.set_value(self.options[i][0], emit=True)
                return

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("sunken"))
        p.drawRoundedRect(QRectF(self.rect()), 7, 7)
        rects = self._seg_rects()
        cur = rects[self._index()]
        ind = QRectF(self._x, cur.y(), cur.width(), cur.height())
        p.setBrush(QColor(20, 32, 43, 18))
        p.drawRoundedRect(ind.adjusted(-0.5, 0, 0.5, 1), 5.5, 5.5)
        p.setBrush(QColor("#FFFFFF"))
        p.drawRoundedRect(ind, 5, 5)
        p.setFont(self._fm_font)
        for i, ((k, t), r) in enumerate(zip(self.options, rects)):
            p.setPen(qc("ink") if k == self.value else qc("ink3"))
            p.drawText(r, Qt.AlignCenter, t)


class Toggle(QWidget):
    toggled = Signal(bool)

    def __init__(self, checked: bool = False):
        super().__init__()
        self.checked = checked
        self.setFixedSize(30, 18)
        self.setCursor(Qt.PointingHandCursor)

    def set_checked(self, v: bool) -> None:
        self.checked = v
        self.update()

    def mouseReleaseEvent(self, e):
        self.checked = not self.checked
        self.update()
        self.toggled.emit(self.checked)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(qc("accent") if self.checked else QColor("#CDD2D6"))
        p.drawRoundedRect(QRectF(self.rect()), 9, 9)
        p.setBrush(QColor("white"))
        x = 14 if self.checked else 2
        p.drawEllipse(QRectF(x, 2, 14, 14))


class Stepper(QWidget):
    def __init__(self, steps: list[str]):
        super().__init__()
        self.steps = steps
        self.current = 0
        self.setFixedHeight(24)
        self._font = font(13)
        self._font_b = font(13, 500)
        self._num = font(11, 600)
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(self._font_b)
        self._w = [20 + 8 + fm.horizontalAdvance(s) for s in steps]
        self.setMinimumWidth(sum(self._w) + 44 * (len(steps) - 1))

    def set_current(self, i: int) -> None:
        self.current = i
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        total = sum(self._w) + 44 * (len(self.steps) - 1)
        x = (self.width() - total) / 2
        cy = self.height() / 2
        for i, s in enumerate(self.steps):
            done, active = i < self.current, i == self.current
            circ = QRectF(x, cy - 10, 20, 20)
            p.setPen(Qt.NoPen)
            p.setBrush(qc("accent") if done else qc("ink") if active else qc("sunken"))
            p.drawEllipse(circ)
            if done:
                p.drawPixmap(QRect(int(x) + 4, int(cy) - 6, 12, 12), pixmap("check", "#FFFFFF", 12, 2.5))
            else:
                p.setFont(self._num)
                p.setPen(QColor("white") if active else qc("ink3"))
                p.drawText(circ, Qt.AlignCenter, str(i + 1))
            p.setFont(self._font_b if active else self._font)
            p.setPen(qc("ink") if active else qc("ink2") if done else qc("ink3"))
            p.drawText(QRectF(x + 28, 0, self._w[i] - 28, self.height()), Qt.AlignVCenter | Qt.AlignLeft, s)
            x += self._w[i]
            if i < len(self.steps) - 1:
                p.setPen(QPen(QColor(31, 90, 110, 120) if done else theme.LINE_STRONG, 1))
                p.drawLine(QPointF(x + 8, cy), QPointF(x + 36, cy))
                x += 44


class EmptyState(QWidget):
    def __init__(self, icon_name: str, title: str, detail: str = "", action: QWidget | None = None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 80, 0, 80)
        v.setSpacing(0)
        v.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        box = QLabel()
        box.setFixedSize(40, 40)
        box.setAlignment(Qt.AlignCenter)
        box.setPixmap(pixmap(icon_name, C["ink3"], 18, 1.5))
        box.setStyleSheet(f"background: {C['sunken']}; border-radius: 8px;")
        v.addWidget(box, 0, Qt.AlignHCenter)
        v.addSpacing(16)
        self.title = label(title, "body")
        self.title.setStyleSheet("font-size: 15px; font-weight: 500;")
        v.addWidget(self.title, 0, Qt.AlignHCenter)
        self.detail = label(detail, "muted")
        v.addSpacing(4)
        v.addWidget(self.detail, 0, Qt.AlignHCenter)  # siempre con padre: sin él, show() lo abre como ventana suelta
        self.detail.setVisible(bool(detail))
        if action:
            v.addSpacing(20)
            v.addWidget(action, 0, Qt.AlignHCenter)


def fade_in(w: QWidget, ms: int = 180) -> None:
    """Fade corto; el efecto se quita al terminar (no penaliza el repintado del video)."""
    eff = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(eff)
    a = QPropertyAnimation(eff, b"opacity", w)
    a.setDuration(ms)
    a.setStartValue(0.0)
    a.setEndValue(1.0)
    a.setEasingCurve(QEasingCurve.OutCubic)
    a.finished.connect(lambda: w.setGraphicsEffect(None))
    a.start(QPropertyAnimation.DeleteWhenStopped)


def discard(w: QWidget) -> None:
    """Elimina un widget SIN dejarlo suelto.

    Al agregar un widget a un contenedor visible, Qt agenda mostrarlo en el próximo ciclo de eventos. Si antes
    de ese ciclo se le quita el padre (setParent(None)), esa muestra pendiente lo abre como ventana suelta
    (pasaba en Reconocimiento, que rehace la lista en cada cuadro). Se oculta (cancela la muestra pendiente)
    y se borra conservando el padre hasta que Qt lo elimine.
    """
    w.hide()
    w.deleteLater()


def clear_layout(layout) -> None:
    while layout.count():
        it = layout.takeAt(0)
        w = it.widget()
        if w is not None:
            discard(w)
        elif it.layout() is not None:
            clear_layout(it.layout())
            it.layout().deleteLater()


def fmt_num(n: int) -> str:
    return f"{n:,}".replace(",", ".")
