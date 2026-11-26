"""Diagrama de pipeline continuo (nodos sobre una línea, sin cards) y gráfico del embedding."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import font, qc
from .widgets import pixmap
from .theme import C


class Pipeline(QWidget):
    def __init__(self, stages: list[tuple[str, str, str]], large: bool = False):
        super().__init__()
        self.stages = stages
        self.large = large
        self.active: int | None = None
        self.setMinimumHeight(118 if large else 82)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        n = len(self.stages)
        col_w = self.width() / n
        node = 40 if self.large else 32
        cy = node / 2 + 1
        p.setPen(QPen(QColor(20, 32, 43, 41), 1))
        p.drawLine(QPointF(col_w / 2, cy), QPointF(self.width() - col_w / 2, cy))
        tf = font(15 if self.large else 13, 500)
        df = font(13 if self.large else 12)
        for i, (ic, title, detail) in enumerate(self.stages):
            cx = col_w * i + col_w / 2
            on = self.active is None or i <= self.active
            r = QRectF(cx - node / 2, 1, node, node)
            p.setBrush(qc("canvas"))
            p.setPen(QPen(QColor(31, 90, 110, 95) if on else QColor(20, 32, 43, 30), 1))
            p.drawEllipse(r)
            isz = 18 if self.large else 15
            p.drawPixmap(int(cx - isz / 2), int(1 + node / 2 - isz / 2),
                         pixmap(ic, C["accent"] if on else C["ink4"], isz))
            if i < n - 1:
                ax = col_w * (i + 1)
                p.drawPixmap(int(ax - 6), int(cy - 6), pixmap("chevron-right", C["ink4"], 12))
            p.setPen(qc("ink"))
            p.setFont(tf)
            p.drawText(QRectF(cx - col_w / 2 + 6, node + 10, col_w - 12, 22), Qt.AlignHCenter | Qt.AlignTop, title)
            if detail:
                p.setPen(qc("ink3"))
                p.setFont(df)
                p.drawText(QRectF(cx - col_w / 2 + 10, node + 32, col_w - 20, 44),
                           Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, detail)


class EmbeddingBars(QWidget):
    """Los 128 valores reales del embedding como barras con signo respecto de cero.
    No sugiere ninguna estructura espacial: es solo el vector, en orden."""

    def __init__(self, height: int = 40):
        super().__init__()
        self.values: np.ndarray | None = None
        self.setFixedHeight(height)
        self.setMinimumWidth(128)

    def set_values(self, v: np.ndarray | None) -> None:
        self.values = None if v is None else np.asarray(v, np.float32)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        mid = h / 2
        p.setPen(QPen(QColor(20, 32, 43, 35), 1))
        p.drawLine(0, int(mid), w, int(mid))
        if self.values is None:
            return
        v = self.values
        m = float(np.max(np.abs(v))) or 1.0
        bw = w / len(v)
        p.setPen(Qt.NoPen)
        for i, x in enumerate(v):
            hh = (abs(x) / m) * (mid - 1)
            p.setBrush(QColor(31, 90, 110, 200) if x >= 0 else QColor(31, 90, 110, 110))
            y = mid - hh if x >= 0 else mid
            p.drawRect(QRectF(i * bw + 0.15 * bw, y, max(1.0, 0.7 * bw), max(0.5, hh)))
