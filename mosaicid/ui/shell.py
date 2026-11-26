"""Ventana principal: sidebar compacto + área de trabajo (QStackedWidget) + toasts."""
from __future__ import annotations

import logging

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, Signal, QPoint
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMainWindow, QStackedWidget, QVBoxLayout,
                               QWidget)

from ..app_context import AppContext
from .theme import C, LINE, font, qc
from .widgets import Dot, fade_in, label, pixmap

log = logging.getLogger(__name__)

PRIMARY = [("home", "Inicio", "house"), ("recognition", "Reconocimiento", "scan-face"),
           ("people", "Personas", "users"), ("gallery", "Galería", "images")]
SECONDARY = [("compare", "Comparar motores", "git-compare-arrows"), ("how", "Cómo funciona", "workflow"),
             ("settings", "Configuración", "settings-2")]
# subrutas → ítem del sidebar
SECTION_OF = {"person": "people", "enroll": "people", "training": "people", "gallery_search": "gallery",
              "gallery_results": "gallery", "photo": "gallery"}


class Logo(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(24)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = 18 / 24
        cols = ["#1F5A6E", "#7FA6B4", "#7FA6B4", "#1F5A6E"]
        for (x, y), c in zip([(2, 2), (13, 2), (2, 13), (13, 13)], cols):
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c))
            p.drawRoundedRect(QRectF(8 + x * s, 3 + y * s, 9 * s, 9 * s), 1.5, 1.5)
        p.setPen(qc("ink"))
        p.setFont(font(15, 600))
        p.drawText(QRectF(36, 0, 200, 24), Qt.AlignVCenter, "MosaicID")


class NavItem(QWidget):
    clicked = Signal(str)

    def __init__(self, key: str, text: str, icon_name: str):
        super().__init__()
        self.key, self.text, self.icon_name = key, text, icon_name
        self.active = False
        self.hover = False
        self.badge = False
        self.kbd_focus = False
        self.setFixedHeight(32)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)

    def set_active(self, on: bool) -> None:
        self.active = on
        self.update()

    def focusInEvent(self, e):
        self.kbd_focus = e.reason() in (Qt.TabFocusReason, Qt.BacktabFocusReason)
        self.update()

    def focusOutEvent(self, e):
        self.kbd_focus = False
        self.update()

    def set_badge(self, on: bool) -> None:
        self.badge = on
        self.update()

    def enterEvent(self, e):
        self.hover = True
        self.update()

    def leaveEvent(self, e):
        self.hover = False
        self.update()

    def mouseReleaseEvent(self, e):
        self.clicked.emit(self.key)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.key)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        if self.active:
            p.setPen(QPen(QColor(20, 32, 43, 13), 1))
            p.setBrush(QColor("#FFFFFF"))
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        elif self.hover:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 10))
            p.drawRoundedRect(r, 6, 6)
        if self.hasFocus() and self.kbd_focus:
            p.setPen(QPen(QColor(31, 90, 110, 90), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 6, 6)
        ic = C["accent"] if self.active else (C["ink2"] if self.hover else C["ink3"])
        p.drawPixmap(8, 8, pixmap(self.icon_name, ic, 16))
        p.setPen(qc("ink") if self.active or self.hover else qc("ink2"))
        p.setFont(font(13.5, 500 if self.active else 400))
        p.drawText(QRectF(34, 0, r.width() - 40, r.height()), Qt.AlignVCenter, self.text)
        if self.badge:
            p.setPen(Qt.NoPen)
            p.setBrush(qc("warn"))
            p.drawEllipse(QRectF(r.width() - 14, r.height() / 2 - 3, 6, 6))


class Sidebar(QWidget):
    navigate = Signal(str)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(208)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {C['rail']};")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 20, 12, 16)
        v.setSpacing(2)
        v.addWidget(Logo())
        v.addSpacing(22)
        self.items: dict[str, NavItem] = {}
        for group in (PRIMARY, SECONDARY):
            for key, text, ic in group:
                it = NavItem(key, text, ic)
                it.clicked.connect(self.navigate)
                self.items[key] = it
                v.addWidget(it)
            if group is PRIMARY:
                v.addSpacing(10)
                sep = QWidget()
                sep.setFixedHeight(1)
                sep.setStyleSheet(f"background: {C['line']}; margin: 0 8px;")
                v.addWidget(sep)
                v.addSpacing(10)
        v.addStretch(1)
        foot = QVBoxLayout()
        foot.setContentsMargins(8, 0, 0, 0)
        foot.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(Dot("ok"), 0, Qt.AlignVCenter)
        t = label("Procesamiento local", "meta")
        t.setStyleSheet(f"color: {C['ink2']}; font-weight: 500;")
        row.addWidget(t)
        row.addStretch(1)
        foot.addLayout(row)
        sub = label("OpenCV 5 · YuNet · SFace", "meta")
        sub.setStyleSheet(f"font-size: 11px; color: {C['ink3']}; padding-left: 14px;")
        foot.addWidget(sub)
        v.addLayout(foot)
        self._indicator = QWidget(self)
        self._indicator.setStyleSheet(f"background: {C['accent']}; border-radius: 1px;")
        self._indicator.resize(2, 16)
        self._anim = QPropertyAnimation(self._indicator, b"pos", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.current = None

    def set_current(self, key: str) -> None:
        key = SECTION_OF.get(key, key)
        for k, it in self.items.items():
            it.set_active(k == key)
        self.current = key
        QTimer.singleShot(0, self._move_indicator)

    def _move_indicator(self) -> None:
        it = self.items.get(self.current)
        if not it:
            self._indicator.hide()
            return
        target = QPoint(0, it.y() + 8)
        self._indicator.show()
        if self._indicator.pos().y() == 0:
            self._indicator.move(target)
            return
        self._anim.stop()
        self._anim.setStartValue(self._indicator.pos())
        self._anim.setEndValue(target)
        self._anim.start()

    def resizeEvent(self, e):
        self._move_indicator()

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setPen(LINE)
        p.drawLine(self.width() - 1, 0, self.width() - 1, self.height())


class Toast(QWidget):
    def __init__(self, parent, title: str, detail: str, tone: str):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: #1B252E; border-radius: 7px;")
        self.setFixedWidth(320)
        h = QHBoxLayout(self)
        h.setContentsMargins(14, 12, 14, 12)
        h.setSpacing(12)
        ic = QLabel()
        ic.setPixmap(pixmap({"warn": "triangle-alert", "info": "info"}.get(tone, "check"),
                            {"warn": "#E9C27A", "info": "#C9CED2"}.get(tone, "#8FD3B0"), 15, 2))
        ic.setStyleSheet("background: transparent;")
        h.addWidget(ic, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("color: white; font-weight: 500; background: transparent;")
        v.addWidget(t)
        if detail:
            d = QLabel(detail)
            d.setWordWrap(True)
            d.setStyleSheet("color: rgba(255,255,255,0.65); font-size: 13px; background: transparent;")
            v.addWidget(d)
        h.addLayout(v, 1)
        self.adjustSize()


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, screens_factory):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("MosaicID")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 700)
        root = QWidget()
        root.setObjectName("Main")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.sidebar = Sidebar()
        self.stack = QStackedWidget()
        h.addWidget(self.sidebar)
        h.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.screens: dict[str, QWidget] = {}
        self.factory = screens_factory
        self.current: str | None = None
        self.toasts: list[Toast] = []
        self.sidebar.navigate.connect(lambda k: self.go(k))
        ctx.navigateRequested.connect(lambda r, p: self.go(r, **p))
        ctx.toastRequested.connect(self.show_toast)
        ctx.svmChanged.connect(self._badges)
        ctx.peopleChanged.connect(self._badges)
        self._badges()
        self.go("home")

    def _badges(self) -> None:
        self.sidebar.items["people"].set_badge(self.ctx.svm_status() == "dirty")

    def screen(self, key: str):
        if key not in self.screens:
            w = self.factory(key, self.ctx)
            self.screens[key] = w
            self.stack.addWidget(w)
        return self.screens[key]

    def go(self, key: str, **params) -> None:
        try:
            prev = self.screens.get(self.current) if self.current else None
            nxt = self.screen(key)
            if prev is not None and prev is not nxt and hasattr(prev, "on_leave"):
                prev.on_leave()
            self.current = key
            self.sidebar.set_current(key)
            if hasattr(nxt, "on_enter"):
                nxt.on_enter(**params)
            if self.stack.currentWidget() is not nxt:
                self.stack.setCurrentWidget(nxt)
                fade_in(nxt, 170)
        except Exception:
            log.exception("Error al navegar a %s", key)
            self.show_toast("Ocurrió un error", "El detalle quedó registrado en logs/mosaicid.log.", "warn")

    def show_toast(self, title: str, detail: str = "", tone: str = "ok") -> None:
        t = Toast(self, title, detail, tone)
        self.toasts = [x for x in self.toasts if x.isVisible()][-2:] + [t]
        self._place_toasts()
        t.show()
        t.raise_()
        fade_in(t, 160)
        QTimer.singleShot(3600, lambda: self._drop_toast(t))

    def _drop_toast(self, t: Toast) -> None:
        if t in self.toasts:
            self.toasts.remove(t)
        t.hide()
        t.deleteLater()
        self._place_toasts()

    def _place_toasts(self) -> None:
        y = self.height() - 20
        for t in reversed(self.toasts):
            y -= t.height()
            t.move(self.width() - t.width() - 20, y)
            y -= 8

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_toasts()

    def closeEvent(self, e):
        for s in self.screens.values():
            if hasattr(s, "on_leave"):
                try:
                    s.on_leave()
                except Exception:
                    pass
        self.ctx.shutdown()
        super().closeEvent(e)
