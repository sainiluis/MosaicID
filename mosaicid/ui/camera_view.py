"""Visor de cámara: frame real + bounding boxes + etiquetas compactas (no HUD)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QImage, QPainter, QPainterPath, QPen
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QWidget

from .theme import C, font, qc
from .widgets import bgr_to_qimage, label, pixmap


@dataclass
class Overlay:
    track_id: int
    box: tuple[float, float, float, float]  # en píxeles del frame
    label: str
    score: str = ""
    tone: str = "known"  # known | unknown | neutral
    selected: bool = False


TONE = {"known": QColor(C["box_known"]), "unknown": QColor(C["box_unknown"]), "neutral": QColor(255, 255, 255, 215)}


def camera_name(index: int) -> str:
    try:
        devs = QMediaDevices.videoInputs()
        if 0 <= index < len(devs):
            return devs[index].description()
    except Exception:
        pass
    return f"Cámara {index}"


class CameraView(QWidget):
    faceClicked = Signal(int)

    def __init__(self, aspect: float = 16 / 9, radius: int = 8):
        super().__init__()
        self.aspect = aspect
        self.radius = radius
        self.img: QImage | None = None
        self.frame_size = (0, 0)
        self.overlays: list[Overlay] = []
        self.state = "stopped"
        self.hint = ""
        self.message = ""
        self.guide: str | None = None  # None | neutral | ok | warn
        self.pills: list[tuple[bool, str]] = []
        self.corner_text = ""
        self.show_live_tag = True
        self.frozen = False
        self.flash = 0.0
        sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self._tag_font = font(11, 500)
        self._label_font = font(11.5, 500)
        self._score_font = font(11.5, 400)

    # tamaño con relación de aspecto fija
    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(w / self.aspect)

    def sizeHint(self) -> QSize:
        return QSize(960, int(960 / self.aspect))

    # ── datos ────────────────────────────────────────────────────────────
    def set_frame(self, frame: np.ndarray) -> None:
        if self.frozen:
            return
        self.img = bgr_to_qimage(frame)
        self.frame_size = (frame.shape[1], frame.shape[0])
        self.update()

    def set_overlays(self, ov: list[Overlay]) -> None:
        if self.frozen:
            return
        self.overlays = ov
        self.update()

    def set_hint(self, text: str) -> None:
        """Aviso sobre la imagen en vivo (p. ej. cámara tapada); vacío = sin aviso."""
        if text != self.hint:
            self.hint = text
            self.update()

    def set_state(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message
        if state in ("stopped", "error"):
            self.overlays = []
        if state == "stopped":
            self.img = None
        self.update()

    # ── geometría: "cover" para video horizontal; "contain" para vertical (celular parado) ──
    def _mapping(self) -> tuple[float, float, float]:
        fw, fh = self.frame_size
        if not fw:
            return 1.0, 0.0, 0.0
        # recortar un video vertical para llenar un recuadro 16:9 cortaría la cabeza: se muestra entero
        fit = min if fh > fw else max
        s = fit(self.width() / fw, self.height() / fh)
        return s, (self.width() - fw * s) / 2, (self.height() - fh * s) / 2

    def _box_rect(self, box) -> QRectF:
        s, ox, oy = self._mapping()
        x, y, w, h = box
        return QRectF(ox + x * s, oy + y * s, w * s, h * s)

    def mouseReleaseEvent(self, e):
        for o in self.overlays:
            if self._box_rect(o.box).contains(e.position()):
                self.faceClicked.emit(o.track_id)
                return

    # ── pintura ──────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, self.radius, self.radius)
        p.setClipPath(clip)
        p.fillRect(r, qc("stage"))

        if self.img is not None and self.state != "stopped":
            s, ox, oy = self._mapping()
            p.drawImage(QRectF(ox, oy, self.frame_size[0] * s, self.frame_size[1] * s), self.img)
            if self.state in ("opening", "error"):
                p.fillRect(r, QColor(10, 14, 18, 150))
        # viñeta sutil para legibilidad
        from PySide6.QtGui import QLinearGradient
        g = QLinearGradient(0, 0, 0, r.height())
        g.setColorAt(0, QColor(0, 0, 0, 70))
        g.setColorAt(0.22, QColor(0, 0, 0, 0))
        g.setColorAt(0.8, QColor(0, 0, 0, 0))
        g.setColorAt(1, QColor(0, 0, 0, 80))
        p.fillRect(r, g)

        if self.guide:
            self._paint_guide(p, r)

        if self.state == "live" or self.frozen:
            for o in self.overlays:
                self._paint_box(p, o)

        if self.flash > 0:
            p.fillRect(r, QColor(255, 255, 255, int(255 * self.flash)))

        # mensajes de estado
        if self.state != "live" and not self.frozen:
            p.setPen(QColor(255, 255, 255, 200))
            p.setFont(font(13))
            msg = {"opening": "Iniciando cámara…", "stopped": self.message or "Cámara detenida",
                   "error": self.message or "Cámara no disponible"}.get(self.state, "")
            ic = "video-off" if self.state != "opening" else "loader-circle"
            p.drawPixmap(int(r.center().x()) - 10, int(r.center().y()) - 30, pixmap(ic, "#D5D9DC", 20, 1.5))
            p.drawText(QRectF(40, r.center().y() - 4, r.width() - 80, 60), Qt.AlignHCenter | Qt.TextWordWrap, msg)

        if self.hint and self.state == "live" and not self.frozen:
            p.setPen(QColor(255, 255, 255, 215))
            p.setFont(font(13))
            p.drawPixmap(int(r.center().x()) - 10, int(r.center().y()) - 30, pixmap("video-off", "#D5D9DC", 20, 1.5))
            p.drawText(QRectF(40, r.center().y() - 4, r.width() - 80, 60), Qt.AlignHCenter | Qt.TextWordWrap, self.hint)

        if self.show_live_tag and self.state in ("live", "opening"):
            live = self.state == "live" and not self.frozen
            text = "Congelado" if self.frozen else ("En vivo" if live else "Conectando")
            self._tag(p, 12, 12, text, QColor("#FF6B5E") if live else QColor(255, 255, 255, 130))
        if self.corner_text:
            fm = QFontMetrics(self._tag_font)
            w = fm.horizontalAdvance(self.corner_text) + 14
            self._tag(p, r.width() - w - 12, r.height() - 32, self.corner_text, None)
        if self.pills:
            x = 12.0
            for ok, text in self.pills:
                x += self._pill(p, x, r.height() - 34, ok, text) + 6

    def _tag(self, p: QPainter, x: float, y: float, text: str, dot: QColor | None) -> None:
        fm = QFontMetrics(self._tag_font)
        w = fm.horizontalAdvance(text) + (22 if dot else 14)
        rr = QRectF(x, y, w, 20)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 115))
        p.drawRoundedRect(rr, 3, 3)
        tx = x + 7
        if dot:
            p.setBrush(dot)
            p.drawEllipse(QRectF(x + 7, y + 7.5, 5, 5))
            tx += 8
        p.setPen(QColor(255, 255, 255, 230))
        p.setFont(self._tag_font)
        p.drawText(QRectF(tx, y, w, 20), Qt.AlignVCenter, text)

    def _pill(self, p: QPainter, x: float, y: float, ok: bool, text: str) -> float:
        f = font(12, 500)
        fm = QFontMetrics(f)
        w = fm.horizontalAdvance(text) + 30
        rr = QRectF(x, y, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 125) if ok else QColor(240, 195, 106, 245))
        p.drawRoundedRect(rr, 3, 3)
        p.drawPixmap(int(x + 8), int(y + 5), pixmap("check" if ok else "triangle-alert",
                                                    C["box_known"] if ok else "#3A2A08", 12, 2.2))
        p.setPen(QColor(255, 255, 255, 235) if ok else QColor("#3A2A08"))
        p.setFont(f)
        p.drawText(QRectF(x + 24, y, w - 24, 22), Qt.AlignVCenter, text)
        return w

    def _paint_box(self, p: QPainter, o: Overlay) -> None:
        br = self._box_rect(o.box)
        col = TONE[o.tone]
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 70), 3.0))
        p.drawRoundedRect(br, 3, 3)
        if o.selected:
            p.setPen(QPen(QColor(255, 255, 255, 60), 6.0))
            p.drawRoundedRect(br.adjusted(-2, -2, 2, 2), 4, 4)
        p.setPen(QPen(col, 1.5))
        p.drawRoundedRect(br, 3, 3)
        if not o.label:
            return
        fm = QFontMetrics(self._label_font)
        fs = QFontMetrics(self._score_font)
        tw = fm.horizontalAdvance(o.label)
        sw = fs.horizontalAdvance(o.score) + 6 if o.score else 0
        w = 7 + 5 + 6 + tw + sw + 7
        h = 20.0
        x, y = br.x() - 0.5, br.y() - h - 4
        if y < 2:
            y = br.bottom() + 4
        lab = QRectF(x, y, w, h)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(lab, 3, 3)
        p.setBrush(col)
        p.drawEllipse(QRectF(x + 7, y + 7.5, 5, 5))
        p.setPen(QColor("white"))
        p.setFont(self._label_font)
        p.drawText(QRectF(x + 18, y, tw + 2, h), Qt.AlignVCenter, o.label)
        if o.score:
            p.setPen(QColor(255, 255, 255, 170))
            p.setFont(self._score_font)
            p.drawText(QRectF(x + 18 + tw + 6, y, sw, h), Qt.AlignVCenter, o.score)

    def _paint_guide(self, p: QPainter, r: QRectF) -> None:
        h = r.height() * 0.74
        w = h * 0.75
        oval = QRectF(r.center().x() - w / 2, r.center().y() - h / 2, w, h)
        mask = QPainterPath()
        mask.addRect(r)
        hole = QPainterPath()
        hole.addEllipse(oval)
        p.fillPath(mask.subtracted(hole), QColor(10, 14, 18, 70))
        col = {"ok": QColor(C["box_known"]), "warn": QColor(240, 195, 106, 240)}.get(self.guide, QColor(255, 255, 255, 190))
        p.setPen(QPen(col, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(oval)


class CameraMeta(QWidget):
    """Línea técnica bajo el video: dispositivo · resolución · FPS medido · inferencia · rostros."""

    def __init__(self, right: QWidget | None = None):
        super().__init__()
        self.setMinimumHeight(44)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 8, 0, 0)
        h.setSpacing(12)
        self.dev_icon = QWidget()
        from .widgets import icon_label
        self.dev = label("", "small2")
        self.res = label("", "small")
        self.fps = label("", "small")
        self.inf = label("", "small")
        self.faces = label("", "small")
        h.addWidget(icon_label("video", C["ink2"], 14))
        h.addSpacing(-6)
        self._sep_before: dict[QWidget, QWidget] = {}
        for i, w in enumerate((self.dev, self.res, self.fps, self.inf, self.faces)):
            if i:
                sep = QWidget()
                sep.setFixedSize(1, 12)
                sep.setStyleSheet(f"background: {C['line_strong']};")
                h.addWidget(sep)
                self._sep_before[w] = sep
            h.addWidget(w)
        h.addStretch(1)
        if right:
            h.addWidget(right)

    def set_picker(self, picker: QWidget, compact: bool = False) -> None:
        """Reemplaza el nombre fijo del dispositivo por un selector de cámara."""
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.dev), picker)
        self.dev.hide()
        if compact:  # la resolución y los rostros ya se muestran en el panel lateral
            for w in (self.res, self.faces):
                w.hide()
                self._sep_before[w].hide()

    def update_info(self, device: str, res: tuple[int, int], fps: float, infer_ms: float | None,
                    infer_fps: float | None, faces: int | None) -> None:
        self.dev.setText(device)
        self.res.setText(f"{res[0]} × {res[1]}" if res[0] else "—")
        self.fps.setText(f"{fps:.0f} FPS" if fps else "— FPS")
        self.inf.setText(f"Inferencia {infer_ms:.0f} ms · {infer_fps:.1f}/s" if infer_ms is not None else "Inferencia —")
        self.faces.setText(f"{faces} rostros" if faces is not None else "")
