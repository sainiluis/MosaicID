"""Comparar motores — la MISMA cara y el MISMO embedding evaluados por SVM y por Distancia.

Todo lo que se muestra es real: frame, detección YuNet (con landmarks), crop alineado por
SFace.alignCrop(), embedding de SFace.feature() y las dos decisiones. Se puede congelar
una muestra para explicarla con calma.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...services.analyzer import LiveAnalysis, LiveFace
from ..camera_view import CameraView
from ..pipeline_widget import EmbeddingBars
from ..theme import C, qc
from ..widgets import (Button, Divider, FaceThumb, Meter, Page, PageHeader, Row, bgr_to_qimage, clear_layout, label,
                       overline, pixmap, wrap)
from .common import LiveBinding, largest, pct, sim


class ImageCell(QWidget):
    """Celda de imagen del pipeline (frame con caja / recorte con landmarks / crop alineado)."""

    def __init__(self, w: int, h: int):
        super().__init__()
        self.setFixedSize(w, h)
        self.img: QImage | None = None

    def set_image(self, img: QImage | None) -> None:
        self.img = img
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.setClipPath(path)
        p.fillRect(r, qc("stage") if self.img is None else qc("sunken"))
        if self.img is not None and not self.img.isNull():
            s = min(r.width() / self.img.width(), r.height() / self.img.height())
            tw, th = self.img.width() * s, self.img.height() * s
            p.drawImage(QRectF((r.width() - tw) / 2, (r.height() - th) / 2, tw, th), self.img)


def frame_with_box(frame: np.ndarray, lf: LiveFace) -> QImage:
    f = frame.copy()
    d = lf.face.detection
    x, y, w, h = (int(round(v)) for v in d.box)
    t = max(2, f.shape[1] // 320)
    cv2.rectangle(f, (x, y), (x + w, y + h), (195, 235, 155), t)
    return bgr_to_qimage(f)


def crop_with_landmarks(frame: np.ndarray, lf: LiveFace) -> QImage:
    d = lf.face.detection
    x, y, w, h = d.box
    pad = 0.25
    x0, y0 = int(max(0, x - w * pad)), int(max(0, y - h * pad))
    x1, y1 = int(min(frame.shape[1], x + w * (1 + pad))), int(min(frame.shape[0], y + h * (1 + pad)))
    crop = frame[y0:y1, x0:x1].copy()
    s = 240 / max(1, max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.rectangle(crop, (int((x - x0) * s), int((y - y0) * s)), (int((x + w - x0) * s), int((y + h - y0) * s)),
                  (195, 235, 155), 1, cv2.LINE_AA)
    for (lx, ly) in d.landmarks:
        cv2.circle(crop, (int((lx - x0) * s), int((ly - y0) * s)), 3, (110, 203, 245), -1, cv2.LINE_AA)
    return bgr_to_qimage(crop)


class Fork(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(36)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(20, 32, 43, 46), 1))
        w, h = self.width(), self.height()
        cx, l, r = w / 2, w * 0.25, w * 0.75
        p.drawLine(QPointF(cx, 0), QPointF(cx, h / 2))
        p.drawLine(QPointF(l, h / 2), QPointF(r, h / 2))
        p.drawLine(QPointF(l, h / 2), QPointF(l, h))
        p.drawLine(QPointF(r, h / 2), QPointF(r, h))


class EngineColumn(QWidget):
    def __init__(self, title: str, quote: str, metric_name: str, list_title: str, left: bool):
        super().__init__()
        self.left = left
        v = QVBoxLayout(self)
        v.setContentsMargins(0 if left else 40, 0, 40 if left else 0, 0)
        v.setSpacing(0)
        v.addWidget(overline(title))
        v.addSpacing(4)
        v.addWidget(label(quote, "small"))
        v.addSpacing(18)
        self.pred_cap = label("", "meta")
        v.addWidget(self.pred_cap)
        self.pred = label("—", "title")
        self.pred.setStyleSheet("font-size: 26px;")
        v.addWidget(self.pred)
        v.addSpacing(14)
        h = QHBoxLayout()
        h.addWidget(label(metric_name, "muted"))
        h.addStretch(1)
        self.metric_lbl = label("", "body")
        self.metric_lbl.setStyleSheet("font-weight: 600;")
        h.addWidget(self.metric_lbl)
        v.addLayout(h)
        v.addSpacing(6)
        self.meter = Meter(0, "accent")
        v.addWidget(self.meter)
        v.addSpacing(4)
        self.thr = label("", "meta")
        v.addWidget(self.thr)
        v.addSpacing(18)
        v.addWidget(label(list_title, "meta"))
        v.addSpacing(4)
        self.list = QVBoxLayout()
        self.list.setSpacing(0)
        v.addLayout(self.list)
        v.addSpacing(6)
        self.note = label("", "meta", wrap=True)
        v.addWidget(self.note)
        v.addStretch(1)

    def paintEvent(self, _):
        if not self.left:
            p = QPainter(self)
            p.setPen(QColor(20, 32, 43, 23))
            p.drawLine(0, 0, 0, self.height())

    def set_rows(self, rows: list[tuple[str, float, bool]], marker: float | None, as_pct: bool = False) -> None:
        clear_layout(self.list)
        for name, v, hi in rows:
            r = Row(32, clickable=False)
            n = label(name, "small2")
            n.setFixedWidth(110)
            r.h.addWidget(n)
            r.h.addWidget(Meter(v, "accent" if hi else "idle", marker), 1)
            val = label(pct(v) if as_pct else f"{v:.2f}", "small")
            val.setFixedWidth(46)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r.h.addWidget(val)
            self.list.addWidget(r)


class CompareScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.frozen = False
        self.track: int | None = None
        self._last_ui = 0.0
        self.current: tuple[LiveAnalysis, LiveFace] | None = None
        self.btn_freeze = Button("Congelar muestra", "secondary", icon_name="snowflake")
        self.btn_freeze.clicked.connect(self._toggle_freeze)
        self.faces_row = QHBoxLayout()
        self.faces_row.setSpacing(4)
        self.add(PageHeader("Comparar motores",
                            "La misma cara y el mismo embedding, dos formas de decidir la identidad.",
                            [wrap(self.faces_row), self.btn_freeze]))

        # pipeline común con datos reales
        top = QHBoxLayout()
        top.addWidget(overline("Pipeline común"))
        top.addStretch(1)
        self.live_lbl = label("Se ejecuta una sola vez por rostro", "meta")
        top.addWidget(self.live_lbl)
        self.add(Divider())
        self.add(20)
        self.add(top)
        self.add(16)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        self.c_frame = ImageCell(220, 124)
        self.c_det = ImageCell(124, 124)
        self.c_align = ImageCell(124, 124)
        self.bars = EmbeddingBars(64)
        self.bars.setMinimumWidth(260)
        cells = [(self.c_frame, "Cámara", "frame real"), (self.c_det, "YuNet", ""), (self.c_align, "SFace.alignCrop()", "112 × 112"),
                 (self.bars, "SFace.feature()", "embedding de 128 valores")]
        self.cap = []
        col = 0
        for i, (w, t, d) in enumerate(cells):
            g.addWidget(w, 0, col, Qt.AlignBottom | Qt.AlignLeft)
            cap = QVBoxLayout()
            cap.setSpacing(0)
            tl = label(t, "small2")
            tl.setStyleSheet(f"color: {C['ink']}; font-weight: 500;")
            dl = label(d, "meta")
            cap.addWidget(tl)
            cap.addWidget(dl)
            self.cap.append(dl)
            g.addLayout(cap, 1, col)
            col += 1
            if i < len(cells) - 1:
                arrow = QLabel()
                arrow.setPixmap(pixmap("chevron-right", C["ink4"], 14))
                g.addWidget(arrow, 0, col, Qt.AlignCenter)
                col += 1
        g.setColumnStretch(col - 1, 1)
        self.add(g)
        self.add(10)
        self.emb_text = label("", "mono", wrap=True)
        self.add(self.emb_text)
        self.add(16)
        self.add(Divider())
        self.add(Fork())

        cols = QHBoxLayout()
        cols.setSpacing(0)
        self.svm_col = EngineColumn("Clasificador SVM", "Aprende fronteras entre identidades.", "Probabilidad (predict_proba)",
                                    "Probabilidad por clase", True)
        self.dist_col = EngineColumn("Distancia", "Compara contra referencias almacenadas.", "Similitud coseno (FR_COSINE)",
                                     "Similitud contra cada persona (mejor referencia)", False)
        cols.addWidget(self.svm_col, 1)
        cols.addWidget(self.dist_col, 1)
        self.add(cols)
        self.add(28)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        self.verdict.setTextFormat(Qt.RichText)
        self.add(self.verdict)
        self.stretch()
        self._hidden_cam = CameraView()  # recibe frames sin mostrarse (la vista es el pipeline)
        self.binding = LiveBinding(ctx, "compare", self._hidden_cam, None, self._on_analysis)

    def _toggle_freeze(self) -> None:
        self.frozen = not self.frozen
        self.btn_freeze.setText("Reanudar en vivo" if self.frozen else "Congelar muestra")
        self.btn_freeze.set_icon("play" if self.frozen else "snowflake")
        self.live_lbl.setText("Muestra congelada: todos los valores corresponden a este cuadro"
                              if self.frozen else "Se ejecuta una sola vez por rostro")

    def _select(self, tid: int) -> None:
        self.track = tid
        if self.current:
            self._render(*self.current, force_faces=True)

    def _on_analysis(self, a: LiveAnalysis) -> None:
        if self.frozen:
            return
        now = time.perf_counter()
        if now - self._last_ui < 0.25:  # 4 actualizaciones/s alcanzan para leer
            return
        self._last_ui = now
        ids = [f.track_id for f in a.faces]
        if self.track not in ids:
            lf = largest(a.faces)
            self.track = lf.track_id if lf else None
        lf = next((f for f in a.faces if f.track_id == self.track), None)
        self._render(a, lf)

    def _render(self, a: LiveAnalysis, lf: LiveFace | None, force_faces: bool = False) -> None:
        clear_layout(self.faces_row)
        for i, f in enumerate(a.faces):
            b = Button(f"Cara {i + 1}", "ghost" if f.track_id != self.track else "secondary", "sm")
            b.clicked.connect(lambda _=False, t=f.track_id: self._select(t))
            self.faces_row.addWidget(b)
        if lf is None or (force_faces and self.track != lf.track_id):
            lf = next((f for f in a.faces if f.track_id == self.track), lf)
        self.current = (a, lf) if lf else None
        snap = a.snapshot
        if lf is None:
            for c in (self.c_frame, self.c_det, self.c_align):
                c.set_image(None)
            self.bars.set_values(None)
            self.emb_text.setText("Mirá a la cámara: no se detecta ningún rostro." if self.ctx.camera.state == "live"
                                  else "Esperando la cámara…")
            self.verdict.hide()
            return
        f = lf.face
        self.c_frame.set_image(frame_with_box(a.frame, lf))
        self.c_det.set_image(crop_with_landmarks(a.frame, lf))
        self.c_align.set_image(bgr_to_qimage(f.aligned))
        self.bars.set_values(f.embedding)
        self.cap[1].setText(f"score {f.detection.score:.3f} · 5 landmarks")
        e = f.embedding
        first = ", ".join(f"{v:+.3f}" for v in e[:8])
        self.emb_text.setText(f"[{first}, …]   dim {e.shape[0]} · {e.dtype} · norma L2 {np.linalg.norm(e):.3f}")

        thr_d = self.ctx.settings.distance_cosine_threshold
        d = lf.decision.distance
        self.dist_col.pred_cap.setText("Mejor coincidencia")
        self.dist_col.pred.setText(snap.name(d.person_id) if not snap.matcher.empty else "Sin referencias")
        self.dist_col.metric_lbl.setText(sim(d.similarity))
        self.dist_col.meter.set(max(0.0, d.similarity), "accent" if d.known else "warn", thr_d)
        self.dist_col.thr.setText(f"Umbral {thr_d:.3f} · mejor candidato: {snap.name(d.candidate_id, '—')}")
        self.dist_col.set_rows([(snap.short(pid), max(0.0, s), s >= thr_d) for pid, s in d.ranking], thr_d)
        self.dist_col.note.setText(f"Estrategia: {self.ctx.settings.distance_strategy} · referencias: {len(snap.matcher.refs)}")

        thr_s = self.ctx.settings.svm_rejection_threshold
        s = lf.decision.svm
        if s is None:
            self.svm_col.pred_cap.setText("Predicción")
            self.svm_col.pred.setText("Sin modelo")
            self.svm_col.metric_lbl.setText("—")
            self.svm_col.meter.set(0, "idle", thr_s)
            self.svm_col.thr.setText("Entrená el clasificador para comparar.")
            self.svm_col.set_rows([], None)
            self.svm_col.note.setText("")
        else:
            self.svm_col.pred_cap.setText("Predicción")
            self.svm_col.pred.setText(snap.name(s.person_id))
            self.svm_col.metric_lbl.setText(pct(s.probability))
            self.svm_col.meter.set(s.probability, "accent" if s.known else "warn", s.threshold)
            self.svm_col.thr.setText(f"Rechazo bajo {s.threshold:.2f} ({len(s.probabilities)} clases, margen {thr_s:.2f})"
                                     f" · candidata: {snap.name(s.candidate_id, '—')}")
            self.svm_col.set_rows([(snap.short(pid), p, pid == s.person_id) for pid, p in s.probabilities], None, as_pct=True)
            missing = [snap.short(p) for p in snap.names if p not in self.ctx.svm.trained_person_ids]
            self.svm_col.note.setText(("Fuera del modelo: " + ", ".join(missing)) if missing else
                                      "Closed-set: las probabilidades siempre suman 1 entre las clases entrenadas.")

        # veredicto
        if s is None:
            tone, icon_, title, text = "idle", "info", "Solo un motor disponible", "El SVM todavía no fue entrenado."
        elif s.person_id == d.person_id:
            tone, icon_, title = "ok", "circle-check", "Ambos motores coinciden"
            text = (f"Los dos identifican a {snap.name(d.person_id)}." if d.person_id is not None
                    else "Ninguno supera su umbral: la persona se informa como desconocida.")
        else:
            tone, icon_, title = "warn", "triangle-alert", "Los motores no coinciden"
            if d.person_id is not None and d.person_id in snap.svm_pending:
                text = f"{snap.name(d.person_id)} ya tiene referencias, pero el clasificador todavía no fue reentrenado."
            elif s.person_id is not None and d.person_id is None:
                text = "El SVM elige una clase conocida (closed-set) aunque la similitud coseno no alcance el umbral."
            else:
                text = "Las métricas son distintas y cada motor aplica su propio umbral."
        bg = {"ok": C["ok_soft"], "warn": C["warn_soft"], "idle": C["sunken"]}[tone]
        col = {"ok": C["ok"], "warn": C["warn"], "idle": C["ink3"]}[tone]
        self.verdict.setText(f"<span style='color:{col};font-weight:600'>{title}</span>"
                             f"&nbsp;&nbsp;<span style='color:{C['ink2']}'>{text}</span>")
        self.verdict.setStyleSheet(f"background: {bg}; border-radius: 8px; padding: 12px 16px; font-size: 14px;")
        self.verdict.show()

    def on_enter(self, track: int | None = None, **_):
        if track is not None:
            self.track = track
        if self.frozen:
            self._toggle_freeze()
        self.binding.start()

    def on_leave(self):
        self.binding.stop()
