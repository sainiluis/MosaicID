"""Reconocimiento — cámara grande + inspector técnico lateral."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...config import camera_label
from ...services.analyzer import LiveAnalysis, LiveFace
from ...core.svm import effective_threshold
from ..camera_view import CameraMeta, CameraView, camera_name
from ..theme import C
from ..widgets import (Button, Divider, FaceThumb, LinkLabel, Meter, Page, PageHeader, PropRow, Row, Segmented,
                       clear_layout, discard, label, overline, status_line, wrap)
from .common import LiveBinding, overlays_for, pct, sim, camera_picker


class DetectionRow(QWidget):
    def __init__(self, on_select):
        super().__init__()
        self.track_id = -1
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.row = Row(48, separator=False, radius=6)
        self.row.h.setContentsMargins(8, 0, 8, 0)
        self.thumb = FaceThumb(30)
        self.name = label("", "body")
        self.name.setStyleSheet("font-weight: 500;")
        self.state = label("", "meta")
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(self.name)
        col.addWidget(self.state)
        self.metric_lbl = label("", "small2")
        self.metric_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.row.h.addWidget(self.thumb)
        self.row.h.addLayout(col, 1)
        self.row.h.addWidget(self.metric_lbl)
        self.row.clicked.connect(lambda: on_select(self.track_id))
        v.addWidget(self.row)
        self.detail = QWidget()
        self.dl = QVBoxLayout(self.detail)
        self.dl.setContentsMargins(50, 2, 8, 10)
        self.dl.setSpacing(6)
        v.addWidget(self.detail)
        self.detail.hide()

    def update_face(self, lf: LiveFace, engine: str, snap, selected: bool, threshold: float) -> None:
        self.track_id = lf.track_id
        self.row.set_selected(selected)
        self.thumb.set_array(lf.face.aligned)
        dp, dm = lf.stable["distance"]
        sp, sm = lf.stable["svm"]
        has_svm = lf.decision.svm is not None
        if engine == "distance":
            known = dp is not None
            self.name.setText(snap.name(dp))
            self.metric_lbl.setText(sim(dm))
            self.state.setText("Reconocido" if known else "Sin coincidencia suficiente")
        elif engine == "svm":
            known = has_svm and sp is not None
            self.name.setText(snap.name(sp) if has_svm else "Sin modelo")
            self.metric_lbl.setText(pct(sm) if has_svm else "—")
            self.state.setText("Reconocido" if known else ("Confianza bajo el umbral" if has_svm else "Entrená el SVM"))
        else:
            agree = has_svm and sp == dp
            known = agree and dp is not None
            self.name.setText(snap.name(dp) if agree else "Sin acuerdo")
            self.metric_lbl.setText(f"{pct(sm) if has_svm else '—'}  ·  {sim(dm)}")
            self.state.setText(("Ambos motores coinciden" if dp is not None else "Ninguno lo reconoce")
                               if agree else "Los motores no coinciden")
            known = agree
        self.state.setStyleSheet(f"color: {C['ink3'] if known else C['warn']}; font-size: 12px;")
        show = selected and engine != "svm"
        self.detail.setVisible(show)
        if show:
            clear_layout(self.dl)
            if engine == "compare":
                g = QHBoxLayout()
                for t, v in (("SVM", snap.name(sp) if has_svm else "Sin modelo"), ("Distancia", snap.name(dp))):
                    c = QVBoxLayout()
                    c.setSpacing(0)
                    c.addWidget(label(t, "meta"))
                    n = label(v, "small2")
                    n.setStyleSheet(f"color: {C['ink']}; font-weight: 500;")
                    c.addWidget(n)
                    g.addLayout(c)
                self.dl.addLayout(g)
            for pid, s in lf.decision.distance.ranking[:4]:
                top = QHBoxLayout()
                n = label(snap.short(pid), "meta")
                if s >= threshold:
                    n.setStyleSheet(f"color: {C['ink']}; font-size: 12px;")
                top.addWidget(n)
                top.addStretch(1)
                top.addWidget(label(sim(s), "meta"))
                self.dl.addLayout(top)
                self.dl.addWidget(Meter(s, "accent" if s >= threshold else "idle", threshold))
            self.dl.addWidget(label(f"Umbral {threshold:.3f} · último cuadro", "meta"))


class RecognitionScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__(max_width=1560)
        self.ctx = ctx
        self.engine = "distance"
        self.selected: int | None = None
        self.paused = False
        self.seg = Segmented([("svm", "SVM"), ("distance", "Distancia"), ("compare", "Comparar")], "distance")
        self.seg.changed.connect(self._set_engine)
        self.add(PageHeader("Reconocimiento", "Identificación en vivo con detección YuNet y embeddings SFace.",
                            [self.seg]))

        grid = QGridLayout()
        grid.setHorizontalSpacing(32)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 320)

        left = QVBoxLayout()
        self.cam = CameraView()
        self.cam.faceClicked.connect(self._select)
        left.addWidget(self.cam)
        self.btn_pause = Button("Pausar", "ghost", "sm", "pause")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop = Button("Detener", "ghost", "sm", "square")
        self.btn_stop.clicked.connect(self._toggle_stop)
        self.meta = CameraMeta(wrap(QHBoxLayout()))
        self.meta.layout().addWidget(self.btn_pause)
        self.meta.layout().addWidget(self.btn_stop)
        self.meta.set_picker(camera_picker(ctx), compact=True)
        left.addWidget(self.meta)
        left.addStretch(1)
        grid.addLayout(left, 0, 0)

        side = QWidget()
        side.setFixedWidth(320)
        s = QVBoxLayout(side)
        s.setContentsMargins(0, 0, 0, 0)
        s.setSpacing(0)
        s.addWidget(overline("Sesión actual"))
        s.addSpacing(6)
        self.p_cam = PropRow("Cámara")
        self.p_res = PropRow("Resolución")
        self.p_fps = PropRow("FPS de cámara (medido)")
        self.p_inf = PropRow("Inferencia")
        self.p_faces = PropRow("Rostros")
        for w in (self.p_cam, self.p_res, self.p_fps, self.p_inf, self.p_faces):
            s.addWidget(w)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(14)
        s.addWidget(overline("Motor"))
        s.addSpacing(6)
        self.p_engine = PropRow("Comparación")
        self.p_yunet = PropRow("YuNet", status_line("Activo", "ok", "body"))
        self.p_sface = PropRow("SFace", status_line("Activo", "ok", "body"))
        self.p_thr_svm = PropRow("Rechazo SVM")
        self.p_thr_dist = PropRow("Umbral coseno")
        for w in (self.p_engine, self.p_yunet, self.p_sface, self.p_thr_svm, self.p_thr_dist):
            s.addWidget(w)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(14)
        head = QHBoxLayout()
        head.addWidget(overline("Detecciones"))
        head.addStretch(1)
        self.compare_link = LinkLabel("Ver comparación →", "meta")
        self.compare_link.setStyleSheet(f"color: {C['accent']}; font-size: 12px;")
        self.compare_link.clicked.connect(lambda: ctx.navigate("compare", track=self.selected))
        head.addWidget(self.compare_link)
        s.addLayout(head)
        s.addSpacing(4)
        self.det_box = QVBoxLayout()
        self.det_box.setSpacing(0)
        s.addLayout(self.det_box)
        self.empty = label("Esperando el primer cuadro…", "small")
        s.addWidget(self.empty)
        s.addSpacing(14)
        self.notice = QLabel()
        self.notice.setWordWrap(True)
        self.notice.setTextFormat(Qt.RichText)
        self.notice.setStyleSheet(f"background: {C['warn_soft']}; border-radius: 6px; padding: 10px 12px;"
                                  f" color: {C['ink2']}; font-size: 13px;")
        self.notice.linkActivated.connect(lambda _: ctx.navigate("training"))
        s.addWidget(self.notice)
        s.addStretch(1)
        grid.addWidget(side, 0, 1, Qt.AlignTop)
        self.add(grid)
        self.stretch()
        self.rows: list[DetectionRow] = []
        self.binding = LiveBinding(ctx, "recognition", self.cam, self.meta, self._on_analysis)

    # ── estado ───────────────────────────────────────────────────────────
    def _set_engine(self, e: str) -> None:
        self.engine = e
        self._refresh_static()
        if self.binding.last:
            self._on_analysis(self.binding.last)

    def _select(self, tid: int) -> None:
        self.selected = tid
        if self.binding.last:
            self._on_analysis(self.binding.last, force=True)

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.cam.frozen = self.paused
        self.cam.update()
        self.btn_pause.setText("Reanudar" if self.paused else "Pausar")
        self.btn_pause.set_icon("play" if self.paused else "pause")

    def _toggle_stop(self) -> None:
        if self.binding.active:
            self.binding.stop()
            self.cam.set_state("stopped", "Cámara detenida")
            self.btn_stop.setText("Iniciar")
            self.btn_stop.set_icon("play")
            self.btn_pause.setEnabled(False)
            self._clear_rows()
        else:
            self.binding.start()
            self.btn_stop.setText("Detener")
            self.btn_stop.set_icon("square")
            self.btn_pause.setEnabled(True)

    def _clear_rows(self) -> None:
        for r in self.rows:
            discard(r)
        self.rows = []
        self.empty.setText("Sin detecciones mientras la cámara no está activa.")
        self.empty.show()

    def _refresh_static(self) -> None:
        st = self.ctx.settings
        self.p_engine.set({"svm": "Clasificador SVM", "distance": "Similitud coseno", "compare": "Ambos motores"}[self.engine])
        self.p_thr_svm.setVisible(self.engine != "distance")
        self.p_thr_dist.setVisible(self.engine != "svm")
        k = self.ctx.svm.n_classes
        self.p_thr_svm.set(f"{effective_threshold(st.svm_rejection_threshold, k):.2f} ({k} clases)" if k
                           else f"margen {st.svm_rejection_threshold:.2f}")
        self.p_thr_dist.set(f"{st.distance_cosine_threshold:.3f}")
        self.compare_link.setVisible(self.engine == "compare")
        status = self.ctx.svm_status()
        snap = self.ctx.snapshot
        msg = ""
        if self.engine != "distance":
            if status == "dirty":
                names = ", ".join(snap.short(p) for p in sorted(snap.svm_pending)) or "Hay cambios"
                msg = (f"Clasificador pendiente de reentrenamiento: {names} no está(n) en el modelo. "
                       f"<a href='t' style='color:{C['accent']};font-weight:500;text-decoration:none'>Entrenar SVM</a>")
            elif status in ("untrained", "empty"):
                msg = ("El clasificador todavía no fue entrenado. "
                       f"<a href='t' style='color:{C['accent']};font-weight:500;text-decoration:none'>Entrenar SVM</a>")
        if self.engine != "svm" and snap.matcher.empty:
            msg = "No hay personas registradas: todas las caras se informan como desconocidas."
        self.notice.setText(msg)
        self.notice.setVisible(bool(msg))

    def _on_analysis(self, a: LiveAnalysis, force: bool = False) -> None:
        if self.paused and not force:
            return
        cam = self.ctx.camera
        self.p_cam.set(camera_label(self.ctx.settings))
        self.p_res.set(f"{cam.resolution[0]} × {cam.resolution[1]}")
        self.p_fps.set(f"{cam.fps:.1f}")
        self.p_inf.set(f"{a.inference_ms:.0f} ms · {a.inference_fps:.1f}/s")
        self.p_faces.set(str(len(a.faces)))
        ids = [f.track_id for f in a.faces]
        if self.selected not in ids:
            self.selected = ids[0] if ids else None
        self.cam.set_overlays(overlays_for(a, self.engine, self.selected))
        while len(self.rows) < len(a.faces):
            r = DetectionRow(self._select)
            self.rows.append(r)
            self.det_box.addWidget(r)
        while len(self.rows) > len(a.faces):
            discard(self.rows.pop())
        for r, lf in zip(self.rows, a.faces):
            r.update_face(lf, self.engine, a.snapshot, lf.track_id == self.selected, self.ctx.settings.distance_cosine_threshold)
        self.empty.setVisible(not a.faces)
        self.empty.setText("No se detectan rostros.")

    def on_enter(self, **_):
        self._refresh_static()
        self.paused = False
        self.cam.frozen = False
        self.btn_pause.setText("Pausar")
        self.btn_stop.setText("Detener")
        self.btn_stop.set_icon("square")
        self.btn_pause.setEnabled(True)
        self.binding.start()

    def on_leave(self):
        self.binding.stop()
