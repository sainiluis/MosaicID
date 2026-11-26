"""Registrar persona — onboarding guiado full-page en 5 pasos.

Identidad → Preparación → Captura → Revisión → Finalizar

Cada muestra usa el ROSTRO COMPLETO (alignCrop de SFace). Solo se acepta una muestra
si pasa el control de calidad (core/quality.py). Las muestras inválidas no se guardan.
"""
from __future__ import annotations

import logging
import uuid

import cv2
from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit, QStackedWidget, QVBoxLayout, QWidget)

from ...app_context import AppContext
from ...core.quality import QualityReport, check_sample
from ...services.analyzer import LiveAnalysis
from ..camera_view import CameraView
from ..theme import C
from ..widgets import (Button, Divider, FaceThumb, Meter, Row, Stepper, clear_layout, icon_label, label, overline,
                       pixmap, wrap)
from .common import LiveBinding, sim, camera_picker

log = logging.getLogger(__name__)

STEPS = ["Identidad", "Preparación", "Captura", "Revisión", "Finalizar"]
CAPTURES = [
    ("Frontal", "Mirá al frente", "Rostro centrado, expresión neutra."),
    ("Izquierda", "Girá ligeramente a la izquierda", "Unos 15°, sin perder los ojos de vista."),
    ("Derecha", "Girá ligeramente a la derecha", "Unos 15°, hacia el otro lado."),
    ("Expresión", "Cambiá ligeramente la expresión", "Una sonrisa suave alcanza."),
    ("Cerca", "Acercate un poco", "Que el rostro ocupe más de la guía, completo."),
    ("Lejos", "Alejate ligeramente", "Dejá aire alrededor de la cabeza."),
    ("Frontal 2", "Frontal nuevamente", "Otra vez al frente, con luz pareja."),
    ("Libre", "Captura libre", "Cualquier posición natural."),
]
CHECKS = [("face", "Rostro detectado"), ("single", "Una sola persona en cuadro"), ("full", "Rostro completo y a buena distancia"),
          ("light", "Iluminación adecuada"), ("sharp", "Imagen nítida")]


def steps_for(n: int) -> list[tuple[str, str, str]]:
    out = list(CAPTURES[:n])
    while len(out) < n:
        out.append(("Libre", "Captura libre", "Cualquier posición natural."))
    return out


class CheckRow(Row):
    def __init__(self, text: str):
        super().__init__(44, clickable=False)
        self.ic = QLabel()
        self.ic.setFixedSize(16, 16)
        self.text = label(text, "body")
        self.note = label("", "small", "warn")
        self.h.addWidget(self.ic)
        self.h.addWidget(self.text)
        self.h.addStretch(1)
        self.h.addWidget(self.note)
        self.set("pending")

    def set(self, state: str, note: str = "") -> None:
        name, col = {"ok": ("circle-check", C["ok"]), "warn": ("triangle-alert", C["warn"]),
                     "pending": ("circle", C["ink4"])}[state]
        self.ic.setPixmap(pixmap(name, col, 16))
        self.text.setStyleSheet(f"color: {C['ink3'] if state == 'pending' else C['ink']};")
        self.note.setText(note)


class EnrollScreen(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"#Page {{ background: {C['canvas']}; }}")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # barra superior
        top = QWidget()
        top.setFixedHeight(56)
        th = QHBoxLayout(top)
        th.setContentsMargins(32, 0, 32, 0)
        t = label("Registrar persona", "body")
        t.setStyleSheet("font-weight: 600;")
        t.setFixedWidth(200)
        th.addWidget(t)
        self.stepper = Stepper(STEPS)
        th.addWidget(self.stepper, 1)
        self.btn_cancel = Button("Cancelar", "ghost", "sm", "x")
        self.btn_cancel.clicked.connect(self._cancel)
        cw = QWidget()
        cw.setFixedWidth(200)
        QHBoxLayout(cw).addWidget(self.btn_cancel, 0, Qt.AlignRight)
        cw.layout().setContentsMargins(0, 0, 0, 0)
        th.addWidget(cw)
        root.addWidget(top)
        root.addWidget(Divider())

        # cuerpo: páginas
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)
        self._build_camera_page()
        self._build_review_page()
        self._build_finish_page()

        # barra inferior
        root.addWidget(Divider())
        self.bottom = QWidget()
        self.bottom.setFixedHeight(64)
        bh = QHBoxLayout(self.bottom)
        bh.setContentsMargins(32, 0, 32, 0)
        self.btn_back = Button("Atrás", "ghost", icon_name="arrow-left")
        self.btn_back.clicked.connect(lambda: self._go(self.step - 1))
        bh.addWidget(self.btn_back)
        bh.addStretch(1)
        self.step_lbl = label("", "small")
        bh.addWidget(self.step_lbl)
        bh.addSpacing(16)
        self.btn_next = Button("Continuar", "primary", icon_right="arrow-right")
        self.btn_next.clicked.connect(self._next)
        bh.addWidget(self.btn_next)
        root.addWidget(self.bottom)

        self.binding = LiveBinding(ctx, "enroll", self.cam, None, self._on_analysis)
        sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc.activated.connect(self._capture)
        self._reset()

    # ── construcción ─────────────────────────────────────────────────────
    def _build_camera_page(self) -> None:
        page = QWidget()
        g = QGridLayout(page)
        g.setContentsMargins(32, 28, 32, 20)
        g.setHorizontalSpacing(40)
        g.setColumnStretch(0, 65)
        g.setColumnStretch(1, 35)
        left = QVBoxLayout()
        self.cam = CameraView()
        left.addWidget(self.cam)
        self.cam_caption = label("", "small", wrap=True)
        left.addSpacing(10)
        cap_row = QHBoxLayout()
        cap_row.addWidget(self.cam_caption, 1)
        cap_row.addWidget(camera_picker(self.ctx), 0, Qt.AlignTop)
        left.addLayout(cap_row)
        self.strip = QHBoxLayout()
        self.strip.setSpacing(6)
        self.strip_w = wrap(self.strip)
        left.addWidget(self.strip_w)
        left.addStretch(1)
        g.addLayout(left, 0, 0)
        self.panels = QStackedWidget()
        g.addWidget(self.panels, 0, 1, Qt.AlignTop)

        # panel identidad
        p0 = QWidget()
        v = QVBoxLayout(p0)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(label("¿Quién es?", "section"))
        v.addSpacing(4)
        v.addWidget(label("El nombre identifica a la persona en los dos motores y en la galería.", "small2", wrap=True))
        v.addSpacing(24)
        v.addWidget(label("Nombre y apellido", "small2"))
        v.addSpacing(6)
        self.name = QLineEdit()
        self.name.setProperty("size", "lg")
        self.name.setPlaceholderText("Ej.: Ana Pérez")
        self.name.textChanged.connect(self._update_buttons)
        self.name.returnPressed.connect(self._next)
        v.addWidget(self.name)
        v.addSpacing(8)
        self.name_err = label("", "small", "warn")
        v.addWidget(self.name_err)
        v.addSpacing(16)
        v.addWidget(label("En los próximos pasos se capturan muestras del rostro completo frente a la cámara. "
                          "Cada muestra genera un embedding de 128 valores que usan ambos motores.", "small", wrap=True))
        v.addStretch(1)
        self.panels.addWidget(p0)

        # panel preparación
        p1 = QWidget()
        v = QVBoxLayout(p1)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(label("Preparación", "section"))
        v.addSpacing(4)
        v.addWidget(label("Ubicate frente a la cámara, solo, con luz de frente.", "small2", wrap=True))
        v.addSpacing(20)
        self.checks = {k: CheckRow(t) for k, t in CHECKS}
        for r in self.checks.values():
            v.addWidget(r)
        v.addSpacing(14)
        self.similar = QLabel()
        self.similar.setWordWrap(True)
        self.similar.setStyleSheet(f"background: {C['warn_soft']}; border-radius: 6px; padding: 10px 12px; color: {C['ink2']}; font-size: 13px;")
        v.addWidget(self.similar)
        v.addStretch(1)
        self.panels.addWidget(p1)

        # panel captura
        p2 = QWidget()
        v = QVBoxLayout(p2)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        h = QHBoxLayout()
        h.addWidget(label("Captura guiada", "section"))
        h.addStretch(1)
        self.cap_count = label("", "small")
        h.addWidget(self.cap_count)
        v.addLayout(h)
        v.addSpacing(4)
        v.addWidget(label("Cada captura usa el rostro completo. Las posiciones solo buscan variedad.", "small", wrap=True))
        v.addSpacing(14)
        self.cap_list = QVBoxLayout()
        self.cap_list.setSpacing(2)
        v.addLayout(self.cap_list)
        v.addSpacing(16)
        row = QHBoxLayout()
        self.kbd = label("Espacio para capturar", "meta")
        row.addWidget(self.kbd)
        row.addStretch(1)
        self.btn_capture = Button("Capturar", "primary", icon_name="camera")
        self.btn_capture.clicked.connect(self._capture)
        row.addWidget(self.btn_capture)
        v.addLayout(row)
        v.addStretch(1)
        self.panels.addWidget(p2)
        self.pages.addWidget(page)

    def _build_review_page(self) -> None:
        page = QWidget()
        g = QGridLayout(page)
        g.setContentsMargins(32, 28, 32, 20)
        g.setHorizontalSpacing(40)
        g.setColumnStretch(0, 65)
        g.setColumnStretch(1, 35)
        left = QVBoxLayout()
        h = QHBoxLayout()
        h.addWidget(label("Revisión", "section"))
        h.addStretch(1)
        h.addWidget(label("Repetí una muestra si no te convence", "small"))
        left.addLayout(h)
        left.addSpacing(12)
        self.review_grid = QGridLayout()
        self.review_grid.setHorizontalSpacing(12)
        self.review_grid.setVerticalSpacing(14)
        left.addLayout(self.review_grid)
        left.addStretch(1)
        g.addLayout(left, 0, 0)
        right = QVBoxLayout()
        right.setSpacing(0)
        right.addWidget(overline("Resumen"))
        right.addSpacing(6)
        self.sum_rows = {}
        for k, t in (("n", "Muestras"), ("score", "Score YuNet medio"), ("sharp", "Nitidez media (Laplaciano)"),
                     ("emb", "Embeddings SFace")):
            r = Row(36, clickable=False)
            r.h.addWidget(label(t, "muted"))
            r.h.addStretch(1)
            val = label("", "body")
            r.h.addWidget(val)
            self.sum_rows[k] = val
            right.addWidget(r)
        right.addSpacing(16)
        right.addWidget(label("Al registrar, los embeddings se guardan como referencias del motor por distancia y se "
                              "agregan al dataset del clasificador SVM.", "small", wrap=True))
        right.addStretch(1)
        g.addLayout(right, 0, 1)
        self.pages.addWidget(page)

    def _build_finish_page(self) -> None:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(32, 40, 32, 20)
        box = QWidget()
        box.setMaximumWidth(760)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        h = QHBoxLayout()
        h.setSpacing(16)
        self.fin_avatar = FaceThumb(56)
        h.addWidget(self.fin_avatar)
        col = QVBoxLayout()
        col.setSpacing(2)
        ok = QHBoxLayout()
        ok.setSpacing(6)
        ok.addWidget(icon_label("circle-check", C["ok"], 14))
        ok.addWidget(label("Registro completo", "small", "ok"))
        ok.addStretch(1)
        col.addLayout(ok)
        self.fin_name = label("", "title")
        col.addWidget(self.fin_name)
        self.fin_meta = label("", "small2")
        col.addWidget(self.fin_meta)
        h.addLayout(col, 1)
        v.addLayout(h)
        v.addSpacing(28)
        v.addWidget(Divider())
        two = QHBoxLayout()
        two.setSpacing(0)
        self.fin_dist = self._engine_col("circle-check", C["ok"], "Motor distancia", "Disponible inmediatamente", "ok",
                                         "Sus embeddings ya se usan como referencias. No requiere entrenamiento.")
        self.fin_svm = self._engine_col("triangle-alert", C["warn"], "Clasificador SVM", "Requiere reentrenamiento", "warn",
                                        "")
        two.addWidget(self.fin_dist, 1)
        two.addWidget(Divider(vertical=True))
        two.addWidget(self.fin_svm, 1)
        v.addLayout(two)
        v.addWidget(Divider())
        v.addSpacing(28)
        btns = QHBoxLayout()
        self.btn_train_now = Button("Entrenar SVM ahora", "primary", icon_name="brain-circuit")
        self.btn_train_now.clicked.connect(lambda: self.ctx.navigate("training"))
        b2 = Button("Probar reconocimiento", "secondary")
        b2.clicked.connect(lambda: self.ctx.navigate("recognition"))
        b3 = Button("Ver persona", "ghost")
        b3.clicked.connect(lambda: self.ctx.navigate("person", person_id=self.saved_pid))
        for b in (self.btn_train_now, b2, b3):
            btns.addWidget(b)
        btns.addStretch(1)
        v.addLayout(btns)
        v.addStretch(1)
        outer.addStretch(1)
        outer.addWidget(box, 3)
        outer.addStretch(1)
        self.pages.addWidget(page)

    def _engine_col(self, ic, col, title, status, tone, text) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0 if tone == "ok" else 32, 20, 32 if tone == "ok" else 0, 20)
        v.setSpacing(4)
        h = QHBoxLayout()
        h.setSpacing(8)
        h.addWidget(icon_label(ic, col, 16))
        t = label(title, "body")
        t.setStyleSheet("font-size: 15px; font-weight: 500;")
        h.addWidget(t)
        h.addStretch(1)
        v.addLayout(h)
        w.status = label(status, "body", tone)
        w.status.setStyleSheet("font-weight: 500; padding-left: 24px;")
        v.addWidget(w.status)
        w.text = label(text, "small", wrap=True)
        w.text.setStyleSheet("padding-left: 24px;")
        v.addWidget(w.text)
        return w

    # ── ciclo de vida ────────────────────────────────────────────────────
    def _reset(self) -> None:
        self.step = 0
        self.person_id: int | None = None
        self.samples: list[dict] = []
        self.redo_index: int | None = None
        self.capture_requested = False
        self.ok_streak = 0
        self.prep_ready = False
        self.saved_pid: int | None = None
        self.warning: str | None = None
        self.name.setText("")
        self.name.setEnabled(True)
        self.name_err.setText("")

    def on_enter(self, person_id: int | None = None, **_):
        self._reset()
        if person_id is not None:
            p = self.ctx.db.get_person(person_id)
            if p:
                self.person_id = p.id
                self.name.setText(p.name)
                self.name.setEnabled(False)
        self.n_samples = self.ctx.settings.enrollment_sample_count
        self.capture_steps = steps_for(self.n_samples)
        self._go(0, animate=False)
        self.binding.start()
        QTimer.singleShot(50, self.name.setFocus)

    def on_leave(self):
        self.binding.stop()

    def _cancel(self) -> None:
        self.binding.stop()
        self.ctx.navigate("people")

    # ── navegación del asistente ─────────────────────────────────────────
    def _go(self, step: int, animate: bool = True) -> None:
        step = max(0, min(4, step))
        forward = step > self.step
        self.step = step
        self.stepper.set_current(step)
        page = 0 if step <= 2 else (1 if step == 3 else 2)
        if self.pages.currentIndex() != page:
            self.pages.setCurrentIndex(page)
            if animate:
                self._slide(self.pages.currentWidget(), forward)
        if step <= 2:
            self.panels.setCurrentIndex(step)
            if animate:
                self._slide(self.panels.currentWidget(), forward)
        if step == 1:
            self.prep_ready = False
            self.ok_streak = 0
        if step == 2:
            self._render_capture_list()
        if step == 3:
            self._render_review()
        self.cam.guide = None if step == 0 else "neutral"
        self.cam.pills = []
        self.cam.corner_text = ""
        self._update_buttons()
        self._render_strip()
        self.cam_caption.setVisible(step == 0)
        self.cam_caption.setText("Ubicate frente a la cámara. Se usará el rostro detectado para registrar la identidad."
                                 if step == 0 else "")
        self.btn_cancel.setVisible(step < 4)
        self.bottom.setVisible(step < 4)

    def _slide(self, w: QWidget, forward: bool) -> None:
        end = w.pos()
        start = QPoint(end.x() + (24 if forward else -24), end.y())
        w.move(start)
        a = QPropertyAnimation(w, b"pos", w)
        a.setDuration(200)
        a.setStartValue(start)
        a.setEndValue(end)
        a.setEasingCurve(QEasingCurve.OutCubic)
        a.start(QPropertyAnimation.DeleteWhenStopped)

    def _update_buttons(self) -> None:
        name = self.name.text().strip()
        err = ""
        if self.step == 0 and name and self.person_id is None:
            other = self.ctx.db.find_person_by_name(name)
            if other:
                err = f"Ya existe una persona llamada {other.name}."
        self.name_err.setText(err)
        ok = {0: len(name) >= 2 and not err, 1: self.prep_ready, 2: len(self.samples) >= self.n_samples and self.redo_index is None,
              3: len(self.samples) >= self.n_samples}.get(self.step, False)
        self.btn_next.setEnabled(ok)
        self.btn_next.setText("Registrar persona" if self.step == 3 else "Continuar")
        self.btn_back.setEnabled(self.step > 0)
        self.step_lbl.setText(f"Paso {self.step + 1} de {len(STEPS)}")

    def _next(self) -> None:
        if not self.btn_next.isEnabled():
            return
        if self.step == 3:
            if not self._save():
                return
            self.binding.stop()
        self._go(self.step + 1)

    # ── análisis en vivo ─────────────────────────────────────────────────
    def _on_analysis(self, a: LiveAnalysis) -> None:
        faces = [lf.face for lf in a.faces]
        rep, face = check_sample(faces, a.frame.shape, self.ctx.settings)
        from ..camera_view import Overlay
        if self.step == 0:
            self.cam.set_overlays([Overlay(lf.track_id, lf.face.detection.box,
                                           a.snapshot.short(lf.stable["distance"][0]), "",
                                           "known" if lf.stable["distance"][0] else "unknown") for lf in a.faces])
            return
        self.cam.set_overlays([])
        if self.step == 1:
            self._update_checks(rep, a)
        elif self.step == 2:
            self.cam.guide = "ok" if rep.ok else "warn"
            if self.capture_requested:
                self.capture_requested = False
                self.btn_capture.set_loading(False, "Capturar")
                if rep.ok and face is not None:
                    self._accept(face, rep)
                else:
                    self.warning = rep.message
            self.cam.pills = ([(False, self.warning)] if self.warning else
                              [(True, "Rostro detectado"), (True, "Buena calidad")] if rep.ok else [(False, rep.message)])
            idx = self.redo_index if self.redo_index is not None else len(self.samples)
            if idx < self.n_samples:
                self.cam.corner_text = f"{idx + 1} / {self.n_samples} · {self.capture_steps[idx][0]}"
            self.cam.update()

    def _update_checks(self, rep: QualityReport, a: LiveAnalysis) -> None:
        code = rep.code
        n = len(a.faces)
        self.checks["face"].set("ok" if n else "warn", "" if n else "No se detecta rostro")
        self.checks["single"].set("ok" if n == 1 else ("warn" if n > 1 else "pending"),
                                  rep.message if code == "multiple" else "")
        single = n == 1
        self.checks["full"].set("ok" if single and code not in ("small", "outside", "low_score") else
                                ("warn" if single else "pending"), rep.message if code in ("small", "outside", "low_score") else "")
        self.checks["light"].set("ok" if single and code not in ("dark", "bright", "small", "outside", "low_score") else
                                 ("warn" if code in ("dark", "bright") else "pending"),
                                 rep.message if code in ("dark", "bright") else "")
        self.checks["sharp"].set("ok" if rep.ok else ("warn" if code == "blur" else "pending"),
                                 rep.message if code == "blur" else "")
        self.ok_streak = self.ok_streak + 1 if rep.ok else 0
        self.cam.guide = "ok" if rep.ok else ("warn" if n else "neutral")
        self.cam.pills = [(True, "Rostro detectado"), (True, "Buena calidad")] if rep.ok else ([(False, rep.message)] if n else [])
        # ¿ya está registrado? (aviso, no bloquea)
        msg = ""
        if single and not a.snapshot.matcher.empty:
            d = a.faces[0].decision.distance
            if d.person_id is not None and d.person_id != self.person_id:
                msg = (f"Este rostro se parece a {a.snapshot.name(d.person_id)} (similitud {sim(d.similarity)}). "
                       "Si es la misma persona, no la registres dos veces: usá «Volver a capturar muestras».")
        self.similar.setText(msg)
        self.similar.setVisible(bool(msg))
        ready = self.ok_streak >= 3
        if ready != self.prep_ready:
            self.prep_ready = ready
            self._update_buttons()

    # ── captura ──────────────────────────────────────────────────────────
    def _capture(self) -> None:
        if self.step != 2 or not self.isVisible():
            return
        if len(self.samples) >= self.n_samples and self.redo_index is None:
            return
        self.warning = None
        self.capture_requested = True
        self.btn_capture.set_loading(True, "Capturando…")

    def _accept(self, face, rep: QualityReport) -> None:
        kind = self.capture_steps[self.redo_index if self.redo_index is not None else len(self.samples)][0]
        s = dict(kind=kind, aligned=face.aligned.copy(), embedding=face.embedding.copy(), quality_score=rep.quality,
                 detection_score=rep.score, sharpness=rep.sharpness)
        if self.redo_index is not None:
            self.samples[self.redo_index] = s
            self.redo_index = None
        else:
            self.samples.append(s)
        self._flash()
        self._render_capture_list()
        self._render_strip()
        self._update_buttons()
        if len(self.samples) >= self.n_samples and self.redo_index is None:
            self.btn_capture.setEnabled(False)

    def _flash(self) -> None:
        self.cam.flash = 0.55
        t = QTimer(self)

        def step():
            self.cam.flash = max(0.0, self.cam.flash - 0.12)
            self.cam.update()
            if self.cam.flash <= 0:
                t.stop()
                t.deleteLater()
        t.timeout.connect(step)
        t.start(30)

    def _render_strip(self) -> None:
        clear_layout(self.strip)
        if self.step != 2:
            self.strip_w.hide()
            return
        self.strip_w.show()
        for i in range(self.n_samples):
            th = FaceThumb(44, round_=False, radius=3)
            if i < len(self.samples):
                th.set_array(self.samples[i]["aligned"])
            self.strip.addWidget(th)
        self.strip.addStretch(1)

    def _render_capture_list(self) -> None:
        clear_layout(self.cap_list)
        cur = self.redo_index if self.redo_index is not None else len(self.samples)
        self.cap_count.setText(f"{len(self.samples)} de {self.n_samples}")
        for i, (kind, instr, hint) in enumerate(self.capture_steps):
            done = i < len(self.samples) and i != self.redo_index
            current = i == cur
            w = QWidget()
            w.setAttribute(Qt.WA_StyledBackground, True)
            if current:
                w.setStyleSheet(f"background: rgba(231,239,241,0.75); border-radius: 6px;")
            h = QHBoxLayout(w)
            h.setContentsMargins(10, 8, 6, 8)
            h.setSpacing(10)
            ic = QLabel()
            ic.setFixedSize(16, 16)
            if done:
                ic.setPixmap(pixmap("check", C["ok"], 14, 2.25))
            else:
                ic.setPixmap(pixmap("circle", C["accent"] if current else C["ink4"], 8 if not current else 10, 4))
            h.addWidget(ic, 0, Qt.AlignTop)
            col = QVBoxLayout()
            col.setSpacing(2)
            t = label(f"{i + 1}. {instr}", "body")
            t.setStyleSheet(f"color: {C['ink'] if current else C['ink2'] if done else C['ink3']};"
                            f" font-weight: {500 if current else 400}; background: transparent;")
            col.addWidget(t)
            if current:
                hl = label(hint, "small2")
                hl.setStyleSheet("background: transparent;")
                col.addWidget(hl)
            h.addLayout(col, 1)
            st = label("Lista" if done else "Actual" if current else "Pendiente", "meta")
            st.setStyleSheet(f"color: {C['accent'] if current else C['ink3'] if done else C['ink4']}; font-size: 12px; background: transparent;")
            h.addWidget(st, 0, Qt.AlignTop)
            self.cap_list.addWidget(w)
        full = len(self.samples) >= self.n_samples and self.redo_index is None
        self.btn_capture.setEnabled(not full)
        self.btn_capture.setText("Capturar" if not full else "Muestras completas")

    # ── revisión ─────────────────────────────────────────────────────────
    def _render_review(self) -> None:
        clear_layout(self.review_grid)
        for i, s in enumerate(self.samples):
            cell = QVBoxLayout()
            cell.setSpacing(6)
            th = FaceThumb(150, round_=False, radius=4)
            th.set_array(s["aligned"])
            cell.addWidget(th)
            cap = QHBoxLayout()
            cap.addWidget(label(f"{i + 1}. {s['kind']}", "meta"))
            cap.addStretch(1)
            redo = Button("Repetir", "quiet", "sm", "rotate-ccw")
            redo.clicked.connect(lambda _=False, k=i: self._redo(k))
            cap.addWidget(redo)
            cell.addLayout(cap)
            self.review_grid.addLayout(cell, i // 4, i % 4)
        n = len(self.samples) or 1
        self.sum_rows["n"].setText(str(len(self.samples)))
        self.sum_rows["score"].setText(f"{sum(s['detection_score'] for s in self.samples) / n:.3f}")
        self.sum_rows["sharp"].setText(f"{sum(s['sharpness'] for s in self.samples) / n:.0f}")
        self.sum_rows["emb"].setText(f"{len(self.samples)} × 128 float32")

    def _redo(self, i: int) -> None:
        self.redo_index = i
        self._go(2)

    # ── guardado ─────────────────────────────────────────────────────────
    def _save(self) -> bool:
        name = self.name.text().strip()
        try:
            pid = self.person_id if self.person_id is not None else self.ctx.db.create_person(name)
            folder = self.ctx.paths.samples / str(pid)
            folder.mkdir(parents=True, exist_ok=True)
            if self.person_id is not None:
                for old in folder.glob("*.jpg"):
                    old.unlink(missing_ok=True)
            rows = []
            for s in self.samples:
                fn = f"{uuid.uuid4().hex[:12]}.jpg"
                ok, buf = cv2.imencode(".jpg", s["aligned"], [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    (folder / fn).write_bytes(buf.tobytes())
                rows.append(dict(embedding=s["embedding"], kind=s["kind"], image_path=f"samples/{pid}/{fn}",
                                 quality_score=s["quality_score"], detection_score=s["detection_score"],
                                 sharpness=s["sharpness"]))
            self.ctx.db.add_samples(pid, rows, replace=self.person_id is not None)
        except Exception as e:
            log.exception("No se pudo registrar la persona")
            self.ctx.toast("No se pudo registrar", str(e), "warn")
            return False
        self.saved_pid = pid
        log.info("Persona %s registrada (id %d) con %d muestras", name, pid, len(rows))
        self.ctx.log("person", f"Se registró a {name}" if self.person_id is None else f"Nuevas muestras de {name}",
                     f"{len(rows)} muestras faciales · disponible por distancia")
        self.ctx.people_changed()
        self.ctx.toast("Persona registrada", f"{name} ya se reconoce por distancia.")
        # pantalla final
        self.fin_name.setText(name)
        self.fin_meta.setText(f"{len(rows)} muestras · {len(rows)} embeddings de 128 valores")
        self.fin_avatar.set_array(self.samples[0]["aligned"])
        people = len(self.ctx.db.list_people())
        if people < 2:
            self.fin_svm.status.setText("Todavía no se puede entrenar")
            self.fin_svm.text.setText("El SVM necesita al menos dos personas registradas para aprender fronteras entre identidades.")
            self.btn_train_now.setVisible(False)
        else:
            self.fin_svm.status.setText("Requiere reentrenamiento")
            self.fin_svm.text.setText(f"El SVM aprende fronteras entre identidades: hasta reentrenarlo, no conoce a "
                                      f"{name.split(' ')[0]}.")
            self.btn_train_now.setVisible(True)
        return True
