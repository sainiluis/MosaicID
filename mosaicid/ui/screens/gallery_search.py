"""Buscar en galería: persona registrada o escaneo del rostro actual por cámara."""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...core.quality import check_sample
from ...services.analyzer import LiveAnalysis
from ..camera_view import CameraView
from ..theme import C
from ..widgets import (BackLink, Button, Divider, FaceThumb, Page, PageHeader, Row, clear_layout, icon_label, label,
                       overline)
from .common import LiveBinding, sim, camera_picker
from .people import person_avatar_path


class GallerySearchScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        back = BackLink("Galería")
        back.clicked.connect(lambda: ctx.navigate("gallery"))
        self.add(PageHeader("Buscar en galería", "Elegí a una persona registrada o escaneá un rostro frente a la cámara.",
                            back=back))
        self.add(Divider())
        cols = QHBoxLayout()
        cols.setSpacing(0)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 24, 40, 0)
        lv.setSpacing(0)
        lv.addWidget(overline("Personas registradas"))
        lv.addSpacing(8)
        self.people = QVBoxLayout()
        self.people.setSpacing(0)
        lv.addLayout(self.people)
        lv.addStretch(1)
        cols.addWidget(left, 1)
        cols.addWidget(Divider(vertical=True))
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(40, 24, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(overline("Escanear rostro"))
        rv.addSpacing(8)
        self.cam = CameraView()
        self.cam.guide = "neutral"
        rv.addWidget(self.cam)
        rv.addSpacing(8)
        rv.addWidget(camera_picker(ctx), 0, Qt.AlignLeft)
        rv.addSpacing(8)
        row = QHBoxLayout()
        self.msg = label("Mirá a la cámara y presioná Escanear.", "small")
        row.addWidget(self.msg, 1)
        self.btn = Button("Escanear", "primary", icon_name="scan-face")
        self.btn.clicked.connect(self._scan)
        row.addWidget(self.btn)
        rv.addLayout(row)
        rv.addStretch(1)
        cols.addWidget(right, 1)
        self.add(cols)
        self.stretch()
        self.scan_requested = False
        self.binding = LiveBinding(ctx, "gallery_search", self.cam, None, self._on_analysis)

    def refresh(self) -> None:
        clear_layout(self.people)
        people = self.ctx.db.list_people()
        if not people:
            self.people.addWidget(label("No hay personas registradas.", "muted"))
        for p in people:
            r = Row(56, radius=6)
            r.h.setContentsMargins(8, 0, 8, 0)
            av = FaceThumb(34)
            av.set_path(person_avatar_path(self.ctx, p.id))
            r.h.addWidget(av)
            c = QVBoxLayout()
            c.setSpacing(0)
            n = label(p.name, "body")
            n.setStyleSheet("font-weight: 500;")
            c.addWidget(n)
            c.addWidget(label(f"{p.sample_count} embeddings de referencia", "meta"))
            r.h.addLayout(c, 1)
            r.h.addWidget(icon_label("chevron-right", C["ink4"], 15))
            r.clicked.connect(lambda pid=p.id: self.ctx.navigate("gallery_results", person_id=pid))
            self.people.addWidget(r)

    def _scan(self) -> None:
        self.scan_requested = True
        self.btn.set_loading(True, "Escaneando…")
        self.msg.setText("Generando embedding del rostro…")

    def _on_analysis(self, a: LiveAnalysis) -> None:
        rep, face = check_sample([lf.face for lf in a.faces], a.frame.shape, self.ctx.settings)
        self.cam.guide = "ok" if rep.ok else ("warn" if a.faces else "neutral")
        self.cam.pills = [] if rep.ok or not a.faces else [(False, rep.message)]
        if not self.scan_requested:
            return
        self.scan_requested = False
        self.btn.set_loading(False, "Escanear")
        if not rep.ok or face is None:
            self.msg.setText(rep.message + " Probá de nuevo.")
            return
        lf = next(l for l in a.faces if l.face is face)
        d = lf.decision.distance
        who = f"Coincide con {a.snapshot.name(d.person_id)} ({sim(d.similarity)})." if d.known else "Rostro no registrado: se busca igual."
        self.msg.setText(who + " Buscando…")
        emb, crop = face.embedding.copy(), face.aligned.copy()
        self.ctx.log("search", "Búsqueda por escaneo facial", who)
        name = a.snapshot.name(d.person_id) if d.known else None
        QTimer.singleShot(350, lambda: self.ctx.navigate("gallery_results", query_embedding=emb, query_crop=crop,
                                                         query_name=name))

    def on_enter(self, **_):
        self.refresh()
        self.msg.setText("Mirá a la cámara y presioná Escanear.")
        self.btn.set_loading(False, "Escanear")
        self.binding.start()

    def on_leave(self):
        self.binding.stop()
