"""Personas (tabla/lista) y detalle de persona (muestras faciales reales)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QMessageBox, QVBoxLayout, QWidget

from ...app_context import AppContext
from ..theme import C
from ..widgets import (BackLink, Badge, Button, Divider, FaceThumb, Page, PageHeader, PropRow, Row, clear_layout,
                       icon_label, label, overline, status_line)
from .common import relative, short_date


def person_avatar_path(ctx: AppContext, pid: int) -> Path | None:
    for s in ctx.db.list_samples(pid):
        if s.image_path:
            p = ctx.paths.data / s.image_path
            if p.exists():
                return p
    return None


class PeopleScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.btn_train = Button("Entrenar clasificador", "secondary", icon_name="brain-circuit")
        self.btn_train.clicked.connect(lambda: ctx.navigate("training"))
        add = Button("Registrar persona", "primary", icon_name="plus")
        add.clicked.connect(lambda: ctx.navigate("enroll"))
        self.header = PageHeader("Personas", "", [self.btn_train, add])
        self.add(self.header)
        self.table = QVBoxLayout()
        self.table.setSpacing(0)
        self.add(self.table)
        self.add(16)
        note = label("Cada persona se registra una sola vez. Sus muestras generan embeddings que usan los dos motores: "
                     "como referencias para distancia y como datos de entrenamiento para el SVM.", "small", wrap=True)
        self.add(note)
        self.stretch()
        ctx.peopleChanged.connect(self.refresh)
        ctx.svmChanged.connect(self.refresh)

    def _head(self) -> QWidget:
        r = Row(36, clickable=False)
        r.h.setContentsMargins(8, 0, 8, 0)
        cols = [("Persona", 0), ("Muestras", 110), ("Distancia", 150), ("Clasificador", 150),
                ("Última actualización", 170), ("", 40)]
        for text, w in cols:
            l = label(text, "meta")
            if w:
                l.setFixedWidth(w)
            r.h.addWidget(l, 1 if not w else 0)
        return r

    def refresh(self) -> None:
        clear_layout(self.table)
        people = self.ctx.db.list_people()
        snap = self.ctx.snapshot
        status = self.ctx.svm_status()
        emb = sum(p.sample_count for p in people)
        self.header.subtitle.setText(f"{len(people)} identidades registradas · {emb} embeddings de referencia")
        self.header.subtitle.show()
        pending = len(snap.svm_pending) if status in ("dirty",) else 0
        self.btn_train.setText(f"Entrenar clasificador ({pending} pendiente{'s' if pending != 1 else ''})"
                               if pending else "Entrenar clasificador")
        if not people:
            from ..widgets import EmptyState
            b = Button("Registrar persona", "primary", icon_name="user-plus")
            b.clicked.connect(lambda: self.ctx.navigate("enroll"))
            self.table.addWidget(EmptyState("users", "Todavía no hay personas registradas.",
                                            "Registrá al primer integrante con la captura guiada.", b))
            return
        top = Divider()
        self.table.addWidget(top)
        self.table.addWidget(self._head())
        trained = self.ctx.svm.trained_person_ids
        for p in people:
            r = Row(56)
            r.h.setContentsMargins(8, 0, 8, 0)
            av = FaceThumb(32)
            av.set_path(person_avatar_path(self.ctx, p.id))
            name = QVBoxLayout()
            name.setSpacing(0)
            n = label(p.name, "body")
            n.setStyleSheet("font-weight: 500;")
            name.addWidget(n)
            name.addWidget(label(f"Registrado {short_date(p.created_at)}", "meta"))
            cell = QHBoxLayout()
            cell.setSpacing(12)
            cell.addWidget(av)
            cell.addLayout(name)
            cell.addStretch(1)
            r.h.addLayout(cell, 1)
            m = label(f"{p.sample_count}", "small2")
            m.setFixedWidth(110)
            r.h.addWidget(m)
            d = Badge("Disponible" if p.sample_count else "Sin muestras", "ok" if p.sample_count else "warn")
            dw = QWidget()
            dw.setFixedWidth(150)
            QHBoxLayout(dw).addWidget(d)
            dw.layout().setContentsMargins(0, 0, 0, 0)
            dw.layout().addStretch(1)
            r.h.addWidget(dw)
            included = p.id in trained and p.id not in snap.svm_pending
            c = Badge("Incluido" if included else "Pendiente", "ok" if included else "warn")
            cw = QWidget()
            cw.setFixedWidth(150)
            QHBoxLayout(cw).addWidget(c)
            cw.layout().setContentsMargins(0, 0, 0, 0)
            cw.layout().addStretch(1)
            r.h.addWidget(cw)
            u = label(relative(p.updated_at), "small")
            u.setFixedWidth(170)
            r.h.addWidget(u)
            ch = icon_label("chevron-right", C["ink4"], 15)
            chw = QWidget()
            chw.setFixedWidth(40)
            QHBoxLayout(chw).addWidget(ch, 0, Qt.AlignRight)
            chw.layout().setContentsMargins(0, 0, 0, 0)
            r.h.addWidget(chw)
            r.clicked.connect(lambda pid=p.id: self.ctx.navigate("person", person_id=pid))
            self.table.addWidget(r)

    def on_enter(self, **_):
        self.refresh()


class PersonDetailScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.pid: int | None = None
        back = BackLink("Personas")
        back.clicked.connect(lambda: ctx.navigate("people"))
        self.add(back)
        self.add(20)
        head = QHBoxLayout()
        head.setSpacing(20)
        self.avatar = FaceThumb(72)
        head.addWidget(self.avatar)
        col = QVBoxLayout()
        col.setSpacing(6)
        self.name = label("", "title")
        col.addWidget(self.name)
        self.status = QHBoxLayout()
        self.status.setSpacing(16)
        col.addLayout(self.status)
        head.addLayout(col, 1)
        self.btn_search = Button("Buscar en galería", "secondary", icon_name="images")
        self.btn_search.clicked.connect(lambda: ctx.navigate("gallery_results", person_id=self.pid))
        self.btn_train = Button("Entrenar SVM", "primary", icon_name="brain-circuit")
        self.btn_train.clicked.connect(lambda: ctx.navigate("training"))
        head.addWidget(self.btn_search)
        head.addWidget(self.btn_train)
        self.add(head)
        self.add(24)
        self.add(Divider())
        self.add(24)
        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 280)
        left = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(label("Muestras faciales", "section"))
        top.addStretch(1)
        self.samples_meta = label("", "small")
        top.addWidget(self.samples_meta)
        left.addLayout(top)
        left.addSpacing(12)
        self.samples = QGridLayout()
        self.samples.setHorizontalSpacing(12)
        self.samples.setVerticalSpacing(16)
        left.addLayout(self.samples)
        left.addStretch(1)
        grid.addLayout(left, 0, 0)

        side = QWidget()
        side.setFixedWidth(280)
        s = QVBoxLayout(side)
        s.setContentsMargins(0, 0, 0, 0)
        s.setSpacing(0)
        s.addWidget(overline("Motores"))
        s.addSpacing(6)
        self.p_dist = PropRow("Distancia", Badge("Disponible", "ok"))
        self.p_refs = PropRow("Referencias")
        self.p_svm = PropRow("Clasificador SVM", Badge("", "ok"))
        self.p_trained = PropRow("Último entrenamiento")
        for w in (self.p_dist, self.p_refs, self.p_svm, self.p_trained):
            s.addWidget(w)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(14)
        s.addWidget(overline("Registro"))
        s.addSpacing(6)
        self.p_created = PropRow("Creado")
        self.p_updated = PropRow("Actualizado")
        s.addWidget(self.p_created)
        s.addWidget(self.p_updated)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(10)
        redo = Button("Volver a capturar muestras", "ghost", "sm", "rotate-ccw")
        redo.clicked.connect(lambda: ctx.navigate("enroll", person_id=self.pid))
        s.addWidget(redo, 0, Qt.AlignLeft)
        delete = Button("Eliminar persona", "danger", "sm", "trash-2")
        delete.clicked.connect(self._delete)
        s.addWidget(delete, 0, Qt.AlignLeft)
        s.addStretch(1)
        grid.addWidget(side, 0, 1, Qt.AlignTop)
        self.add(grid)
        self.stretch()

    def _delete(self) -> None:
        p = self.ctx.db.get_person(self.pid)
        if not p:
            return
        r = QMessageBox.question(self, "Eliminar persona",
                                 f"¿Eliminar a {p.name} y sus {p.sample_count} muestras?\n\n"
                                 "Se borran sus embeddings de la base local. No se puede deshacer.",
                                 QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if r != QMessageBox.Yes:
            return
        import shutil
        self.ctx.db.delete_person(p.id)
        shutil.rmtree(self.ctx.paths.samples / str(p.id), ignore_errors=True)
        self.ctx.log("person", f"Se eliminó a {p.name}", "El clasificador queda pendiente de reentrenamiento")
        self.ctx.people_changed()
        self.ctx.toast("Persona eliminada", "Reentrená el SVM para quitarla del clasificador.", "info")
        self.ctx.navigate("people")

    def on_enter(self, person_id: int | None = None, **_):
        if person_id is not None:
            self.pid = person_id
        p = self.ctx.db.get_person(self.pid) if self.pid is not None else None
        if not p:
            self.ctx.navigate("people")
            return
        snap = self.ctx.snapshot
        included = p.id in self.ctx.svm.trained_person_ids and p.id not in snap.svm_pending
        self.name.setText(p.name)
        self.avatar.set_path(person_avatar_path(self.ctx, p.id))
        clear_layout(self.status)
        self.status.addWidget(status_line("Distancia disponible", "ok"))
        self.status.addWidget(status_line("Incluido en SVM" if included else "SVM pendiente de reentrenamiento",
                                          "ok" if included else "warn"))
        self.status.addWidget(label(f"Registrado el {short_date(p.created_at)}", "small"))
        self.status.addStretch(1)
        self.btn_train.setVisible(not included)
        self.p_refs.set(f"{p.sample_count} embeddings")
        self.p_svm.value.set("Incluido" if included else "Pendiente", "ok" if included else "warn")
        self.p_trained.set(relative(self.ctx.svm.meta.get("trained_at")) if self.ctx.svm.trained else "Nunca")
        self.p_created.set(short_date(p.created_at))
        self.p_updated.set(relative(p.updated_at))
        samples = self.ctx.db.list_samples(p.id)
        self.samples_meta.setText(f"{len(samples)} muestras · rostro completo alineado (112 × 112)")
        clear_layout(self.samples)
        for i, s in enumerate(samples):
            cell = QVBoxLayout()
            cell.setSpacing(6)
            th = FaceThumb(150, round_=False, radius=4)
            th.set_path(self.ctx.paths.data / s.image_path if s.image_path else None)
            cell.addWidget(th)
            cap = QHBoxLayout()
            cap.addWidget(label(f"{i + 1}. {s.kind or 'Muestra'}", "meta"))
            cap.addStretch(1)
            q = f"nitidez {s.sharpness:.0f}" if s.sharpness else ""
            cap.addWidget(label(q, "meta"))
            cell.addLayout(cap)
            self.samples.addLayout(cell, i // 4, i % 4)
