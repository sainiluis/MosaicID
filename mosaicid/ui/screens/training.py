"""Entrenamiento del clasificador SVM: dataset real, etapas reales, metadata real."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...core.svm import can_train, effective_threshold
from ...services.workers import TrainingWorker, keep_until_finished
from ..theme import C
from ..widgets import (BackLink, Badge, Button, Divider, FaceThumb, Meter, Page, PageHeader, PropRow, Row, Spinner,
                       clear_layout, label, overline, pixmap)
from .common import relative
from .people import person_avatar_path

STAGES = [("prepare", "Preparando dataset", "Embeddings L2-normalizados y etiquetados por persona"),
          ("train", "Entrenando SVM", "SVC con probability=True (calibración de Platt)"),
          ("evaluate", "Evaluando", "Validación cruzada estratificada, si hay datos suficientes"),
          ("save", "Guardando modelo", "data/svm/svm_model.joblib + svm_meta.json")]


class StageRow(Row):
    def __init__(self, title: str, detail: str):
        super().__init__(44, clickable=False)
        self.ic = QLabel()
        self.ic.setFixedSize(16, 16)
        self.spin = Spinner(16, C["accent"])
        self.spin.hide()
        self.title = label(title, "body")
        self.h.addWidget(self.ic)
        self.h.addWidget(self.spin)
        self.h.addWidget(self.title)
        self.h.addWidget(label(detail, "small"))
        self.h.addStretch(1)
        self.set("pending")

    def set(self, st: str) -> None:
        self.spin.setVisible(st == "running")
        self.ic.setVisible(st != "running")
        if st == "done":
            self.ic.setPixmap(pixmap("circle-check", C["ok"], 16))
        elif st == "error":
            self.ic.setPixmap(pixmap("triangle-alert", C["err"], 16))
        else:
            self.ic.setPixmap(pixmap("circle", C["ink4"], 16, 1.5))
        self.title.setStyleSheet(f"color: {C['ink3'] if st == 'pending' else C['ink']};")


class TrainingScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.worker: TrainingWorker | None = None
        back = BackLink("Personas")
        back.clicked.connect(lambda: ctx.navigate("people"))
        self.btn_train = Button("Entrenar SVM", "primary", icon_name="brain-circuit")
        self.btn_train.clicked.connect(self._start)
        self.btn_try = Button("Probar reconocimiento", "secondary")
        self.btn_try.clicked.connect(lambda: ctx.navigate("recognition"))
        self.add(PageHeader("Entrenamiento del clasificador",
                            "El SVM aprende fronteras entre identidades a partir de los embeddings registrados.",
                            [self.btn_try, self.btn_train], back=back))
        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setColumnStretch(0, 1)
        grid.setColumnMinimumWidth(1, 300)
        left = QVBoxLayout()
        left.setSpacing(0)
        h = QHBoxLayout()
        h.addWidget(label("Dataset", "section"))
        h.addStretch(1)
        self.ds_meta = label("", "small")
        h.addWidget(self.ds_meta)
        left.addLayout(h)
        left.addSpacing(8)
        self.table = QVBoxLayout()
        self.table.setSpacing(0)
        left.addLayout(self.table)
        self.reason = label("", "small", "warn", wrap=True)
        left.addSpacing(8)
        left.addWidget(self.reason)
        left.addSpacing(28)
        h2 = QHBoxLayout()
        h2.addWidget(label("Proceso", "section"))
        h2.addStretch(1)
        self.proc_meta = label("", "small")
        h2.addWidget(self.proc_meta)
        left.addLayout(h2)
        left.addSpacing(10)
        self.progress = Meter(0, "accent")
        left.addWidget(self.progress)
        left.addSpacing(6)
        self.stage_rows = {}
        for key, t, d in STAGES:
            r = StageRow(t, d)
            self.stage_rows[key] = r
            left.addWidget(r)
        left.addStretch(1)
        grid.addLayout(left, 0, 0)

        side = QWidget()
        side.setFixedWidth(300)
        s = QVBoxLayout(side)
        s.setContentsMargins(0, 0, 0, 0)
        s.setSpacing(0)
        s.addWidget(overline("Parámetros"))
        s.addSpacing(6)
        self.p_kernel = PropRow("Kernel")
        self.p_c = PropRow("C")
        self.p_prob = PropRow("probability", "True")
        self.p_rej = PropRow("Umbral de rechazo")
        self.p_in = PropRow("Entrada", "Embedding 128D")
        for w in (self.p_kernel, self.p_c, self.p_prob, self.p_rej, self.p_in):
            s.addWidget(w)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(14)
        s.addWidget(overline("Modelo actual"))
        s.addSpacing(6)
        self.meta_box = QVBoxLayout()
        self.meta_box.setSpacing(0)
        s.addLayout(self.meta_box)
        s.addSpacing(14)
        s.addWidget(Divider())
        s.addSpacing(12)
        s.addWidget(label("El SVM es un clasificador closed-set: siempre elige una de las clases entrenadas. "
                          "«Desconocido» se informa cuando la probabilidad queda por debajo del umbral de rechazo, que se ajusta al azar (1/K): "
                          "con pocas personas el SVM distingue peor a un desconocido. "
                          "El motor por distancia no necesita este paso.", "small", wrap=True))
        s.addStretch(1)
        grid.addWidget(side, 0, 1, Qt.AlignTop)
        self.add(grid)
        self.stretch()
        ctx.peopleChanged.connect(self.refresh)

    def refresh(self) -> None:
        ctx = self.ctx
        snap = ctx.snapshot
        ds = snap.dataset
        people = ctx.db.list_people()
        counts = ds.counts() if ds is not None else {}
        self.ds_meta.setText(f"{len(counts)} clases · {len(ds.y) if ds is not None else 0} embeddings")
        clear_layout(self.table)
        self.table.addWidget(Divider())
        head = Row(34, clickable=False)
        for t, w in (("Persona", 0), ("Muestras", 90), ("Distribución", 220), ("Estado", 120)):
            l = label(t, "meta")
            if w:
                l.setFixedWidth(w)
            head.h.addWidget(l, 0 if w else 1)
        self.table.addWidget(head)
        mx = max(counts.values(), default=1)
        trained = ctx.svm.trained_person_ids
        for p in people:
            r = Row(48, clickable=False)
            av = FaceThumb(26)
            av.set_path(person_avatar_path(ctx, p.id))
            cell = QHBoxLayout()
            cell.setSpacing(10)
            cell.addWidget(av)
            cell.addWidget(label(p.name, "body"))
            cell.addStretch(1)
            r.h.addLayout(cell, 1)
            n = label(str(counts.get(p.id, 0)), "small2")
            n.setFixedWidth(90)
            r.h.addWidget(n)
            m = Meter(counts.get(p.id, 0) / mx, "accent")
            m.setFixedWidth(220)
            r.h.addWidget(m)
            inc = p.id in trained and p.id not in snap.svm_pending
            b = Badge("Incluido" if inc else "Pendiente", "ok" if inc else "warn")
            bw = QWidget()
            bw.setFixedWidth(120)
            QHBoxLayout(bw).addWidget(b)
            bw.layout().setContentsMargins(0, 0, 0, 0)
            bw.layout().addStretch(1)
            r.h.addWidget(bw)
            self.table.addWidget(r)
        ok, why = can_train(ds) if ds is not None else (False, "")
        self.reason.setText("" if ok else why)
        self.reason.setVisible(not ok)
        running = self.worker is not None and self.worker.isRunning()
        self.btn_train.setEnabled(ok and not running)
        st = ctx.svm_status()
        self.proc_meta.setText({"dirty": f"{len(snap.svm_pending)} identidad(es) por incluir",
                                "trained": "El modelo está al día", "untrained": "Sin modelo entrenado",
                                "empty": "Sin muestras"}.get(st, ""))
        s = ctx.settings
        self.p_kernel.set("Lineal" if s.svm_kernel == "linear" else "RBF")
        self.p_c.set(f"{s.svm_c:.1f}")
        k = ctx.svm.n_classes
        self.p_rej.set(f"{effective_threshold(s.svm_rejection_threshold, k):.2f} · {k} clases" if k
                       else f"margen {s.svm_rejection_threshold:.2f}")
        self._render_meta()

    def _render_meta(self) -> None:
        clear_layout(self.meta_box)
        m = self.ctx.svm.meta
        if not self.ctx.svm.trained:
            self.meta_box.addWidget(label("Todavía no hay un modelo entrenado.", "small"))
            return
        names = {p.id: p.name.split(" ")[0] for p in self.ctx.db.list_people()}
        rows = [("Entrenado", relative(m.get("trained_at"))), ("Clases", str(len(m.get("classes", [])))),
                ("Muestras", str(m.get("samples"))), ("Duración", f"{m.get('duration_s', 0):.2f} s"),
                ("Pipeline", m.get("pipeline_version", "").split("/")[-1])]
        ev = m.get("evaluation") or {}
        if ev.get("method"):
            rows.append(("Exactitud (CV)", f"{ev['accuracy_mean'] * 100:.1f}% ± {ev['accuracy_std'] * 100:.1f}"))
        for k, v in rows:
            self.meta_box.addWidget(PropRow(k, v))
        self.meta_box.addSpacing(6)
        cls = ", ".join(names.get(int(c), f"#{c} (eliminada)") for c in m.get("classes", []))
        self.meta_box.addWidget(label(f"Clases: {cls}", "meta", wrap=True))
        if ev.get("method"):
            self.meta_box.addWidget(label(f"{ev['method']}. {ev.get('note', '')}", "meta", wrap=True))
        elif ev.get("note"):
            self.meta_box.addWidget(label(ev["note"], "meta", wrap=True))

    def _set_stage(self, key: str) -> None:
        keys = [k for k, _, _ in STAGES]
        i = keys.index(key)
        for j, k in enumerate(keys):
            self.stage_rows[k].set("done" if j < i else "running" if j == i else "pending")
        self.progress.set((i + 0.5) / len(keys), "accent")

    def _start(self) -> None:
        ctx = self.ctx
        ds = ctx.snapshot.dataset
        for r in self.stage_rows.values():
            r.set("pending")
        self.btn_train.set_loading(True, "Entrenando…")
        self.worker = TrainingWorker(ctx.svm, ds, ctx.settings.svm_c, ctx.settings.svm_kernel)
        keep_until_finished(self.worker)
        self.worker.stage.connect(self._set_stage)
        self.worker.finished_ok.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _done(self, meta: dict) -> None:
        for r in self.stage_rows.values():
            r.set("done")
        self.progress.set(1.0, "ok")
        self.btn_train.set_loading(False, "Entrenar SVM")
        self.ctx.svm_trained()
        ev = meta.get("evaluation") or {}
        acc = f" · exactitud CV {ev['accuracy_mean'] * 100:.1f}%" if ev.get("method") else ""
        self.ctx.log("svm", "Clasificador SVM actualizado", f"{len(meta['classes'])} personas · {meta['samples']} embeddings{acc}")
        self.ctx.toast("Clasificador actualizado", f"{len(meta['classes'])} personas incluidas.")
        self.proc_meta.setText(f"Completado en {meta['duration_s']:.2f} s")
        self.refresh()
        self.proc_meta.setText(f"Completado en {meta['duration_s']:.2f} s")

    def _failed(self, msg: str) -> None:
        self.btn_train.set_loading(False, "Entrenar SVM")
        for r in self.stage_rows.values():
            if r.spin.isVisible():
                r.set("error")
        self.progress.set(0, "err")
        self.ctx.toast("No se pudo entrenar", msg, "warn")
        self.refresh()

    def on_enter(self, **_):
        for r in self.stage_rows.values():
            r.set("pending")
        self.progress.set(0, "accent")
        self.refresh()
