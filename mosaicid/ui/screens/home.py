"""Inicio — workspace operativo: reconocimiento en vivo (protagonista) + galería editorial + actividad."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...core.gallery import thumbnail_path
from ..camera_view import CameraMeta, CameraView
from ..theme import C, qc
from ..widgets import Button, Dot, LinkLabel, Page, PageHeader, Row, icon_label, label, overline, pixmap, status_line
from .common import LiveBinding, overlays_for, relative, camera_picker

ACT_ICON = {"svm": "brain-circuit", "index": "folder-sync", "person": "user-plus", "search": "search"}


class StatusStrip(QWidget):
    """Franja de estado liviana (no KPIs): motores · volumen del índice."""

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setFixedHeight(40)
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(0, 0, 0, 0)
        self.h.setSpacing(12)

    def refresh(self) -> None:
        from ..widgets import clear_layout
        clear_layout(self.h)
        ctx = self.ctx
        ok = ctx.models_ok
        items: list[QWidget] = [status_line("YuNet listo" if ok else "YuNet no disponible", "ok" if ok else "err"),
                                status_line("SFace listo" if ok else "SFace no disponible", "ok" if ok else "err")]
        st = ctx.svm_status()
        if st == "trained":
            items.append(status_line("SVM entrenado", "ok"))
        elif st == "dirty":
            n = len(ctx.snapshot.svm_pending)
            w = status_line(f"SVM pendiente · {n} {'identidad' if n == 1 else 'identidades'}", "warn")
            w.text.setProperty("tone", "warn")
            w.setCursor(Qt.PointingHandCursor)
            w.mouseReleaseEvent = lambda e: ctx.navigate("training")
            items.append(w)
        else:
            items.append(status_line("SVM sin entrenar", "idle"))
        people = ctx.db.list_people()
        photos, faces = ctx.db.gallery_counts()
        from ..widgets import fmt_num
        nums = [(len(people), "identidades"), (photos, "fotografías"), (faces, "rostros")]
        for i, w in enumerate(items):
            if i:
                self.h.addWidget(label("·", "small"))
            self.h.addWidget(w)
        sep = QWidget()
        sep.setFixedSize(1, 14)
        sep.setStyleSheet(f"background: {C['line_strong']};")
        self.h.addSpacing(4)
        self.h.addWidget(sep)
        self.h.addSpacing(4)
        for i, (n, t) in enumerate(nums):
            if i:
                self.h.addWidget(label("·", "small"))
            l = label(f"<b style='font-weight:500;color:{C['ink']}'>{fmt_num(n)}</b> {t}", "small2")
            l.setTextFormat(Qt.RichText)
            self.h.addWidget(l)
        self.h.addStretch(1)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(QColor(20, 32, 43, 23))
        p.drawLine(0, 0, self.width(), 0)
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


def system_status(ctx: AppContext) -> QWidget:
    if not ctx.models_ok:
        return status_line("Modelos no disponibles", "err")
    if ctx.svm_status() == "dirty":
        return status_line("Clasificador pendiente de reentrenamiento", "warn")
    return status_line("Todo listo", "ok")


class Collage(QWidget):
    """Una imagen grande + dos chicas, sin bordes."""

    clicked = None

    def __init__(self, on_click):
        super().__init__()
        self.images: list[QImage] = []
        self.on_click = on_click
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(220)

    def set_paths(self, paths: list[Path]) -> None:
        self.images = [QImage(str(p)) for p in paths[:3]]
        self.update()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return int(w * 0.82)

    def mouseReleaseEvent(self, e):
        self.on_click()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = QRectF(self.rect())
        clip = QPainterPath()
        clip.addRoundedRect(r, 8, 8)
        p.setClipPath(clip)
        gap = 4
        lw = (r.width() - gap) * 0.575
        cells = [QRectF(0, 0, lw, r.height()),
                 QRectF(lw + gap, 0, r.width() - lw - gap, (r.height() - gap) / 2),
                 QRectF(lw + gap, (r.height() + gap) / 2, r.width() - lw - gap, (r.height() - gap) / 2)]
        for i, c in enumerate(cells):
            p.fillRect(c, qc("sunken"))
            if i < len(self.images) and not self.images[i].isNull():
                img = self.images[i]
                s = max(c.width() / img.width(), c.height() / img.height())
                p.save()
                p.setClipRect(c, Qt.IntersectClip)
                p.drawImage(QRectF(c.center().x() - img.width() * s / 2, c.center().y() - img.height() * s / 2,
                                   img.width() * s, img.height() * s), img)
                p.restore()


class HomeScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.header = PageHeader("MosaicID", "Reconocimiento facial y búsqueda inteligente de fotografías.")
        self.status_holder = QHBoxLayout()
        self.header.aside.addLayout(self.status_holder)
        self.add(self.header)
        self.strip = StatusStrip(ctx)
        self.add(self.strip)
        self.add(28)

        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setColumnStretch(0, 7)
        grid.setColumnStretch(1, 3)

        left = QVBoxLayout()
        left.setSpacing(0)
        title_row = QHBoxLayout()
        title_row.addWidget(label("Reconocimiento en vivo", "section"))
        title_row.addStretch(1)
        self.engine_lbl = label("", "small")
        title_row.addWidget(self.engine_lbl)
        left.addLayout(title_row)
        left.addSpacing(12)
        self.cam = CameraView()
        self.cam.faceClicked.connect(lambda _: ctx.navigate("recognition"))
        left.addWidget(self.cam)
        start = Button("Iniciar reconocimiento", "primary", icon_right="arrow-right")
        start.clicked.connect(lambda: ctx.navigate("recognition"))
        self.meta = CameraMeta(start)
        self.meta.set_picker(camera_picker(ctx))
        left.addWidget(self.meta)
        left.addStretch(1)
        grid.addLayout(left, 0, 0)

        right = QVBoxLayout()
        right.setSpacing(0)
        o = overline("Galería inteligente")
        o.setFixedHeight(24)
        right.addWidget(o)
        right.addSpacing(12)
        self.collage = Collage(lambda: ctx.navigate("gallery_search"))
        right.addWidget(self.collage)
        right.addSpacing(20)
        h = label("Un rostro. Todas sus fotos.", "section")
        h.setStyleSheet("font-size: 20px;")
        right.addWidget(h)
        right.addSpacing(6)
        right.addWidget(label("Encontrá a una persona en tu biblioteca.", "small2"))
        right.addSpacing(20)
        b = Button("Buscar en galería", "secondary", icon_name="images")
        b.setStyleSheet("text-align: left; padding-left: 12px;")
        b.clicked.connect(lambda: ctx.navigate("gallery_search"))
        right.addWidget(b)
        right.addSpacing(6)
        reg = Row(40, separator=False, radius=6)
        reg.h.setContentsMargins(4, 0, 4, 0)
        reg.h.setSpacing(10)
        reg.h.addWidget(icon_label("user-plus", C["ink3"], 15))
        reg.h.addWidget(label("Registrar persona", "small2"))
        reg.h.addStretch(1)
        reg.h.addWidget(icon_label("chevron-right", C["ink4"], 15))
        reg.clicked.connect(lambda: ctx.navigate("enroll"))
        right.addWidget(reg)
        right.addStretch(1)
        grid.addLayout(right, 0, 1)
        self.add(grid)
        self.add(36)

        self.add(label("Actividad reciente", "section"))
        self.add(4)
        self.activity = QVBoxLayout()
        self.activity.setSpacing(0)
        self.add(self.activity)
        self.stretch()

        self.binding = LiveBinding(ctx, "home", self.cam, self.meta, self._on_analysis)
        for sig in (ctx.peopleChanged, ctx.svmChanged, ctx.galleryChanged, ctx.activityChanged, ctx.settingsChanged):
            sig.connect(self.refresh)

    def _engine(self) -> str:
        return "svm" if self.ctx.svm_status() == "trained" else "distance"

    def _on_analysis(self, a) -> None:
        self.cam.set_overlays(overlays_for(a, self._engine()))

    def refresh(self) -> None:
        from ..widgets import clear_layout
        clear_layout(self.status_holder)
        self.status_holder.addWidget(system_status(self.ctx))
        self.strip.refresh()
        self.engine_lbl.setText("Clasificador SVM" if self._engine() == "svm" else "Motor por distancia")
        # collage: fotos reales indexadas si existen; si no, imágenes locales de muestra
        photos = [ph for ph in self.ctx.db.list_photos(only_real=True) if ph.face_count][:3]
        paths = [thumbnail_path(self.ctx.paths.thumbnails, Path(ph.path), ph.mtime, ph.size) for ph in photos]
        if len(paths) < 3:
            ed = self.ctx.paths.assets / "editorial"
            paths = [ed / "p09.jpg", ed / "p07.jpg", ed / "p21.jpg"]
        self.collage.set_paths(paths)
        clear_layout(self.activity)
        rows = self.ctx.db.recent_activity(4)
        if not rows:
            self.activity.addWidget(label("Todavía no hay actividad. Registrá una persona para comenzar.", "muted"))
        for r in rows:
            row = Row(44, clickable=False)
            row.h.addWidget(icon_label(ACT_ICON.get(r["kind"], "info"), C["ink3"], 15))
            row.h.addWidget(label(r["title"], "body"))
            if r["detail"]:
                row.h.addWidget(label(r["detail"], "small"))
            row.h.addStretch(1)
            row.h.addWidget(label(relative(r["at"]), "small"))
            self.activity.addWidget(row)

    def on_enter(self, **_):
        self.refresh()
        self.binding.start()

    def on_leave(self):
        self.binding.stop()
