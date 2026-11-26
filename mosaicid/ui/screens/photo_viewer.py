"""Visor de fotografía: imagen original (escalada para pantalla) + caras + inspector."""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QThreadPool
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ...app_context import AppContext
from ...config import SOURCE_TYPES
from ...services.workers import ThumbLoader, thumb_emitter
from ..theme import C, font, qc
from ..widgets import Button, Divider, FaceThumb, PropRow, Row, clear_layout, label, overline
from .common import long_date, sim
from .gallery import SOURCE_LABELS

TAKEN_SRC = {"exif": "EXIF DateTimeOriginal", "filesystem": "fecha del archivo"}


class Stage(QWidget):
    def __init__(self):
        super().__init__()
        self.img: QImage | None = None
        self.faces: list[dict] = []
        self.match_id: int | None = None
        self.size_orig = (1, 1)

    def set_photo(self, img: QImage | None, faces: list[dict], match_id: int | None, size_orig) -> None:
        self.img, self.faces, self.match_id, self.size_orig = img, faces, match_id, size_orig
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), qc("stage"))
        if self.img is None or self.img.isNull():
            p.setPen(QColor(255, 255, 255, 120))
            p.setFont(font(13))
            p.drawText(self.rect(), Qt.AlignCenter, "Cargando…" if self.img is None else "No se pudo abrir la imagen")
            return
        r = QRectF(self.rect()).adjusted(40, 8, -40, -40)
        s = min(r.width() / self.img.width(), r.height() / self.img.height())
        tw, th = self.img.width() * s, self.img.height() * s
        x0, y0 = r.x() + (r.width() - tw) / 2, r.y() + (r.height() - th) / 2
        p.drawImage(QRectF(x0, y0, tw, th), self.img)
        k = tw / self.size_orig[0]
        for f in self.faces:
            br = QRectF(x0 + f["x"] * k, y0 + f["y"] * k, f["w"] * k, f["h"] * k)
            hit = f["id"] == self.match_id
            p.setBrush(Qt.NoBrush)
            if hit:
                p.setPen(QPen(QColor(0, 0, 0, 80), 3))
                p.drawRoundedRect(br, 3, 3)
                p.setPen(QPen(QColor(C["box_known"]), 1.6))
            else:
                p.setPen(QPen(QColor(255, 255, 255, 90), 1, Qt.DashLine))
            p.drawRoundedRect(br, 3, 3)


class PhotoViewerScreen(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {C['canvas']};")
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        left = QWidget()
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(f"background: {C['stage']};")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 8)
        self.back = Button("Volver", "dark", "sm", "arrow-left")
        self.back.clicked.connect(self._back)
        bar.addWidget(self.back)
        bar.addStretch(1)
        self.counter = label("", "small")
        self.counter.setStyleSheet("color: rgba(255,255,255,0.5); background: transparent;")
        bar.addWidget(self.counter)
        self.prev = Button("", "dark", "sm", "chevron-left")
        self.next = Button("", "dark", "sm", "chevron-right")
        self.prev.clicked.connect(lambda: self._step(-1))
        self.next.clicked.connect(lambda: self._step(1))
        bar.addWidget(self.prev)
        bar.addWidget(self.next)
        lv.addLayout(bar)
        self.stage = Stage()
        lv.addWidget(self.stage, 1)
        h.addWidget(left, 1)

        side = QScrollArea()
        side.setWidgetResizable(True)
        side.setFixedWidth(310)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setMaximumWidth(300)
        body.setStyleSheet(f"background: {C['canvas']};")
        self.ins = QVBoxLayout(body)
        self.ins.setContentsMargins(20, 20, 20, 20)
        self.ins.setSpacing(0)
        side.setWidget(body)
        h.addWidget(side)
        for key, d in ((Qt.Key_Right, 1), (Qt.Key_Left, -1)):
            QShortcut(QKeySequence(key), self, activated=lambda d=d: self._step(d))
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._back)
        thumb_emitter().loaded.connect(self._loaded)
        self.seq: list[int] = []
        self.matches: dict = {}
        self.photo_id: int | None = None
        self.back_route = "gallery"
        self.who = ""

    def on_enter(self, photo_id: int, sequence: list[int] | None = None, matches: dict | None = None,
                 back: str = "gallery", who: str = "", **_):
        self.seq = list(sequence or [photo_id])
        self.matches = matches or {}
        self.back_route = back
        self.who = who
        self.back.setText("Resultados" if back == "gallery_results" else "Galería")
        self._show(photo_id)

    def _back(self) -> None:
        if self.isVisible():
            self.ctx.navigate(self.back_route)

    def _step(self, d: int) -> None:
        if not self.isVisible() or self.photo_id not in self.seq:
            return
        i = self.seq.index(self.photo_id) + d
        if 0 <= i < len(self.seq):
            self._show(self.seq[i])

    def _loaded(self, key: str, img: QImage) -> None:
        if key == f"full:{self.photo_id}":
            ph = self.ctx.db.get_photo(self.photo_id)
            self.stage.set_photo(img, self.faces, self.match_face, (ph.width or img.width(), ph.height or img.height()))

    def _show(self, photo_id: int) -> None:
        ph = self.ctx.db.get_photo(photo_id)
        if ph is None:
            self._back()
            return
        self.photo_id = photo_id
        i = self.seq.index(photo_id) if photo_id in self.seq else 0
        self.counter.setText(f"{i + 1} / {len(self.seq)}")
        self.prev.setEnabled(i > 0)
        self.next.setEnabled(i < len(self.seq) - 1)
        self.faces = self.ctx.db.photo_faces(photo_id)
        m = self.matches.get(photo_id)
        self.match_face = m[0] if m else None
        self.stage.set_photo(None, [], None, (1, 1))
        # la original se carga en segundo plano, escalada a tamaño de pantalla (respetando orientación EXIF)
        QThreadPool.globalInstance().start(_FullLoader(f"full:{photo_id}", Path(ph.path), 2400))
        self._inspector(ph, m)

    def _inspector(self, ph, m) -> None:
        L = self.ins
        clear_layout(L)
        L.addWidget(overline("Fotografía"))
        L.addSpacing(4)
        t = label(ph.filename.replace("_", "_​"), "section", wrap=True)
        L.addWidget(t)
        L.addSpacing(2)
        L.addWidget(label(long_date(ph.taken_at), "small"))
        L.addWidget(label(f"Fecha según {TAKEN_SRC.get(ph.taken_at_source, 'archivo')}", "meta"))
        L.addSpacing(16)
        L.addWidget(Divider())
        L.addSpacing(10)
        if m:
            L.addWidget(PropRow(f"Similitud con {self.who}", sim(m[1])))
        L.addWidget(PropRow("Rostros detectados", str(ph.face_count)))
        L.addWidget(PropRow("Tamaño", f"{ph.width} × {ph.height}" if ph.width else "—"))
        combo = QComboBox()
        for k, t_ in SOURCE_LABELS:
            combo.addItem(t_, k)
        combo.setCurrentIndex(SOURCE_TYPES.index(ph.source_type) if ph.source_type in SOURCE_TYPES else 4)
        combo.setFixedWidth(140)
        combo.currentIndexChanged.connect(lambda _: self._set_source(ph.id, combo.currentData()))
        L.addWidget(PropRow("Tipo de fuente", combo))
        L.addSpacing(10)
        L.addWidget(Divider())
        L.addSpacing(14)
        L.addWidget(overline("Personas en la foto"))
        L.addSpacing(6)
        snap = self.ctx.snapshot
        unknown = 0
        for f in self.faces:
            d = snap.matcher.identify(f["embedding"]) if not snap.matcher.empty else None
            if d is None or not d.known:
                unknown += 1
                continue
            r = Row(36, clickable=True, separator=False, radius=6)
            r.h.setContentsMargins(4, 0, 4, 0)
            r.h.addWidget(label(snap.name(d.person_id), "body"), 1)
            r.h.addWidget(label(sim(d.similarity), "small"))
            r.clicked.connect(lambda pid=d.person_id: self.ctx.navigate("person", person_id=pid))
            L.addWidget(r)
        if unknown:
            L.addWidget(label(f"{unknown} sin identificar", "muted"))
        L.addWidget(label("Identificación por distancia con las referencias actuales.", "meta", wrap=True))
        L.addSpacing(14)
        L.addWidget(Divider())
        L.addSpacing(14)
        L.addWidget(overline("Ubicación"))
        L.addSpacing(6)
        L.addWidget(label(ph.path.replace("\\", "\\​").replace("/", "/​"), "mono", wrap=True))
        L.addSpacing(10)
        b = Button("Abrir ubicación", "secondary", "sm", "folder-open")
        b.clicked.connect(lambda: self._reveal(ph.path))
        L.addWidget(b, 0, Qt.AlignLeft)
        L.addStretch(1)

    def _set_source(self, pid: int, st: str) -> None:
        self.ctx.db.set_photo_source(pid, st)
        self.ctx.galleryChanged.emit()
        self.ctx.toast("Clasificación actualizada", "Se conserva aunque se reindexe la carpeta.", "info")

    def _reveal(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self.ctx.toast("El archivo ya no existe", path, "warn")
            return
        subprocess.Popen(["explorer", "/select,", str(p)])


class _FullLoader(ThumbLoader):
    """Carga la imagen original con orientación EXIF (Pillow), reducida para pantalla."""

    def run(self) -> None:
        try:
            from PIL import Image, ImageOps
            with Image.open(self.path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((self.max_side, self.max_side))
                data = im.tobytes()
                img = QImage(data, im.width, im.height, im.width * 3, QImage.Format_RGB888).copy()
        except Exception:
            img = QImage()
        ThumbLoader.emitter.loaded.emit(self.key, img)
