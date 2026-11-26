"""Galería (biblioteca indexada) y resultados de búsqueda por rostro."""
from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from time import perf_counter
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QInputDialog, QScrollArea, QVBoxLayout, QWidget)

from ...app_context import AppContext
from ...core.gallery import search_faces, thumbnail_path
from ...db import PhotoRow
from ...services.workers import IndexWorker, keep_until_finished
from ..photo_grid import GridItem, GridSection, PhotoGrid
from ..theme import C
from ..widgets import (Button, Divider, EmptyState, FaceThumb, Meter, Segmented, clear_layout, fade_in, fmt_num,
                       icon_label, label, wrap)
from .common import relative, short_date, sim
from .people import person_avatar_path

log = logging.getLogger(__name__)

SOURCE_LABELS = [("real", "Fotos reales"), ("synthetic", "Simuladas"), ("generated", "Generadas"),
                 ("test", "Prueba"), ("unknown", "Sin clasificar")]


def year_of(p: PhotoRow) -> int | None:
    try:
        return datetime.fromisoformat(p.taken_at).year if p.taken_at else None
    except ValueError:
        return None


def to_item(ctx: AppContext, p: PhotoRow, metric: str = "", payload=None, boxes=None) -> GridItem:
    aspect = (p.width / p.height) if p.width and p.height else 1.5
    return GridItem(str(p.id), thumbnail_path(ctx.paths.thumbnails, Path(p.path), p.mtime, p.size), aspect,
                    short_date(p.taken_at), metric, payload if payload is not None else p.id, boxes or [])


class Toolbar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(56)
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(24, 0, 24, 0)
        self.h.setSpacing(12)

    def paintEvent(self, _):
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        from ..theme import LINE
        p.setPen(LINE)
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)


class GridScroll(QScrollArea):
    def __init__(self, grid: PhotoGrid, extra_top: QWidget | None = None):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("PageBody")
        self.v = QVBoxLayout(body)
        self.v.setContentsMargins(24, 4, 24, 32)
        self.v.setSpacing(0)
        if extra_top:
            self.v.addWidget(extra_top)
        self.v.addWidget(grid)
        self.v.addStretch(1)
        self.setWidget(body)


def start_indexing(ctx: AppContext, screen, folder_ids: list[int]) -> IndexWorker | None:
    if getattr(ctx, "index_worker", None) is not None and ctx.index_worker.isRunning():
        ctx.toast("Ya hay una indexación en curso", "", "info")
        return None
    w = IndexWorker(ctx.db, ctx.paths, deepcopy(ctx.settings), folder_ids)
    ctx.index_worker = w
    keep_until_finished(w)  # la referencia de ctx se suelta en done(), antes de que el hilo termine

    def done(rep):
        ctx.index_worker = None
        ctx.galleryChanged.emit()
        if rep.cancelled:
            ctx.toast("Indexación cancelada", f"{rep.indexed} fotografías procesadas.", "info")
        else:
            detail = (f"{rep.indexed} nuevas o modificadas · {rep.unchanged} sin cambios · {rep.faces} rostros"
                      + (f" · {len(rep.errors)} con error" if rep.errors else ""))
            ctx.toast("Indexación completa", detail)
            ctx.log("index", f"{rep.indexed} fotografías indexadas", detail)

    def failed(msg):
        ctx.index_worker = None
        ctx.toast("Falló la indexación", msg, "warn")
        ctx.galleryChanged.emit()

    w.finished_ok.connect(done)
    w.failed.connect(failed)
    w.start()
    return w


INCLUDED_KEY = "included_gallery_folder_id"


def included_gallery_folder(ctx: AppContext) -> int | None:
    """Registra la carpeta `galeria_incluida/` (fotos que vienen con la app) como fuente «real».

    Solo la agrega una vez: si el usuario la quita de la galería no vuelve a aparecer. Si la aplicación se
    movió de lugar, actualiza la ruta guardada. Devuelve el id de carpeta si hay que (re)indexarla.
    """
    d = ctx.paths.included_gallery
    has_images = d.is_dir() and any(f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp") for f in d.iterdir())
    if not has_images:
        return None
    db = ctx.db
    saved = db.get_kv(INCLUDED_KEY)
    if saved is None:
        fid = db.add_folder(str(d), "real")
        db.set_kv(INCLUDED_KEY, str(fid))
        log.info("Galería incluida registrada: %s", d)
        return fid
    row = next((r for r in db.list_folders() if str(r["id"]) == saved), None)
    if row is None:
        return None  # el usuario la quitó
    if Path(row["path"]) != d:
        db.update_folder_path(row["id"], str(d))
        log.info("Galería incluida movida a %s: se reindexa", d)
        return row["id"]
    return None if db.photo_state(row["id"]) else row["id"]  # indexada a medias o vacía → indexar


def index_included_gallery(ctx: AppContext) -> None:
    fid = included_gallery_folder(ctx)
    if fid is not None and ctx.models_ok:
        start_indexing(ctx, None, [fid])


def choose_source_type(parent) -> str | None:
    labels = [t for _, t in SOURCE_LABELS]
    item, ok = QInputDialog.getItem(
        parent, "Tipo de fotografías",
        "¿Qué tipo de fotografías contiene esta carpeta?\n"
        "La clasificación la decidís vos (MosaicID no la infiere).\n"
        "Subcarpetas llamadas real / synthetic / generated / test se clasifican con ese nombre.",
        labels, 0, False)
    if not ok:
        return None
    return dict((t, k) for k, t in SOURCE_LABELS)[item]


class GalleryScreen(QWidget):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"#Page, #PageBody {{ background: {C['canvas']}; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        tb = Toolbar()
        tb.h.addWidget(icon_label("folder-open", C["ink3"], 17))
        col = QVBoxLayout()
        col.setSpacing(0)
        self.title = label("", "body")
        self.title.setStyleSheet("font-weight: 600;")
        self.meta = label("", "meta")
        col.addWidget(self.title)
        col.addWidget(self.meta)
        tb.h.addLayout(col)
        tb.h.addStretch(1)
        self.source = Segmented([("all", "Todas"), ("real", "Solo reales")], "all", small=True)
        self.source.changed.connect(lambda _: self.refresh())
        tb.h.addWidget(self.source)
        tb.h.addSpacing(8)
        self.btn_refresh = Button("Actualizar índice", "ghost", "sm", "refresh-cw")
        self.btn_refresh.clicked.connect(self._reindex)
        self.btn_add = Button("Agregar carpeta", "ghost", "sm", "folder-search")
        self.btn_add.clicked.connect(self._add_folder)
        self.btn_search = Button("Buscar persona", "primary", "sm", "search")
        self.btn_search.clicked.connect(lambda: ctx.navigate("gallery_search"))
        for b in (self.btn_refresh, self.btn_add, self.btn_search):
            tb.h.addWidget(b)
        v.addWidget(tb)

        # progreso de indexación (no bloquea la interfaz)
        self.prog = QWidget()
        ph = QHBoxLayout(self.prog)
        ph.setContentsMargins(24, 10, 24, 10)
        ph.setSpacing(12)
        self.prog_lbl = label("", "small2")
        self.prog_lbl.setMinimumWidth(360)
        self.prog_bar = Meter(0, "accent")
        self.prog_cancel = Button("Cancelar", "ghost", "sm", "x")
        self.prog_cancel.clicked.connect(self._cancel)
        ph.addWidget(self.prog_lbl)
        ph.addWidget(self.prog_bar, 1)
        ph.addWidget(self.prog_cancel)
        self.prog.hide()
        v.addWidget(self.prog)

        self.grid = PhotoGrid(190)
        self.grid.itemClicked.connect(self._open)
        self.scroll = GridScroll(self.grid)
        v.addWidget(self.scroll, 1)
        sel = Button("Seleccionar carpeta", "primary")
        sel.clicked.connect(self._add_folder)
        self.empty = EmptyState("folder-open", "No hay fotografías indexadas.", "Elegí una carpeta para comenzar.", sel)
        v.addWidget(self.empty)
        self.photos: list[PhotoRow] = []
        ctx.galleryChanged.connect(self.refresh)

    # ── acciones ─────────────────────────────────────────────────────────
    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Elegí la carpeta de fotografías")
        if not d:
            return
        st = choose_source_type(self)
        if st is None:
            return
        fid = self.ctx.db.add_folder(str(Path(d)), st)
        log.info("Carpeta agregada a la galería: %s (%s)", d, st)
        self._run([fid])

    def _reindex(self) -> None:
        ids = [r["id"] for r in self.ctx.db.list_folders()]
        if ids:
            self._run(ids)

    def _run(self, ids: list[int]) -> None:
        w = start_indexing(self.ctx, self, ids)
        if w is None:
            return
        self._attach(w)

    def _attach(self, w: IndexWorker) -> None:
        self.prog.show()
        self.prog_bar.set(0)
        self.prog_lbl.setText("Preparando…")
        self.btn_refresh.set_loading(True, "Indexando…")
        self.btn_add.setEnabled(False)

        def prog(i, n, name):
            self.prog_lbl.setText(f"Indexando {i} de {n}" + (f" · {name}" if name else "") if n else "Sin cambios para indexar")
            self.prog_bar.set(i / n if n else 1.0)
        w.progress.connect(prog)
        w.finished.connect(self._detach)

    def _detach(self) -> None:
        self.prog.hide()
        self.btn_refresh.set_loading(False, "Actualizar índice")
        self.btn_add.setEnabled(True)

    def _cancel(self) -> None:
        w = getattr(self.ctx, "index_worker", None)
        if w is not None:
            w.cancel.set()

    def _open(self, photo_id) -> None:
        ids = [p.id for p in self.photos]
        self.ctx.navigate("photo", photo_id=photo_id, sequence=ids, back="gallery")

    # ── datos ────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        db = self.ctx.db
        folders = db.list_folders()
        total, faces = db.gallery_counts()
        has = bool(folders)
        if not has:
            self.title.setText("Sin carpeta")
        elif len(folders) == 1:
            self.title.setText(Path(folders[0]["path"]).name or folders[0]["path"])
        else:
            self.title.setText(f"{len(folders)} carpetas")
        self.meta.setText(f"{fmt_num(total)} fotografías · {fmt_num(faces)} caras · Indexado "
                          f"{relative(db.last_indexed()).lower()}" if has else "No hay fotografías indexadas")
        only_real = self.source.value == "real"
        self.photos = db.list_photos(only_real=only_real)
        self.btn_refresh.setEnabled(has and not (getattr(self.ctx, "index_worker", None)))
        self.btn_search.setEnabled(total > 0)
        indexing = getattr(self.ctx, "index_worker", None) is not None
        self.empty.setVisible(not has and not indexing)
        self.scroll.setVisible(has or indexing)
        by_year: dict[object, list[PhotoRow]] = defaultdict(list)
        for p in self.photos:
            by_year[year_of(p) or "Sin fecha"].append(p)
        keys = sorted(by_year, key=lambda k: (isinstance(k, str), -k if isinstance(k, int) else 0))
        secs = [GridSection(str(k), f"{len(by_year[k])} fotografías", [to_item(self.ctx, p) for p in by_year[k]])
                for k in keys]
        if has and not self.photos:
            secs = [GridSection("", "", [])]
            self.meta.setText(self.meta.text() + (" · ninguna marcada como real" if only_real else ""))
        self.grid.set_sections(secs)

    def on_enter(self, **_):
        self.refresh()
        w = getattr(self.ctx, "index_worker", None)
        if w is not None and w.isRunning():
            self._attach(w)


class GalleryResultsScreen(QWidget):
    """Resultados de búsqueda: persona registrada (todas sus referencias) o rostro escaneado."""

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Page")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"#Page, #PageBody {{ background: {C['canvas']}; }}")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        tb = Toolbar()
        back = Button("", "ghost", "sm", "arrow-left")
        back.setFixedWidth(30)
        back.clicked.connect(lambda: ctx.navigate("gallery_search"))
        tb.h.addWidget(back)
        self.avatar = FaceThumb(30)
        tb.h.addWidget(self.avatar)
        col = QVBoxLayout()
        col.setSpacing(0)
        self.title = label("", "body")
        self.title.setStyleSheet("font-weight: 600;")
        self.meta = label("", "meta")
        col.addWidget(self.title)
        col.addWidget(self.meta)
        tb.h.addLayout(col)
        tb.h.addStretch(1)
        self.source = Segmented([("all", "Todas"), ("real", "Solo reales")], "all", small=True)
        self.source.changed.connect(lambda _: self._search())
        tb.h.addWidget(label("Fuente", "meta"))
        tb.h.addWidget(self.source)
        tb.h.addSpacing(8)
        tb.h.addWidget(label("Ordenar", "meta"))
        self.sort = Segmented([("date", "Fecha"), ("sim", "Similitud")], "date", small=True)
        self.sort.changed.connect(lambda _: self._render())
        tb.h.addWidget(self.sort)
        v.addWidget(tb)
        # resultado: franja verde con el éxito de la búsqueda
        self.banner = QWidget()
        self.banner.setObjectName("MatchBanner")
        self.banner.setAttribute(Qt.WA_StyledBackground, True)
        self.banner.setStyleSheet(f"#MatchBanner {{ background: {C['ok_soft']}; border-bottom: 1px solid {C['line']}; }}")
        bh = QHBoxLayout(self.banner)
        bh.setContentsMargins(24, 12, 24, 12)
        bh.setSpacing(12)
        bh.addWidget(icon_label("circle-check-big", C["ok"], 26))
        bcol = QVBoxLayout()
        bcol.setSpacing(1)
        self.banner_title = label("", "body")
        self.banner_title.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {C['ok']};")
        self.banner_detail = label("", "small")
        bcol.addWidget(self.banner_title)
        bcol.addWidget(self.banner_detail)
        bh.addLayout(bcol, 1)
        self.banner.hide()
        v.addWidget(self.banner)
        # búsqueda en curso
        self.searching = QWidget()
        sv = QVBoxLayout(self.searching)
        sv.setContentsMargins(0, 110, 0, 0)
        sv.setSpacing(0)
        sv.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.search_face = FaceThumb(84)
        sv.addWidget(self.search_face, 0, Qt.AlignHCenter)
        sv.addSpacing(18)
        self.search_title = label("", "body")
        self.search_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        sv.addWidget(self.search_title, 0, Qt.AlignHCenter)
        sv.addSpacing(14)
        self.search_bar = Meter(0, "accent")
        self.search_bar.setFixedWidth(320)
        sv.addWidget(self.search_bar, 0, Qt.AlignHCenter)
        sv.addSpacing(10)
        self.search_detail = label("", "small")
        sv.addWidget(self.search_detail, 0, Qt.AlignHCenter)
        self.searching.hide()
        v.addWidget(self.searching, 1)
        self._anim = QTimer(self, interval=16, timeout=self._tick)
        self._anim_t0 = 0.0
        self.boxes: dict[int, list] = {}
        self.mode = "person"
        self.years = QHBoxLayout()
        self.years.setContentsMargins(20, 0, 24, 0)
        self.years.setSpacing(2)
        self.yw = wrap(self.years)
        self.yw.setFixedHeight(44)
        v.addWidget(self.yw)
        self.ydiv = Divider()
        v.addWidget(self.ydiv)
        self.grid = PhotoGrid(180)
        self.grid.itemClicked.connect(self._open)
        self.scroll = GridScroll(self.grid)
        v.addWidget(self.scroll, 1)
        self.empty = EmptyState("search", "Sin coincidencias.", "")
        v.addWidget(self.empty)
        self.year: int | None = None
        self.hits = []
        self.photos: dict[int, PhotoRow] = {}
        self.queries: np.ndarray | None = None
        self.who = ""
        ctx.galleryChanged.connect(lambda: self.isVisible() and self._search())

    def on_enter(self, person_id: int | None = None, query_embedding=None, query_crop=None, query_name=None,
                 **_):
        ctx = self.ctx
        if query_embedding is None and person_id is None:
            self._search()  # volver desde el visor: sin animación
            return
        self.year = None
        if query_embedding is not None:
            self.queries = np.atleast_2d(np.asarray(query_embedding, np.float32))
            self.who = query_name or "este rostro"
            self.avatar.set_array(query_crop)
            self.search_face.set_array(query_crop)
            self.mode = "scan"
        elif person_id is not None:
            p = ctx.db.get_person(person_id)
            if not p:
                ctx.navigate("gallery_search")
                return
            self.queries = np.stack([s.embedding for s in ctx.db.list_samples(p.id)]) if p.sample_count else None
            self.who = p.name.split(" ")[0]
            self.avatar.set_path(person_avatar_path(ctx, p.id))
            self.search_face.set_path(person_avatar_path(ctx, p.id))
            self.mode = "person"
        self._start_animation()

    # ── animación «buscando»: la comparación real tarda milisegundos; esto da un feedback legible ──
    SEARCH_MS = 1100

    def _start_animation(self) -> None:
        total, faces = self.ctx.db.gallery_counts()
        self.search_title.setText(f"Buscando a {self.who}…")
        self.search_detail.setText(f"Comparando con {fmt_num(faces)} rostros en {fmt_num(total)} fotografías")
        self.search_bar.set(0)
        for w in (self.banner, self.yw, self.ydiv, self.scroll, self.empty):
            w.hide()
        self.searching.show()
        fade_in(self.searching, 150)
        self._anim_t0 = perf_counter()
        self._anim.start()

    def _tick(self) -> None:
        t = min(1.0, (perf_counter() - self._anim_t0) * 1000 / self.SEARCH_MS)
        self.search_bar.set(1 - (1 - t) ** 2)
        if t >= 1.0:
            self._anim.stop()
            self.searching.hide()
            self._search()
            for w in (self.scroll, self.empty, self.banner):
                if w.isVisible():
                    fade_in(w, 260)

    def on_leave(self):
        self._anim.stop()

    def _search(self) -> None:
        ctx = self.ctx
        only_real = self.source.value == "real"
        F, fids, pids = ctx.db.face_matrix(only_real=only_real)
        thr = ctx.settings.gallery_match_threshold
        self.hits = [] if self.queries is None else search_faces(self.queries, F, fids, pids, thr)
        self.photos = {h.photo_id: ctx.db.get_photo(h.photo_id) for h in self.hits}
        n = len(self.hits)
        self.boxes = self._match_boxes()
        self.title.setText("Búsqueda por rostro escaneado" if self.mode == "scan" else f"Fotos de {self.who}")
        best = max((h.similarity for h in self.hits), default=0.0)
        self.banner_title.setText(f"Encontramos a {self.who} en {n} {'fotografía' if n == 1 else 'fotografías'}")
        self.banner_detail.setText(f"Su rostro está marcado en verde en cada foto · mejor coincidencia {sim(best)}")
        self.banner.setVisible(n > 0)
        self.yw.show()
        self.ydiv.show()
        q = len(self.queries) if self.queries is not None else 0
        self.meta.setText(f"Similitud coseno ≥ {thr:.3f} · {q} {'referencias' if q != 1 else 'consulta'} · "
                          f"{'solo fotos reales' if only_real else 'todas las fuentes'} · {len(F)} rostros comparados")
        self._render()

    def _render(self) -> None:
        clear_layout(self.years)
        by_year: dict[object, list] = defaultdict(list)
        for h in self.hits:
            by_year[year_of(self.photos[h.photo_id]) or "Sin fecha"].append(h)
        keys = sorted(by_year, key=lambda k: (isinstance(k, str), -k if isinstance(k, int) else 0))
        self._tab("Todos", len(self.hits), self.year is None, None)
        for k in keys:
            self._tab(str(k), len(by_year[k]), self.year == k, k)
        self.years.addStretch(1)
        hits = self.hits if self.year is None else by_year.get(self.year, [])
        if self.sort.value == "sim":
            hits = sorted(hits, key=lambda h: -h.similarity)
            secs = [GridSection("", "", [self._item(h) for h in hits])]
        else:
            hits = sorted(hits, key=lambda h: self.photos[h.photo_id].taken_at or "", reverse=True)
            groups: dict[object, list] = defaultdict(list)
            for h in hits:
                groups[year_of(self.photos[h.photo_id]) or "Sin fecha"].append(h)
            secs = [GridSection(str(k), f"{len(groups[k])} fotografías", [self._item(h) for h in groups[k]])
                    for k in sorted(groups, key=lambda k: (isinstance(k, str), -k if isinstance(k, int) else 0))]
        self.sequence = [h for s in secs for h in [it.payload for it in s.items]]
        self.grid.set_sections(secs)
        self.empty.setVisible(not self.hits)
        self.scroll.setVisible(bool(self.hits))
        total, _ = self.ctx.db.gallery_counts()
        self.empty.title.setText(f"No encontramos a {self.who}.")
        self.empty.detail.setText("No hay fotografías indexadas todavía." if not total else
                                  "Ninguna cara indexada supera el umbral de similitud.")
        self.empty.detail.setVisible(not self.hits)

    def _match_boxes(self) -> dict[int, list]:
        """Rectángulo (normalizado a la foto) de la cara que coincidió en cada resultado."""
        out = {}
        for h in self.hits:
            ph = self.photos[h.photo_id]
            if not ph.width or not ph.height:
                continue
            f = next((f for f in self.ctx.db.photo_faces(h.photo_id) if f["id"] == h.face_id), None)
            if f:
                out[h.photo_id] = [(f["x"] / ph.width, f["y"] / ph.height, f["w"] / ph.width, f["h"] / ph.height)]
        return out

    def _item(self, h) -> GridItem:
        return to_item(self.ctx, self.photos[h.photo_id], sim(h.similarity), payload=h,
                       boxes=self.boxes.get(h.photo_id))

    def _tab(self, text: str, n: int, active: bool, key) -> None:
        b = Button(f"{text}  {n}", "ghost", "sm")
        if active:
            b.setStyleSheet(f"color: {C['ink']}; font-weight: 600; border-bottom: 2px solid {C['accent']};"
                            " border-radius: 0; padding-bottom: 2px;")
        b.clicked.connect(lambda: self._set_year(key))
        self.years.addWidget(b)

    def _set_year(self, k) -> None:
        self.year = k
        self._render()

    def _open(self, hit) -> None:
        self.ctx.navigate("photo", photo_id=hit.photo_id, sequence=[h.photo_id for h in self.sequence],
                          matches={h.photo_id: (h.face_id, h.similarity) for h in self.sequence},
                          back="gallery_results", who=self.who)
