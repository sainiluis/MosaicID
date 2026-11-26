"""Indexación incremental de una carpeta de la galería (sin Qt, testeable).

Por foto: lectura con orientación EXIF → fecha → thumbnail en caché → pipeline completo
(todas las caras: bbox, score, embedding) → una transacción en SQLite.
Archivos sin cambios (misma ruta, mtime y tamaño) no se reprocesan.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..config import Settings
from ..db import Database
from .gallery import load_image, make_thumbnail, plan_index, scan_folder, source_from_path, taken_at, thumbnail_path
from .pipeline import FacePipeline

log = logging.getLogger(__name__)

ROTATIONS = (-30.0, 30.0)


def detector_signature(settings: Settings) -> str:
    """Identifica la configuración de detección: si cambia, la galería se reindexa completa."""
    rot = ",".join(f"{a:g}" for a in ROTATIONS) if settings.gallery_rotated_detection else "-"
    return f"max{settings.gallery_max_side}/multiscale/rot{rot}/score{settings.yunet_score_threshold:g}"


@dataclass
class IndexReport:
    folder: str
    total_files: int = 0
    indexed: int = 0
    unchanged: int = 0
    missing: int = 0
    errors: list[str] = field(default_factory=list)
    faces: int = 0
    seconds: float = 0.0
    cancelled: bool = False


def index_folder(db: Database, pipeline: FacePipeline, folder_id: int, settings: Settings, thumbs_dir: Path,
                 progress: Callable[[int, int, str], None] | None = None,
                 cancel: threading.Event | None = None) -> IndexReport:
    row = next(r for r in db.list_folders() if r["id"] == folder_id)
    root = Path(row["path"])
    rep = IndexReport(folder=str(root))
    t0 = time.perf_counter()
    if not root.exists():
        rep.errors.append(f"La carpeta no existe: {root}")
        return rep
    files = list(scan_folder(root))
    rep.total_files = len(files)
    sig = detector_signature(settings)
    sig_key = f"detector_signature:{folder_id}"
    plan = plan_index(files, db.photo_state(folder_id))
    if db.get_kv(sig_key) != sig and plan.unchanged:
        # la configuración de detección cambió: reprocesar todo (se conservan clasificaciones manuales)
        log.info("Detección cambió (%s → %s): se reindexa la carpeta completa", db.get_kv(sig_key), sig)
        plan.to_index, plan.unchanged = list(files), 0
    rep.unchanged = plan.unchanged
    rep.missing = len(plan.missing_ids)
    db.mark_missing(plan.missing_ids, delete=settings.gallery_missing_policy == "delete")
    n = len(plan.to_index)
    for i, f in enumerate(plan.to_index):
        if cancel is not None and cancel.is_set():
            rep.cancelled = True
            break
        progress and progress(i, n, f.name)
        try:
            st = f.stat()
            img, pil = load_image(f)
            date, date_src = taken_at(f, pil)
            make_thumbnail(pil, thumbnail_path(thumbs_dir, f, st.st_mtime, st.st_size))
            faces = pipeline.process(img, max_side=settings.gallery_max_side, multiscale=True,
                                     rotations=ROTATIONS if settings.gallery_rotated_detection else ())
            meta = dict(path=str(f), filename=f.name, taken_at=date, taken_at_source=date_src, mtime=st.st_mtime,
                        size=st.st_size, width=img.shape[1], height=img.shape[0],
                        source_type=source_from_path(f, root, row["source_type"]))
            db.upsert_photo(folder_id, meta, [dict(x=fc.detection.x, y=fc.detection.y, w=fc.detection.w,
                                                   h=fc.detection.h, score=fc.detection.score,
                                                   embedding=fc.embedding) for fc in faces])
            rep.indexed += 1
            rep.faces += len(faces)
        except Exception as e:
            log.warning("No se pudo indexar %s: %s", f, e)
            rep.errors.append(f"{f.name}: {e}")
            try:
                st = f.stat()
                db.upsert_photo(folder_id, dict(path=str(f), filename=f.name, mtime=st.st_mtime, size=st.st_size,
                                                source_type=row["source_type"], status="error", error=str(e)), [])
            except Exception:
                pass
    if not rep.cancelled:
        db.set_kv(sig_key, sig)
    progress and progress(n, n, "")
    rep.seconds = time.perf_counter() - t0
    log.info("Indexación %s: %d nuevas/cambiadas, %d sin cambios, %d faltantes, %d caras, %d errores, %.1fs",
             root, rep.indexed, rep.unchanged, rep.missing, rep.faces, len(rep.errors), rep.seconds)
    return rep
