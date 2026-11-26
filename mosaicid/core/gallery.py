"""Galería: escaneo incremental, lectura de imágenes/EXIF, thumbnails y búsqueda por rostro.

Sin dependencias de Qt (se ejecuta en hilos de trabajo y en tests).
La fotografía original nunca se modifica.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from PIL import Image, ImageOps

from ..config import SOURCE_TYPES
from .embeddings import as_embedding, l2_normalize

log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXIF_IFD = 0x8769
DATETIME_ORIGINAL = 36867
DATETIME = 306
THUMB_MAX = 480

Image.MAX_IMAGE_PIXELS = 200_000_000  # fotos grandes legítimas; evita el aviso de bomba de descompresión


# ── escaneo ──────────────────────────────────────────────────────────────
def scan_folder(root: Path) -> Iterator[Path]:
    """Recorre recursivamente y devuelve imágenes soportadas (orden estable)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(dirpath) / name


def source_from_path(path: Path, root: Path, folder_default: str) -> str:
    """Clasificación explícita por carpeta: si algún subdirectorio (relativo a la raíz)
    se llama exactamente real/synthetic/generated/test, se usa ese tipo. Si no, el tipo
    elegido por el usuario para la carpeta. Nunca se infiere desde el contenido."""
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = ()
    for p in reversed(parts):
        if p.lower() in SOURCE_TYPES:
            return p.lower()
    return folder_default


@dataclass
class IndexDecision:
    to_index: list[Path]
    unchanged: int
    missing_ids: list[int]


def plan_index(files: list[Path], known: dict[str, tuple[float, int, str, int]]) -> IndexDecision:
    """Decide qué reprocesar: nuevo o con mtime/size distinto → indexar; igual → saltear;
    registrado pero ausente en disco → faltante."""
    to_index, unchanged, seen = [], 0, set()
    for f in files:
        key = str(f)
        seen.add(key)
        st = f.stat()
        prev = known.get(key)
        if prev and abs(prev[0] - st.st_mtime) < 1e-3 and prev[1] == st.st_size and prev[2] in ("ok", "error"):
            unchanged += 1
        else:
            to_index.append(f)
    missing = [v[3] for k, v in known.items() if k not in seen]
    return IndexDecision(to_index, unchanged, missing)


# ── imágenes ─────────────────────────────────────────────────────────────
def load_image(path: Path) -> tuple[np.ndarray, Image.Image]:
    """Carga respetando la orientación EXIF. Devuelve (BGR uint8, imagen PIL orientada)."""
    with Image.open(path) as im:
        im.load()
        exif = im.getexif()
        oriented = ImageOps.exif_transpose(im).convert("RGB")
    oriented.info["mosaic_exif"] = exif
    arr = np.asarray(oriented)[:, :, ::-1].copy()
    return arr, oriented


def taken_at(path: Path, pil: Image.Image | None = None) -> tuple[str, str]:
    """Fecha de toma: EXIF DateTimeOriginal si existe; si no, fecha de modificación del archivo."""
    exif = None
    try:
        if pil is not None:
            exif = pil.info.get("mosaic_exif")
        if exif is None:
            with Image.open(path) as im:
                exif = im.getexif()
        raw = None
        if exif:
            raw = exif.get_ifd(EXIF_IFD).get(DATETIME_ORIGINAL) or exif.get(DATETIME)
        if raw:
            dt = datetime.strptime(str(raw).strip()[:19], "%Y:%m:%d %H:%M:%S")
            return dt.isoformat(timespec="seconds"), "exif"
    except Exception:  # EXIF ausente o mal formado: no es un error
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"), "filesystem"


def thumbnail_path(cache_dir: Path, path: Path, mtime: float, size: int) -> Path:
    """La clave incluye mtime y tamaño: si la fuente cambia, el thumbnail se invalida solo."""
    h = hashlib.sha1(str(path).lower().encode("utf-8")).hexdigest()[:20]
    return cache_dir / h[:2] / f"{h}_{int(mtime)}_{size}.jpg"


def make_thumbnail(pil: Image.Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for old in dest.parent.glob(dest.name.split("_")[0] + "_*.jpg"):
        if old != dest:
            try:
                old.unlink()
            except OSError:
                pass
    t = pil.copy()
    t.thumbnail((THUMB_MAX, THUMB_MAX), Image.Resampling.LANCZOS)
    t.save(dest, "JPEG", quality=84, optimize=True)


# ── búsqueda ─────────────────────────────────────────────────────────────
@dataclass
class SearchHit:
    photo_id: int
    face_id: int
    similarity: float


def search_faces(queries: np.ndarray, face_embeddings: np.ndarray, face_ids: np.ndarray, photo_ids: np.ndarray,
                 threshold: float) -> list[SearchHit]:
    """Compara una o más consultas (p. ej. todas las referencias de una persona) contra todas
    las caras indexadas. Por cara: mejor similitud entre consultas. Por foto: mejor cara."""
    if len(face_embeddings) == 0:
        return []
    Q = l2_normalize(np.stack([as_embedding(q) for q in np.atleast_2d(queries)]))
    F = l2_normalize(face_embeddings)
    sims = (F @ Q.T).max(axis=1)
    best: dict[int, SearchHit] = {}
    for i in np.nonzero(sims >= threshold)[0]:
        pid = int(photo_ids[i])
        s = float(sims[i])
        if pid not in best or s > best[pid].similarity:
            best[pid] = SearchHit(pid, int(face_ids[i]), s)
    return sorted(best.values(), key=lambda h: -h.similarity)


ProgressFn = Callable[[int, int, str], None]
