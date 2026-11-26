"""Base local SQLite (data/mosaicid.db).

- Una conexión por hilo (sqlite3 no comparte conexiones entre hilos) en modo WAL.
- Migraciones simples por número de versión (tabla schema_version).
- Embeddings como BLOB float32 (serialize_embedding / deserialize_embedding). Sin pickle.
- Una sola tabla `people`: los mismos embeddings alimentan al SVM y a Distancia.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import SOURCE_TYPES
from ..core.embeddings import deserialize_embedding, serialize_embedding

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, str] = {
    1: f"""
    CREATE TABLE people (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE COLLATE NOCASE,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    CREATE TABLE person_samples (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id       INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        kind            TEXT,
        image_path      TEXT,
        embedding       BLOB NOT NULL,
        quality_score   REAL,
        detection_score REAL,
        sharpness       REAL,
        created_at      TEXT NOT NULL
    );
    CREATE INDEX idx_samples_person ON person_samples(person_id);

    CREATE TABLE gallery_folders (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        path        TEXT NOT NULL UNIQUE,
        source_type TEXT NOT NULL DEFAULT 'unknown',
        added_at    TEXT NOT NULL
    );
    CREATE TABLE photos (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        folder_id       INTEGER REFERENCES gallery_folders(id) ON DELETE CASCADE,
        path            TEXT NOT NULL UNIQUE,
        filename        TEXT NOT NULL,
        taken_at        TEXT,
        taken_at_source TEXT,
        mtime           REAL NOT NULL,
        size            INTEGER NOT NULL,
        width           INTEGER,
        height          INTEGER,
        source_type     TEXT NOT NULL DEFAULT 'unknown'
                        CHECK (source_type IN ({",".join(repr(s) for s in SOURCE_TYPES)})),
        source_manual   INTEGER NOT NULL DEFAULT 0,
        status          TEXT NOT NULL DEFAULT 'ok',
        error           TEXT,
        face_count      INTEGER NOT NULL DEFAULT 0,
        indexed_at      TEXT
    );
    CREATE INDEX idx_photos_folder ON photos(folder_id);
    CREATE TABLE photo_faces (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        photo_id        INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
        x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
        embedding       BLOB NOT NULL,
        detection_score REAL NOT NULL
    );
    CREATE INDEX idx_faces_photo ON photo_faces(photo_id);

    CREATE TABLE activity (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        kind    TEXT NOT NULL,
        title   TEXT NOT NULL,
        detail  TEXT,
        at      TEXT NOT NULL
    );
    CREATE TABLE settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """,
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class Person:
    id: int
    name: str
    created_at: str
    updated_at: str
    sample_count: int = 0


@dataclass
class Sample:
    id: int
    person_id: int
    kind: str | None
    image_path: str | None
    embedding: np.ndarray
    quality_score: float | None
    detection_score: float | None
    sharpness: float | None
    created_at: str


@dataclass
class PhotoRow:
    id: int
    path: str
    filename: str
    taken_at: str | None
    taken_at_source: str | None
    mtime: float
    size: int
    width: int | None
    height: int | None
    source_type: str
    status: str
    face_count: int
    indexed_at: str | None
    folder_id: int | None


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.migrate()

    # ── conexión ─────────────────────────────────────────────────────────
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=15)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            c.execute("PRAGMA journal_mode = WAL")
            c.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        c = self.conn
        try:
            c.execute("BEGIN")
            yield c
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    def migrate(self) -> None:
        c = self.conn
        c.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = c.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current = row[0] or 0
        for v in sorted(MIGRATIONS):
            if v > current:
                log.info("Aplicando migración de base v%d", v)
                c.executescript("BEGIN;" + MIGRATIONS[v] + f"; INSERT INTO schema_version(version) VALUES ({v}); COMMIT;")

    @property
    def schema_version(self) -> int:
        return self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] or 0

    # ── personas ─────────────────────────────────────────────────────────
    def list_people(self) -> list[Person]:
        rows = self.conn.execute(
            "SELECT p.*, COUNT(s.id) AS n FROM people p LEFT JOIN person_samples s ON s.person_id = p.id "
            "GROUP BY p.id ORDER BY p.created_at, p.id").fetchall()
        return [Person(r["id"], r["name"], r["created_at"], r["updated_at"], r["n"]) for r in rows]

    def get_person(self, pid: int) -> Person | None:
        return next((p for p in self.list_people() if p.id == pid), None)

    def find_person_by_name(self, name: str) -> Person | None:
        r = self.conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (name.strip(),)).fetchone()
        return self.get_person(r["id"]) if r else None

    def create_person(self, name: str) -> int:
        name = name.strip()
        if not name:
            raise ValueError("El nombre no puede estar vacío")
        with self.tx() as c:
            cur = c.execute("INSERT INTO people(name, created_at, updated_at) VALUES (?,?,?)", (name, now_iso(), now_iso()))
            return int(cur.lastrowid)

    def delete_person(self, pid: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM people WHERE id = ?", (pid,))

    def add_samples(self, pid: int, samples: list[dict], replace: bool = False) -> list[int]:
        """samples: dicts con embedding, kind, image_path, quality_score, detection_score, sharpness."""
        ids = []
        with self.tx() as c:
            if replace:
                c.execute("DELETE FROM person_samples WHERE person_id = ?", (pid,))
            for s in samples:
                cur = c.execute(
                    "INSERT INTO person_samples(person_id, kind, image_path, embedding, quality_score, detection_score,"
                    " sharpness, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, s.get("kind"), s.get("image_path"), serialize_embedding(s["embedding"]),
                     s.get("quality_score"), s.get("detection_score"), s.get("sharpness"), now_iso()))
                ids.append(int(cur.lastrowid))
            c.execute("UPDATE people SET updated_at = ? WHERE id = ?", (now_iso(), pid))
        return ids

    def list_samples(self, pid: int | None = None) -> list[Sample]:
        q = "SELECT * FROM person_samples" + (" WHERE person_id = ?" if pid is not None else "") + " ORDER BY id"
        rows = self.conn.execute(q, (pid,) if pid is not None else ()).fetchall()
        return [Sample(r["id"], r["person_id"], r["kind"], r["image_path"], deserialize_embedding(r["embedding"]),
                       r["quality_score"], r["detection_score"], r["sharpness"], r["created_at"]) for r in rows]

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(embeddings (n,128), person_ids (n,), sample_ids (n,)) — fuente única para ambos motores."""
        rows = self.conn.execute("SELECT id, person_id, embedding FROM person_samples ORDER BY id").fetchall()
        if not rows:
            return np.zeros((0, 128), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        X = np.stack([deserialize_embedding(r["embedding"]) for r in rows])
        return X, np.array([r["person_id"] for r in rows], np.int64), np.array([r["id"] for r in rows], np.int64)

    # ── galería ──────────────────────────────────────────────────────────
    def add_folder(self, path: str, source_type: str = "unknown") -> int:
        if source_type not in SOURCE_TYPES:
            raise ValueError(source_type)
        with self.tx() as c:
            r = c.execute("SELECT id FROM gallery_folders WHERE path = ?", (path,)).fetchone()
            if r:
                c.execute("UPDATE gallery_folders SET source_type = ? WHERE id = ?", (source_type, r["id"]))
                return int(r["id"])
            return int(c.execute("INSERT INTO gallery_folders(path, source_type, added_at) VALUES (?,?,?)",
                                 (path, source_type, now_iso())).lastrowid)

    def list_folders(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM gallery_folders ORDER BY id").fetchall()

    def update_folder_path(self, folder_id: int, path: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE gallery_folders SET path = ? WHERE id = ?", (path, folder_id))

    def remove_folder(self, folder_id: int) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM gallery_folders WHERE id = ?", (folder_id,))

    def photo_state(self, folder_id: int) -> dict[str, tuple[float, int, str, int]]:
        """path → (mtime, size, status, id) para indexación incremental."""
        rows = self.conn.execute("SELECT id, path, mtime, size, status FROM photos WHERE folder_id = ?", (folder_id,))
        return {r["path"]: (r["mtime"], r["size"], r["status"], r["id"]) for r in rows}

    def upsert_photo(self, folder_id: int, meta: dict, faces: list[dict]) -> int:
        """Inserta o reemplaza una foto y TODAS sus caras en una sola transacción."""
        with self.tx() as c:
            r = c.execute("SELECT id, source_manual, source_type FROM photos WHERE path = ?", (meta["path"],)).fetchone()
            source = meta["source_type"]
            if r and r["source_manual"]:
                source = r["source_type"]  # respeta clasificación manual del usuario
            values = (folder_id, meta["path"], meta["filename"], meta.get("taken_at"), meta.get("taken_at_source"),
                      meta["mtime"], meta["size"], meta.get("width"), meta.get("height"), source,
                      meta.get("status", "ok"), meta.get("error"), len(faces), now_iso())
            if r:
                pid = int(r["id"])
                c.execute("UPDATE photos SET folder_id=?, path=?, filename=?, taken_at=?, taken_at_source=?, mtime=?,"
                          " size=?, width=?, height=?, source_type=?, status=?, error=?, face_count=?, indexed_at=?"
                          " WHERE id=?", values + (pid,))
                c.execute("DELETE FROM photo_faces WHERE photo_id = ?", (pid,))
            else:
                pid = int(c.execute(
                    "INSERT INTO photos(folder_id, path, filename, taken_at, taken_at_source, mtime, size, width, height,"
                    " source_type, status, error, face_count, indexed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    values).lastrowid)
            c.executemany(
                "INSERT INTO photo_faces(photo_id, x, y, w, h, embedding, detection_score) VALUES (?,?,?,?,?,?,?)",
                [(pid, f["x"], f["y"], f["w"], f["h"], serialize_embedding(f["embedding"]), f["score"]) for f in faces])
        return pid

    def mark_missing(self, ids: list[int], delete: bool) -> None:
        if not ids:
            return
        with self.tx() as c:
            if delete:
                c.executemany("DELETE FROM photos WHERE id = ?", [(i,) for i in ids])
            else:
                c.executemany("UPDATE photos SET status = 'missing' WHERE id = ?", [(i,) for i in ids])

    def set_photo_source(self, photo_id: int, source_type: str) -> None:
        if source_type not in SOURCE_TYPES:
            raise ValueError(source_type)
        with self.tx() as c:
            c.execute("UPDATE photos SET source_type = ?, source_manual = 1 WHERE id = ?", (source_type, photo_id))

    def _photo(self, r: sqlite3.Row) -> PhotoRow:
        return PhotoRow(r["id"], r["path"], r["filename"], r["taken_at"], r["taken_at_source"], r["mtime"], r["size"],
                        r["width"], r["height"], r["source_type"], r["status"], r["face_count"], r["indexed_at"],
                        r["folder_id"])

    def list_photos(self, only_real: bool = False, include_missing: bool = False) -> list[PhotoRow]:
        q = "SELECT * FROM photos WHERE 1=1"
        if not include_missing:
            q += " AND status = 'ok'"
        if only_real:
            q += " AND source_type = 'real'"
        q += " ORDER BY COALESCE(taken_at, '') DESC, id DESC"
        return [self._photo(r) for r in self.conn.execute(q)]

    def get_photo(self, photo_id: int) -> PhotoRow | None:
        r = self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return self._photo(r) if r else None

    def photo_faces(self, photo_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM photo_faces WHERE photo_id = ? ORDER BY id", (photo_id,)).fetchall()
        return [dict(id=r["id"], x=r["x"], y=r["y"], w=r["w"], h=r["h"], score=r["detection_score"],
                     embedding=deserialize_embedding(r["embedding"])) for r in rows]

    def face_matrix(self, only_real: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(embeddings, face_ids, photo_ids) de todas las caras indexadas vigentes."""
        q = ("SELECT f.id, f.photo_id, f.embedding FROM photo_faces f JOIN photos p ON p.id = f.photo_id"
             " WHERE p.status = 'ok'" + (" AND p.source_type = 'real'" if only_real else ""))
        rows = self.conn.execute(q).fetchall()
        if not rows:
            return np.zeros((0, 128), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        return (np.stack([deserialize_embedding(r["embedding"]) for r in rows]),
                np.array([r["id"] for r in rows], np.int64), np.array([r["photo_id"] for r in rows], np.int64))

    def gallery_counts(self) -> tuple[int, int]:
        r = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(face_count),0) FROM photos WHERE status='ok'").fetchone()
        return int(r[0]), int(r[1])

    def last_indexed(self) -> str | None:
        return self.conn.execute("SELECT MAX(indexed_at) FROM photos").fetchone()[0]

    # ── actividad y kv ───────────────────────────────────────────────────
    def log_activity(self, kind: str, title: str, detail: str | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO activity(kind, title, detail, at) VALUES (?,?,?,?)", (kind, title, detail, now_iso()))

    def recent_activity(self, limit: int = 5) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM activity ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def get_kv(self, key: str, default: str | None = None) -> str | None:
        r = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else default

    def set_kv(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO settings(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                      (key, value))
