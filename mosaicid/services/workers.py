"""Trabajos en segundo plano: entrenamiento SVM, indexación de galería y carga de thumbnails.
Nada de esto corre en el hilo de la interfaz."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThread, Signal
from PySide6.QtGui import QImage

from ..core.indexer import IndexReport, index_folder
from ..core.pipeline import FacePipeline
from ..core.svm import Dataset, SvmEngine

log = logging.getLogger(__name__)

_retired: set[QThread] = set()


def keep_until_finished(t: QThread) -> None:
    """Mantiene viva la referencia a un QThread hasta que termine de verdad.

    Si Python libera un QThread que todavía corre, Qt aborta el proceso («QThread: Destroyed while thread is
    still running», 0xC0000409 en Qt6Core). Pasa cuando se suelta la última referencia en la señal de fin
    emitida desde run() o cuando wait() vence el plazo (p. ej. una cámara IP que tarda en responder).
    """
    if t.isFinished():
        t.deleteLater()
        return
    _retired.add(t)

    def done(t=t):
        _retired.discard(t)
        t.deleteLater()
    t.finished.connect(done)


class TrainingWorker(QThread):
    stage = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)

    def __init__(self, engine: SvmEngine, dataset: Dataset, C: float, kernel: str):
        super().__init__()
        self.engine, self.dataset, self.C, self.kernel = engine, dataset, C, kernel

    def run(self) -> None:
        try:
            meta = self.engine.train(self.dataset, self.C, self.kernel, progress=self.stage.emit)
            self.finished_ok.emit(meta)
        except Exception as e:
            log.exception("Fallo el entrenamiento")
            self.failed.emit(str(e))


class IndexWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(object)  # IndexReport
    failed = Signal(str)

    def __init__(self, db, paths, settings, folder_ids: list[int]):
        super().__init__()
        self.db, self.paths, self.settings, self.folder_ids = db, paths, settings, folder_ids
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            s = self.settings
            pipe = FacePipeline(self.paths.yunet_model, self.paths.sface_model, s.yunet_score_threshold,
                                s.yunet_nms_threshold, s.yunet_top_k)
            total = IndexReport(folder=", ".join(str(i) for i in self.folder_ids))
            for fid in self.folder_ids:
                r = index_folder(self.db, pipe, fid, s, self.paths.thumbnails, self.progress.emit, self.cancel)
                total.total_files += r.total_files
                total.indexed += r.indexed
                total.unchanged += r.unchanged
                total.missing += r.missing
                total.faces += r.faces
                total.errors += r.errors
                total.seconds += r.seconds
                total.cancelled = total.cancelled or r.cancelled
                if r.cancelled:
                    break
            self.finished_ok.emit(total)
        except Exception as e:
            log.exception("Fallo la indexación")
            self.failed.emit(str(e))
        finally:
            self.db.close()  # conexión propia de este hilo


class _Emitter(QObject):
    loaded = Signal(str, QImage)


class ThumbLoader(QRunnable):
    """Carga un thumbnail (o imagen) escalado en el QThreadPool global."""

    emitter = None

    def __init__(self, key: str, path: Path, max_side: int):
        super().__init__()
        self.key, self.path, self.max_side = key, Path(path), max_side
        self.setAutoDelete(True)

    def run(self) -> None:
        img = QImage(str(self.path))
        if not img.isNull() and max(img.width(), img.height()) > self.max_side:
            img = img.scaled(QSize(self.max_side, self.max_side), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        ThumbLoader.emitter.loaded.emit(self.key, img)


def thumb_emitter() -> _Emitter:
    if ThumbLoader.emitter is None:
        ThumbLoader.emitter = _Emitter()
    return ThumbLoader.emitter
