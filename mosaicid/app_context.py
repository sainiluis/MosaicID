"""Contexto de aplicación: configuración, base, servicios y señales de cambio.

Es el único lugar que conoce a todos los servicios; las pantallas lo reciben y
reaccionan a sus señales.
"""
from __future__ import annotations

import logging
from dataclasses import replace

from PySide6.QtCore import QObject, Signal

from .config import Settings, camera_rotation, load_settings, save_settings
from .core.svm import SvmEngine
from .db import Database
from .paths import Paths
from .services.analyzer import LiveAnalyzer
from .services.camera import CameraService
from .services.engines import EngineSnapshot, build_snapshot

log = logging.getLogger(__name__)


class AppContext(QObject):
    peopleChanged = Signal()
    svmChanged = Signal()
    galleryChanged = Signal()
    settingsChanged = Signal()
    activityChanged = Signal()
    toastRequested = Signal(str, str, str)  # título, detalle, tono (ok|info|warn)
    navigateRequested = Signal(str, dict)

    def __init__(self, paths: Paths):
        super().__init__()
        self.paths = paths.ensure()
        self.settings: Settings = load_settings(paths.config_file)
        self.db = Database(paths.database)
        self.svm = SvmEngine(paths.svm_model, paths.svm_meta)
        self.models_ok = paths.yunet_model.exists() and paths.sface_model.exists()
        self.models_error = "" if self.models_ok else "Faltan los modelos ONNX en la carpeta models/."
        self.snapshot: EngineSnapshot = self._build_snapshot()
        self.camera = CameraService(lambda: self.settings)
        self.analyzer = LiveAnalyzer(self.camera.buffer, paths.yunet_model, paths.sface_model,
                                     lambda: self.settings, lambda: self.snapshot)
        self.analyzer.failed.connect(self._on_analyzer_failed)
        self._camera_users: set[str] = set()
        if self.models_ok:
            self.analyzer.start()
        self.camera.stateChanged.connect(self._on_camera_state)

    # ── motores ──────────────────────────────────────────────────────────
    def _build_snapshot(self) -> EngineSnapshot:
        X, y, sids = self.db.all_embeddings()
        names = {p.id: p.name for p in self.db.list_people()}
        return build_snapshot(X, y, sids, names, self.svm, self.settings)

    def refresh_engines(self) -> None:
        """Reconstruye la instantánea (reemplazo atómico; el analizador toma la nueva en el próximo frame)."""
        self.snapshot = self._build_snapshot()

    def people_changed(self) -> None:
        self.refresh_engines()
        self.peopleChanged.emit()
        self.svmChanged.emit()

    def svm_trained(self) -> None:
        self.svm.load()
        self.refresh_engines()
        self.svmChanged.emit()

    def svm_status(self) -> str:
        """trained | dirty | untrained | empty"""
        if not self.snapshot.dataset or len(self.snapshot.dataset.y) == 0:
            return "empty"
        if not self.svm.trained:
            return "untrained"
        return "dirty" if self.snapshot.svm_dirty else "trained"

    # ── cámara compartida ────────────────────────────────────────────────
    def acquire_camera(self, owner: str) -> None:
        self._camera_users.add(owner)
        self.camera.acquire(owner)
        self.analyzer.set_active(True)

    def release_camera(self, owner: str) -> None:
        self._camera_users.discard(owner)
        self.camera.release(owner)
        if not self._camera_users:
            self.analyzer.set_active(False)

    def _on_camera_state(self, state: str, msg: str) -> None:
        if state == "error":
            log.warning("Cámara: %s", msg)

    def _on_analyzer_failed(self, msg: str) -> None:
        self.models_ok = False
        self.models_error = msg
        self.toast("No se pudieron cargar los modelos", msg, "warn")

    # ── configuración ────────────────────────────────────────────────────
    def save_settings(self, s: Settings) -> None:
        old = self.settings
        self.settings = replace(s).validate()
        save_settings(self.paths.config_file, self.settings)
        log.info("Configuración guardada")
        self.refresh_engines()
        cam_keys = lambda x: (x.camera_source, x.camera_ip, x.camera_port, x.camera_index, x.width, x.height, x.fps)  # noqa: E731
        if cam_keys(old) != cam_keys(self.settings):
            self.camera.restart()
        elif camera_rotation(old) != camera_rotation(self.settings):
            self.camera.set_rotation(camera_rotation(self.settings))
        self.settingsChanged.emit()
        self.svmChanged.emit()

    # ── utilidades ───────────────────────────────────────────────────────
    def toast(self, title: str, detail: str = "", tone: str = "ok") -> None:
        self.toastRequested.emit(title, detail, tone)

    def log(self, kind: str, title: str, detail: str | None = None) -> None:
        self.db.log_activity(kind, title, detail)
        self.activityChanged.emit()

    def navigate(self, route: str, **params) -> None:
        self.navigateRequested.emit(route, params)

    def shutdown(self) -> None:
        self.camera.shutdown()
        self.analyzer.stop()
        self.analyzer.wait(3000)
        self.db.close()
