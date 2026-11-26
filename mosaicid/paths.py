"""Resolución de rutas: funciona igual desde el código fuente y desde el .exe de PyInstaller.

- Recursos de solo lectura (modelos ONNX, assets): junto al código o dentro del bundle.
- Datos escribibles (base, config, cache, logs): junto al ejecutable / raíz del proyecto.
Nunca se usan rutas absolutas fijas.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Carpeta escribible de la aplicación."""
    override = os.environ.get("MOSAICID_HOME")
    if override:
        return Path(override).resolve()
    if _frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    """Carpeta de recursos empaquetados (models/, assets/)."""
    if _frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


class Paths:
    def __init__(self, root: Path | None = None, resources: Path | None = None):
        self.root = Path(root) if root else app_root()
        self.resources = Path(resources) if resources else resource_root()

    # recursos
    @property
    def models(self) -> Path:
        local = self.root / "models"
        return local if local.exists() else self.resources / "models"

    @property
    def assets(self) -> Path:
        return self.resources / "assets"

    @property
    def yunet_model(self) -> Path:
        return self.models / "face_detection_yunet_2023mar.onnx"

    @property
    def sface_model(self) -> Path:
        return self.models / "face_recognition_sface_2021dec.onnx"

    # datos escribibles
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def database(self) -> Path:
        return self.data / "mosaicid.db"

    @property
    def samples(self) -> Path:
        return self.data / "samples"

    @property
    def svm_model(self) -> Path:
        return self.data / "svm" / "svm_model.joblib"

    @property
    def svm_meta(self) -> Path:
        return self.data / "svm" / "svm_meta.json"

    @property
    def config_file(self) -> Path:
        return self.root / "config" / "settings.json"

    @property
    def thumbnails(self) -> Path:
        return self.root / "cache" / "thumbnails"

    @property
    def included_gallery(self) -> Path:
        """Fotos que vienen con la app y se indexan solas en el primer arranque."""
        return self.root / "galeria_incluida"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "Paths":
        for p in (self.data, self.samples, self.svm_model.parent, self.config_file.parent, self.thumbnails, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self
