"""logs/mosaicid.log con rotación. Nunca se loguea cada frame."""
from __future__ import annotations

import logging
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(log_dir / "mosaicid.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr is not None:  # en el .exe sin consola stderr es None
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def log_versions() -> None:
    import cv2
    import numpy
    import PySide6
    import sklearn

    from . import APP_VERSION, PIPELINE_VERSION

    logging.getLogger("mosaicid").info(
        "MosaicID %s | pipeline %s | Python %s | OpenCV %s | PySide6 %s | NumPy %s | scikit-learn %s | %s",
        APP_VERSION, PIPELINE_VERSION, platform.python_version(), cv2.__version__, PySide6.__version__,
        numpy.__version__, sklearn.__version__, platform.platform())
