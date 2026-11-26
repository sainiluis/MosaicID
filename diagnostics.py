"""Diagnóstico de MosaicID: verifica entorno, modelos, base y cámara. 100 % local.

Uso:  python diagnostics.py [--camera N] [--no-camera]
"""
from __future__ import annotations

import argparse
import platform
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, FAIL, WARN = "OK  ", "FALLA", "AVISO"
results: list[tuple[str, str, str]] = []


def check(name: str, fn):
    try:
        status, detail = fn()
    except Exception as e:  # noqa: BLE001
        status, detail = FAIL, f"{type(e).__name__}: {e}"
    results.append((status, name, detail))
    print(f"[{status}] {name:<22} {detail}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-camera", action="store_true")
    a = ap.parse_args()

    from mosaicid import APP_VERSION, PIPELINE_VERSION
    from mosaicid.paths import Paths
    paths = Paths()
    print(f"MosaicID {APP_VERSION} · pipeline {PIPELINE_VERSION}\nRaíz: {paths.root}\n")

    check("Python", lambda: (OK if sys.version_info >= (3, 10) else FAIL,
                             f"{platform.python_version()} · {platform.platform()}"))

    def cv():
        import cv2
        major = int(cv2.__version__.split(".")[0])
        has = hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")
        return (OK if has and major >= 5 else WARN if has else FAIL), f"{cv2.__version__} · FaceDetectorYN/FaceRecognizerSF: {has}"
    check("OpenCV", cv)

    def qt():
        import PySide6
        from PySide6 import QtCore, QtSvg, QtWidgets  # noqa: F401
        return OK, f"{PySide6.__version__} (Qt {QtCore.qVersion()})"
    check("PySide6", qt)
    check("NumPy", lambda: (OK, __import__("numpy").__version__))
    check("scikit-learn", lambda: (OK, __import__("sklearn").__version__))
    check("joblib", lambda: (OK, __import__("joblib").__version__))
    check("Pillow", lambda: (OK, __import__("PIL").__version__))

    pipe_holder = {}

    def models():
        from mosaicid.core.pipeline import FacePipeline
        for p in (paths.yunet_model, paths.sface_model):
            if not p.exists():
                return FAIL, f"falta {p}"
        t = time.perf_counter()
        pipe_holder["p"] = FacePipeline(paths.yunet_model, paths.sface_model)
        return OK, f"cargados en {(time.perf_counter() - t) * 1000:.0f} ms"
    check("YuNet + SFace", models)

    def pipeline():
        import numpy as np
        p = pipe_holder.get("p")
        if p is None:
            return FAIL, "modelos no cargados"
        sample = paths.assets / "diagnostics_face.jpg"
        import cv2
        img = cv2.imread(str(sample)) if sample.exists() else np.zeros((480, 640, 3), np.uint8)
        t = time.perf_counter()
        faces = p.process(img)
        ms = (time.perf_counter() - t) * 1000
        if faces:
            e = faces[0].embedding
            return OK, f"{len(faces)} rostro(s) · embedding {e.shape} {e.dtype} · {ms:.0f} ms"
        return WARN, f"sin rostros en la imagen de prueba ({ms:.0f} ms)"
    check("Pipeline", pipeline)

    def db():
        from mosaicid.db import Database
        if not paths.database.exists():
            return WARN, "todavía no existe (se crea al abrir la app)"
        d = Database(paths.database)
        n = len(d.list_people())
        ph, fc = d.gallery_counts()
        ic = d.conn.execute("PRAGMA integrity_check").fetchone()[0]
        return (OK if ic == "ok" else FAIL), f"esquema v{d.schema_version} · {n} personas · {ph} fotos · {fc} rostros · integridad {ic}"
    check("SQLite", db)

    def svm():
        from mosaicid.core.svm import SvmEngine
        e = SvmEngine(paths.svm_model, paths.svm_meta)
        if not e.trained:
            return WARN, "sin modelo entrenado"
        return OK, f"{len(e.meta.get('classes', []))} clases · entrenado {e.meta.get('trained_at')}"
    check("Clasificador SVM", svm)

    def config():
        from mosaicid.config import load_settings
        s = load_settings(paths.config_file)
        return OK, (f"cámara {s.camera_index} · coseno {s.distance_cosine_threshold} · margen SVM "
                    f"{s.svm_rejection_threshold} · {paths.config_file}")
    check("Configuración", config)

    if not a.no_camera:
        def camera():
            import cv2
            from mosaicid.config import load_settings
            from mosaicid.services.camera import backend
            idx = a.camera if a.camera is not None else load_settings(paths.config_file).camera_index
            cap = cv2.VideoCapture(idx, backend())
            try:
                if not cap.isOpened():
                    return FAIL, f"no se pudo abrir la cámara {idx} (¿ocupada o desconectada?)"
                ok, f = cap.read()
                if not ok or f is None:
                    return FAIL, f"cámara {idx} abierta pero sin frames"
                return OK, f"cámara {idx}: {f.shape[1]}×{f.shape[0]}"
            finally:
                cap.release()
        check("Cámara", camera)

    fails = sum(1 for r in results if r[0] == FAIL)
    print(f"\n{len(results)} verificaciones · {fails} con falla")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
