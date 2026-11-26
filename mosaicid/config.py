"""Configuración centralizada (config/settings.json).

Todos los umbrales del sistema viven acá. Las claves desconocidas se ignoran y los
valores con tipo incorrecto vuelven al valor por defecto (se registra en el log).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE_TYPES = ("real", "synthetic", "generated", "test", "unknown")
DISTANCE_STRATEGIES = ("best", "centroid", "topk")


def camera_url(s: "Settings") -> str:
    """Stream MJPEG de DroidCam (la app del celular). Admite una sola conexión a la vez."""
    return f"http://{s.camera_ip}:{s.camera_port}/video"


def camera_rotation(s: "Settings") -> int:
    """Giro horario (grados) de la fuente de cámara activa."""
    return s.camera_rotation_ip if s.camera_source == "ip" else s.camera_rotation_device


def with_rotation(s: "Settings", degrees: int) -> "Settings":
    from dataclasses import replace
    key = "camera_rotation_ip" if s.camera_source == "ip" else "camera_rotation_device"
    return replace(s, **{key: int(degrees) % 360})


def camera_label(s: "Settings") -> str:
    if s.camera_source == "ip":
        return f"DroidCam · {s.camera_ip}:{s.camera_port}"
    from .ui.camera_view import camera_name  # import tardío: config no depende de Qt
    return camera_name(s.camera_index)


@dataclass
class Settings:
    # cámara: "ip" = stream MJPEG de DroidCam por wifi (sin cliente de PC); "device" = cámara de Windows
    camera_source: str = "ip"
    camera_ip: str = "192.168.0.244"
    camera_port: int = 4747
    camera_index: int = 0
    # giro de la imagen en grados horarios (0/90/180/270), por fuente: el celular en vertical llega de costado
    camera_rotation_ip: int = 0
    camera_rotation_device: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    # detección (YuNet)
    yunet_score_threshold: float = 0.9
    yunet_nms_threshold: float = 0.3
    yunet_top_k: int = 5000
    # motor distancia (SFace, FR_COSINE)
    distance_cosine_threshold: float = 0.363  # umbral recomendado por OpenCV para SFace
    distance_strategy: str = "best"  # best | centroid | topk
    distance_top_k: int = 3
    # motor SVM
    svm_c: float = 1.0
    svm_kernel: str = "linear"
    svm_rejection_threshold: float = 0.40  # margen sobre el azar: t = 1/K + m·(1−1/K) (calibration/CALIBRACION.md)
    # onboarding / calidad de muestra
    enrollment_sample_count: int = 8
    blur_threshold: float = 60.0  # varianza del Laplaciano sobre el rostro alineado
    min_face_size: int = 90  # px, lado menor del bounding box
    min_brightness: float = 60.0
    max_brightness: float = 205.0
    # rendimiento / estabilidad
    inference_every_n_frames: int = 2
    tracker_history: int = 7
    # galería
    gallery_match_threshold: float = 0.363
    gallery_max_side: int = 1600  # lado máximo para detectar en fotos grandes
    gallery_rotated_detection: bool = True  # detectar también rostros inclinados (±30°)
    gallery_missing_policy: str = "mark"  # mark | delete

    def validate(self) -> "Settings":
        self.camera_source = self.camera_source if self.camera_source in ("ip", "device") else "device"
        self.camera_ip = str(self.camera_ip).strip() or "192.168.0.244"
        self.camera_port = max(1, min(65535, int(self.camera_port)))
        self.camera_rotation_ip = int(self.camera_rotation_ip) % 360 // 90 * 90
        self.camera_rotation_device = int(self.camera_rotation_device) % 360 // 90 * 90
        self.distance_strategy = self.distance_strategy if self.distance_strategy in DISTANCE_STRATEGIES else "best"
        self.svm_kernel = self.svm_kernel if self.svm_kernel in ("linear", "rbf") else "linear"
        self.gallery_missing_policy = self.gallery_missing_policy if self.gallery_missing_policy in ("mark", "delete") else "mark"
        self.enrollment_sample_count = max(3, min(12, int(self.enrollment_sample_count)))
        self.inference_every_n_frames = max(1, min(6, int(self.inference_every_n_frames)))
        self.tracker_history = max(1, min(15, int(self.tracker_history)))
        for name in ("yunet_score_threshold", "yunet_nms_threshold", "distance_cosine_threshold",
                     "svm_rejection_threshold", "gallery_match_threshold"):
            setattr(self, name, float(min(1.0, max(0.0, getattr(self, name)))))
        return self


def load_settings(path: Path) -> Settings:
    s = Settings()
    if not path.exists():
        save_settings(path, s)
        return s
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        log.error("settings.json ilegible (%s); se usan valores por defecto", e)
        return s
    for f in fields(Settings):
        if f.name not in raw:
            continue
        default = getattr(s, f.name)
        value = raw[f.name]
        try:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
            elif isinstance(default, str):
                value = str(value)
            setattr(s, f.name, value)
        except (TypeError, ValueError):
            log.warning("Valor inválido para %s: %r (se usa %r)", f.name, value, default)
    return s.validate()


def save_settings(path: Path, s: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
