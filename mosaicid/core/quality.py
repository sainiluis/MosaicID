"""Control de calidad de muestras para el onboarding.

Una muestra solo se guarda si:
- hay exactamente una cara,
- el score de YuNet supera el umbral,
- el rostro tiene un tamaño mínimo,
- el rostro está completo dentro del cuadro,
- la cara alineada es nítida (varianza del Laplaciano),
- la iluminación es razonable (brillo medio de la cara alineada).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import Settings
from .pipeline import Face


@dataclass
class QualityReport:
    ok: bool
    message: str  # texto para el usuario
    code: str  # ok | no_face | multiple | low_score | small | outside | blur | dark | bright
    sharpness: float = 0.0
    brightness: float = 0.0
    score: float = 0.0
    face_size: float = 0.0

    @property
    def quality(self) -> float:
        """Puntaje 0–1 orientativo para mostrar (no se usa para decidir)."""
        if not self.ok:
            return 0.0
        return float(min(1.0, 0.5 * self.score + 0.5 * min(1.0, self.sharpness / 300.0)))


def sharpness(aligned: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(aligned: np.ndarray) -> float:
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = gray[h // 6: h - h // 6, w // 6: w - w // 6]  # evita fondo
    return float(center.mean())


def check_sample(faces: list[Face], frame_shape: tuple[int, ...], s: Settings) -> tuple[QualityReport, Face | None]:
    if not faces:
        return QualityReport(False, "No se detecta ningún rostro.", "no_face"), None
    if len(faces) > 1:
        return QualityReport(False, "Se detectaron dos rostros." if len(faces) == 2 else f"Se detectaron {len(faces)} rostros.", "multiple"), None
    f = faces[0]
    d = f.detection
    sh, br = sharpness(f.aligned), brightness(f.aligned)
    size = min(d.w, d.h)
    base = dict(sharpness=sh, brightness=br, score=d.score, face_size=size)
    H, W = frame_shape[:2]
    margin = 4
    if d.score < s.yunet_score_threshold:
        return QualityReport(False, "Rostro poco claro. Mirá a la cámara.", "low_score", **base), f
    if d.x < margin or d.y < margin or d.x + d.w > W - margin or d.y + d.h > H - margin:
        return QualityReport(False, "El rostro no está completo en el cuadro.", "outside", **base), f
    if size < s.min_face_size:
        return QualityReport(False, "Acercate un poco.", "small", **base), f
    if br < s.min_brightness:
        return QualityReport(False, "Mejorá la iluminación.", "dark", **base), f
    if br > s.max_brightness:
        return QualityReport(False, "Hay demasiada luz sobre el rostro.", "bright", **base), f
    if sh < s.blur_threshold:
        return QualityReport(False, "Imagen borrosa.", "blur", **base), f
    return QualityReport(True, "Muestra aceptada.", "ok", **base), f
