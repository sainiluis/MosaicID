"""Pipeline facial — implementado UNA sola vez y compartido por todo el sistema.

    frame → YuNet (bbox + 5 landmarks + score) → SFace.alignCrop() → cara 112×112
          → SFace.feature() → embedding 128D

Ambos motores (SVM y Distancia), el onboarding y la galería consumen la salida de
`FacePipeline.process()`. Las redes DNN de OpenCV no son thread-safe: cada hilo
crea su propia instancia (ver `services`).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .embeddings import as_embedding

log = logging.getLogger(__name__)

try:  # silencia avisos del nuevo motor de grafos de OpenCV 5
    cv2.setLogLevel(2)  # ERROR
except Exception:  # pragma: no cover
    pass

FR_COSINE = getattr(cv2, "FaceRecognizerSF_FR_COSINE", 0)
LANDMARK_NAMES = ("ojo derecho", "ojo izquierdo", "nariz", "comisura derecha", "comisura izquierda")


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray  # (5, 2)
    raw: np.ndarray  # fila YuNet (15,) en coordenadas de la imagen procesada

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def scaled(self, s: float) -> "Detection":
        return Detection(self.x * s, self.y * s, self.w * s, self.h * s, self.score, self.landmarks * s, self.raw)


@dataclass
class Face:
    """Resultado completo del pipeline para un rostro."""

    detection: Detection
    aligned: np.ndarray  # 112×112 BGR (salida real de alignCrop)
    embedding: np.ndarray  # (128,) float32, salida cruda de SFace.feature()
    timings_ms: dict = field(default_factory=dict)


class ModelError(RuntimeError):
    pass


class FacePipeline:
    def __init__(self, yunet_path: Path, sface_path: Path, score_threshold: float = 0.9,
                 nms_threshold: float = 0.3, top_k: int = 5000):
        for p in (yunet_path, sface_path):
            if not Path(p).exists():
                raise ModelError(f"No se encontró el modelo {p}")
        t = time.perf_counter()
        self.detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), score_threshold, nms_threshold, top_k)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self._input_size: tuple[int, int] | None = None
        self.score_threshold = score_threshold
        log.info("Modelos cargados en %.0f ms (YuNet=%s, SFace=%s)", (time.perf_counter() - t) * 1000,
                 Path(yunet_path).name, Path(sface_path).name)

    def set_thresholds(self, score: float, nms: float) -> None:
        self.detector.setScoreThreshold(float(score))
        self.detector.setNMSThreshold(float(nms))
        self.score_threshold = score

    # ── etapas ────────────────────────────────────────────────────────────
    def detect(self, img: np.ndarray) -> list[Detection]:
        h, w = img.shape[:2]
        if self._input_size != (w, h):
            self.detector.setInputSize((w, h))
            self._input_size = (w, h)
        _, faces = self.detector.detect(img)
        if faces is None:
            return []
        out = []
        for row in faces:
            row = np.asarray(row, dtype=np.float32)
            out.append(Detection(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[14]),
                                 row[4:14].reshape(5, 2).copy(), row.copy()))
        return out

    def align(self, img: np.ndarray, det: Detection) -> np.ndarray:
        return self.recognizer.alignCrop(img, det.raw)

    def embed(self, aligned: np.ndarray) -> np.ndarray:
        return as_embedding(self.recognizer.feature(aligned).reshape(-1).copy())

    def match(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno usando la implementación oficial de OpenCV."""
        return float(self.recognizer.match(as_embedding(a).reshape(1, -1), as_embedding(b).reshape(1, -1), FR_COSINE))

    # ── pipeline completo ────────────────────────────────────────────────
    def process(self, img: np.ndarray, max_side: int | None = None, multiscale: bool = False,
                rotations: tuple[float, ...] = ()) -> list[Face]:
        """Detecta TODAS las caras y calcula alineamiento + embedding de cada una.

        - `max_side`: si la imagen es más grande, se procesa una copia reducida; las cajas se
          devuelven siempre en coordenadas de la imagen original.
        - YuNet pierde rostros muy grandes en la imagen de trabajo (≳350 px). Por eso:
          * `multiscale=True` (galería): se detecta también a media escala y se fusiona por IoU.
          * en vivo: si no se detecta nada, se reintenta una vez a media escala.
        - YuNet tampoco detecta bien rostros muy inclinados (selfies en diagonal, ≳25°).
          `rotations` (galería) repite la detección sobre la imagen rotada esos grados; un rostro
          nuevo solo se agrega si no se superpone con uno ya encontrado. alignCrop() corrige el
          giro con los landmarks, así que el embedding es el de la cara derecha.
        Cada rostro se alinea sobre la imagen de trabajo donde fue detectado.
        """
        def work_at(limit: int | None) -> tuple[np.ndarray, float]:
            if limit and max(img.shape[:2]) > limit:
                sc = limit / max(img.shape[:2])
                return cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), sc
            return img, 1.0

        t0 = time.perf_counter()
        # cada hallazgo: (detección en su imagen de trabajo, imagen de trabajo, detección en coords originales)
        found: list[tuple[Detection, np.ndarray, Detection]] = []

        def add(dets, w_img, to_orig) -> None:
            for d in dets:
                o = to_orig(d)
                if all(_iou(o.box, f[2].box) < 0.3 and not _center_inside(o.box, f[2].box) for f in found):
                    found.append((d, w_img, o))

        work, scale = work_at(max_side)
        add(self.detect(work), work, lambda d: d.scaled(1 / scale) if scale != 1.0 else d)
        half_limit = max(320, int(max(work.shape[:2]) * 0.5))
        if (multiscale or not found) and max(work.shape[:2]) > 400:
            work2, scale2 = work_at(half_limit)
            add(self.detect(work2), work2, lambda d: d.scaled(1 / scale2))
        for ang in rotations:
            for limit in (max_side, half_limit) if multiscale else (max_side,):
                base, sc = work_at(limit)
                rot, inv = _rotate(base, ang)
                add(self.detect(rot), rot, lambda d, inv=inv, sc=sc: _unrotate(d, inv, 1 / sc))
        t1 = time.perf_counter()
        faces: list[Face] = []
        for d, w_img, orig in found:
            ta = time.perf_counter()
            aligned = self.align(w_img, d)
            tb = time.perf_counter()
            emb = self.embed(aligned)
            tc = time.perf_counter()
            faces.append(Face(orig, aligned, emb,
                              {"detect": (t1 - t0) * 1000 / max(1, len(found)), "align": (tb - ta) * 1000,
                               "embed": (tc - tb) * 1000}))
        return faces


def _rotate(img: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rota sin recortar (lienzo ampliado). Devuelve la imagen y la afín inversa (rotada → base)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c), int(h * c + w * s)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rot = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rot, cv2.invertAffineTransform(m)


def _unrotate(d: Detection, inv: np.ndarray, k: float) -> Detection:
    """Lleva una detección de la imagen rotada a coords originales (caja = envolvente del rectángulo girado)."""
    pts = np.array([[d.x, d.y], [d.x + d.w, d.y], [d.x, d.y + d.h], [d.x + d.w, d.y + d.h]], np.float32)
    corners = (pts @ inv[:, :2].T + inv[:, 2]) * k
    lm = (d.landmarks @ inv[:, :2].T + inv[:, 2]) * k
    # se encoge la envolvente hacia el centro para que se parezca a la caja de una cara derecha
    cx, cy = corners.mean(axis=0)
    side = (d.w + d.h) / 2 * k
    return Detection(float(cx - side / 2), float(cy - side / 2), float(side), float(side), d.score,
                     lm.astype(np.float32), d.raw)


def _center_inside(a, b) -> bool:
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
