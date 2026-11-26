"""LiveAnalyzer — inferencia en vivo en un hilo propio.

- Toma siempre el frame MÁS RECIENTE del FrameBuffer (descarta intermedios: sin retraso acumulado).
- Ejecuta el pipeline cada `inference_every_n_frames` frames de cámara.
- Para cada cara: pipeline único → embedding → ambos motores (SVM y Distancia) → tracker.
- Emite `LiveAnalysis` con cajas, crops alineados reales, embeddings, decisiones crudas
  y decisiones estabilizadas por votación. Consumidores: Inicio, Reconocimiento, Comparar,
  Onboarding y Escaneo de galería.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core.pipeline import Face, FacePipeline
from ..core.tracker import Tracker
from .engines import EngineSnapshot, FaceDecision

log = logging.getLogger(__name__)


@dataclass
class LiveFace:
    track_id: int
    face: Face  # detección + crop alineado + embedding
    decision: FaceDecision  # resultado crudo del frame
    stable: dict[str, tuple[int | None, float]]  # motor → (persona, métrica promediada)


@dataclass
class LiveAnalysis:
    frame_id: int
    frame: np.ndarray
    faces: list[LiveFace]
    snapshot: EngineSnapshot
    inference_ms: float
    inference_fps: float
    timings_ms: dict = field(default_factory=dict)


class LiveAnalyzer(QThread):
    analysis = Signal(object)
    failed = Signal(str)

    def __init__(self, buffer, yunet, sface, settings_getter, snapshot_getter):
        super().__init__()
        self.buffer = buffer
        self.yunet, self.sface = yunet, sface
        self._settings = settings_getter
        self._snapshot = snapshot_getter
        self._stop = threading.Event()
        self._active = threading.Event()
        self.tracker = Tracker()
        self._reset = False

    def set_active(self, on: bool) -> None:
        if on:
            self._active.set()
        else:
            self._active.clear()
            self._reset = True  # el hilo de trabajo reinicia el tracker (sin carreras)

    def stop(self) -> None:
        self._stop.set()
        self._active.set()

    def run(self) -> None:
        try:
            s = self._settings()
            pipe = FacePipeline(self.yunet, self.sface, s.yunet_score_threshold, s.yunet_nms_threshold, s.yunet_top_k)
        except Exception as e:
            log.exception("No se pudo iniciar el analizador")
            self.failed.emit(str(e))
            return
        last_id = 0  # último frame procesado
        seen_id = 0  # último frame visto
        stamps: list[float] = []
        thr = (s.yunet_score_threshold, s.yunet_nms_threshold)
        while not self._stop.is_set():
            if not self._active.wait(0.3):
                continue
            frame, fid = self.buffer.wait_newer(seen_id, 0.3)
            if frame is None or fid <= seen_id:
                continue
            seen_id = fid
            s = self._settings()
            if last_id and fid - last_id < s.inference_every_n_frames:
                continue
            last_id = fid
            if (s.yunet_score_threshold, s.yunet_nms_threshold) != thr:
                thr = (s.yunet_score_threshold, s.yunet_nms_threshold)
                pipe.set_thresholds(*thr)
            if self._reset or self.tracker.history != s.tracker_history:
                self._reset = False
                self.tracker = Tracker(history=s.tracker_history)
            snap = self._snapshot()
            t0 = time.perf_counter()
            try:
                faces = pipe.process(frame)
                decisions = [snap.decide(f.embedding) for f in faces]
            except Exception as e:  # un frame defectuoso no detiene el análisis
                log.warning("Error procesando frame: %s", e)
                continue
            tracks = self.tracker.update([f.detection.box for f in faces])
            live = []
            for f, d, tr in zip(faces, decisions, tracks):
                tr.add("distance", d.distance.person_id, d.distance.similarity)
                if d.svm is not None:
                    tr.add("svm", d.svm.person_id, d.svm.probability)
                live.append(LiveFace(tr.id, f, d, {"distance": tr.label("distance"), "svm": tr.label("svm")}))
            dt = (time.perf_counter() - t0) * 1000
            now = time.perf_counter()
            stamps.append(now)
            stamps = [t for t in stamps if now - t <= 2.0]
            ifps = (len(stamps) - 1) / max(1e-3, stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            timings = {}
            for f in faces:
                for k, v in f.timings_ms.items():
                    timings[k] = timings.get(k, 0.0) + v
            self.analysis.emit(LiveAnalysis(fid, frame, live, snap, dt, ifps, timings))
