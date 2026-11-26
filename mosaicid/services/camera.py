"""CameraService — único dueño de la cámara.

- Un solo VideoCapture para toda la app: las vistas piden la cámara con acquire(owner)
  y la liberan con release(owner). Se abre con el primer interesado y se cierra con el último.
- Captura en un QThread propio. El último frame queda en un FrameBuffer (sin cola): el
  analizador toma siempre el más reciente y nunca acumula retraso.
- Maneja dispositivo ocupado, desconexión (reintentos) y frames inválidos.
- El FPS informado es el medido, no el configurado.
"""
from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal

log = logging.getLogger(__name__)


class FrameBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.frame: np.ndarray | None = None
        self.frame_id = 0

    def put(self, frame: np.ndarray) -> int:
        with self._cond:
            self.frame = frame
            self.frame_id += 1
            self._cond.notify_all()
            return self.frame_id

    def wait_newer(self, last_id: int, timeout: float = 0.5) -> tuple[np.ndarray | None, int]:
        with self._cond:
            if self.frame_id <= last_id:
                self._cond.wait(timeout)
            return self.frame, self.frame_id

    def clear(self) -> None:
        with self._cond:
            self.frame = None


class _StillCapture:
    """Solo para desarrollo/pruebas sin webcam: MOSAICID_FAKE_CAMERA=<imagen o carpeta>.
    Devuelve las imágenes en bucle como si fueran frames de cámara."""

    def __init__(self, source: str, w: int, h: int, fps: int):
        import os
        from pathlib import Path
        p = Path(source)
        files = sorted(x for x in (p.iterdir() if p.is_dir() else [p]) if x.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.frames = []
        for f in files:
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                s = min(w / img.shape[1], h / img.shape[0], 1.0)
                self.frames.append(cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        self.i, self.dt, self.t = 0, 1.0 / max(1, fps), 0.0
        self.hold = int(os.environ.get("MOSAICID_FAKE_HOLD", "90"))  # frames por imagen

    def isOpened(self):
        return bool(self.frames)

    def set(self, *_):
        return True

    def read(self):
        wait = self.dt - (time.perf_counter() - self.t)
        if wait > 0:
            time.sleep(wait)
        self.t = time.perf_counter()
        f = self.frames[(self.i // self.hold) % len(self.frames)]
        self.i += 1
        return True, f.copy()

    def release(self):
        self.frames = []


def backend() -> int:
    return cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY


DARK_LEVEL = 12  # brillo medio (0–255) por debajo del cual la imagen se considera negra


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class CameraWorker(QThread):
    frameReady = Signal(object, int)
    state = Signal(str, str)  # opening | live | error | stopped, mensaje
    darkChanged = Signal(bool)  # imagen negra sostenida: obturador de privacidad, tapa o sin luz

    def __init__(self, index: int, width: int, height: int, fps: int, buffer: FrameBuffer):
        super().__init__()
        # `index`: entero (cámara de Windows) o URL (cámara IP, p. ej. DroidCam por wifi)
        self.index, self.width, self.height, self.fps_target = index, width, height, fps
        self.is_url = isinstance(index, str)
        self.label = index if self.is_url else f"{index}"
        self.buffer = buffer
        self._stop = threading.Event()
        self.measured_fps = 0.0
        self.resolution = (0, 0)
        self.rotation = 0  # grados horarios; se puede cambiar en caliente desde el hilo de UI

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> cv2.VideoCapture | None:
        self.state.emit("opening", "Abriendo cámara…")
        import os
        fake = os.environ.get("MOSAICID_FAKE_CAMERA")
        if fake:
            cap = _StillCapture(fake, self.width, self.height, self.fps_target)
        elif self.is_url:
            # stream de red: tiempo máximo para conectar y para cada lectura (si el celular se va, no se cuelga)
            cap = cv2.VideoCapture(self.index, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
        else:
            cap = cv2.VideoCapture(self.index, backend())
        if not cap.isOpened():
            cap.release()
            return None
        if not self.is_url:  # la resolución de un stream IP la define la app del celular
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps_target)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, f = cap.read()
        if not ok or f is None or f.size == 0:
            cap.release()
            return None
        self.resolution = (f.shape[1], f.shape[0])
        log.info("Cámara %s abierta: %dx%d (pedido %dx%d @%d)", self.label, f.shape[1], f.shape[0],
                 self.width, self.height, self.fps_target)
        return cap

    def run(self) -> None:
        cap = None
        fails = 0
        stamps: list[float] = []
        n, dark, dark_votes = 0, False, 0
        while not self._stop.is_set():
            if cap is None:
                cap = self._open()
                if cap is None:
                    msg = (f"No se pudo conectar a {self.label}. Revisá que el celular esté en la misma red con DroidCam "
                           "abierta y que ninguna otra aplicación (por ejemplo el cliente de DroidCam en la PC) la esté usando: "
                           "admite una conexión a la vez." if self.is_url else
                           f"No se pudo abrir la cámara {self.index}. Puede estar ocupada por otra aplicación o desconectada.")
                    log.warning(msg)
                    self.state.emit("error", msg)
                    self._stop.wait(2.0)
                    continue
                self.state.emit("live", "")
                fails = 0
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0 or frame.ndim != 3:
                fails += 1
                if fails >= (2 if self.is_url else 20):  # cada lectura de red ya espera hasta 5 s
                    log.warning("Cámara %s sin frames válidos: se reintenta", self.label)
                    self.state.emit("error", "La cámara dejó de enviar imágenes. Reintentando…")
                    cap.release()
                    cap = None
                    self.buffer.clear()
                    self._stop.wait(1.5)
                else:
                    time.sleep(0.01)
                continue
            fails = 0
            rot = _ROTATE.get(self.rotation)
            if rot is not None:
                frame = cv2.rotate(frame, rot)
            now = time.perf_counter()
            stamps.append(now)
            while stamps and now - stamps[0] > 1.0:
                stamps.pop(0)
            self.measured_fps = float(len(stamps) - 1) / max(1e-3, stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
            self.resolution = (frame.shape[1], frame.shape[0])
            fid = self.buffer.put(frame)
            self.frameReady.emit(frame, fid)
            n += 1
            if n % 15 == 0:  # ~2 veces por segundo, sobre una muestra de píxeles (barato)
                is_dark = float(frame[::16, ::16].mean()) < DARK_LEVEL
                dark_votes = dark_votes + 1 if is_dark != dark else 0
                if dark_votes >= 2:  # cambio sostenido (~1 s) para no parpadear
                    dark, dark_votes = is_dark, 0
                    log.info("Cámara %s: imagen %s", self.label, "negra/oscura" if dark else "normal")
                    self.darkChanged.emit(dark)
        if cap is not None:
            cap.release()
        self.state.emit("stopped", "")
        log.info("Cámara %s cerrada", self.label)


class CameraService(QObject):
    frameReady = Signal(object, int)
    stateChanged = Signal(str, str)
    darkChanged = Signal(bool)

    def __init__(self, settings_getter):
        super().__init__()
        self._settings = settings_getter
        self.buffer = FrameBuffer()
        self._owners: set[str] = set()
        self._worker: CameraWorker | None = None
        self.state, self.message = "stopped", ""
        self.dark = False
        self._stop_timer = QTimer(self, singleShot=True, interval=600, timeout=self._stop_now)

    # ── API ──────────────────────────────────────────────────────────────
    def acquire(self, owner: str) -> None:
        self._owners.add(owner)
        self._stop_timer.stop()
        if self._worker is None:
            self._start()

    def release(self, owner: str) -> None:
        self._owners.discard(owner)
        if not self._owners:
            # pequeña demora: al navegar entre dos vistas con cámara no se reabre el dispositivo
            self._stop_timer.start()

    def set_rotation(self, degrees: int) -> None:
        """Gira la imagen sin reabrir la cámara (reabrir un stream IP tarda varios segundos)."""
        if self._worker is not None:
            self._worker.rotation = int(degrees) % 360
            self.buffer.clear()

    def restart(self) -> None:
        if self._worker is not None:
            self._stop_now()
            if self._owners:
                self._start()

    @property
    def fps(self) -> float:
        return self._worker.measured_fps if self._worker else 0.0

    @property
    def resolution(self) -> tuple[int, int]:
        return self._worker.resolution if self._worker else (0, 0)

    @property
    def index(self) -> int:
        return self._settings().camera_index

    def shutdown(self) -> None:
        self._owners.clear()
        self._stop_now()

    # ── interno ──────────────────────────────────────────────────────────
    def _start(self) -> None:
        s = self._settings()
        from ..config import camera_rotation, camera_url
        src = camera_url(s) if s.camera_source == "ip" else s.camera_index
        w = CameraWorker(src, s.width, s.height, s.fps, self.buffer)
        w.rotation = camera_rotation(s)
        w.frameReady.connect(self.frameReady)
        w.state.connect(self._on_state)
        w.darkChanged.connect(self._on_dark)
        self._worker = w
        w.start()

    def _stop_now(self) -> None:
        w, self._worker = self._worker, None
        if w is not None:
            for sig in (w.state, w.frameReady, w.darkChanged):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
            w.stop()
            if not w.wait(3000):
                log.warning("La cámara sigue cerrándose en segundo plano (%s)", w.label)
            from .workers import keep_until_finished
            keep_until_finished(w)
        self.buffer.clear()
        self._on_dark(False)
        self._on_state("stopped", "")

    def _on_state(self, state: str, msg: str) -> None:
        self.state, self.message = state, msg
        self.stateChanged.emit(state, msg)

    def _on_dark(self, dark: bool) -> None:
        if dark != self.dark:
            self.dark = dark
            self.darkChanged.emit(dark)


def probe_cameras(max_index: int = 5) -> list[int]:
    """Índices que se pueden abrir (bloqueante: llamar fuera del hilo de UI)."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend())
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
        cap.release()
    return found
