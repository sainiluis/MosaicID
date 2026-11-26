"""Utilidades compartidas por las pantallas con cámara en vivo."""
from __future__ import annotations

from datetime import datetime

from ...app_context import AppContext
from ...config import camera_label
from ...services.analyzer import LiveAnalysis, LiveFace
from ..camera_view import CameraMeta, CameraView, Overlay, camera_name


def pct(v: float) -> str:
    return f"{round(v * 100)}%"


def sim(v: float) -> str:
    return f"{v:.2f}"


def relative(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    m = (datetime.now() - d).total_seconds() / 60
    if m < 1:
        return "Ahora"
    if m < 60:
        return f"Hace {int(m)} min"
    if m < 60 * 20:
        return f"Hace {int(m / 60)} h"
    if m < 60 * 48:
        return "Ayer"
    return f"Hace {int(m / 60 / 24)} días"


MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MONTHS_L = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
            "noviembre", "diciembre"]


def short_date(iso: str | None) -> str:
    if not iso:
        return "Sin fecha"
    d = datetime.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def long_date(iso: str | None) -> str:
    if not iso:
        return "Sin fecha"
    d = datetime.fromisoformat(iso)
    return f"{d.day} de {MONTHS_L[d.month - 1]} de {d.year} · {d:%H:%M}"


def overlays_for(a: LiveAnalysis, engine: str, selected: int | None = None) -> list[Overlay]:
    """Etiquetas estabilizadas (votación del tracker) para el motor elegido."""
    snap = a.snapshot
    out = []
    for lf in a.faces:
        box = lf.face.detection.box
        dp, dm = lf.stable["distance"]
        sp, sm = lf.stable["svm"]
        has_svm = lf.decision.svm is not None
        if engine == "distance":
            if snap.matcher.empty:
                out.append(Overlay(lf.track_id, box, "Desconocido", "", "unknown", lf.track_id == selected))
                continue
            out.append(Overlay(lf.track_id, box, snap.short(dp), sim(dm) if dp else "", "known" if dp else "unknown",
                               lf.track_id == selected))
        elif engine == "svm":
            if not has_svm:
                out.append(Overlay(lf.track_id, box, "SVM sin entrenar", "", "neutral", lf.track_id == selected))
                continue
            out.append(Overlay(lf.track_id, box, snap.short(sp), f"SVM {pct(sm)}" if sp else "",
                               "known" if sp else "unknown", lf.track_id == selected))
        else:
            a_ = snap.short(sp, "—") if has_svm else "—"
            b_ = snap.short(dp, "—")
            agree = has_svm and sp == dp
            lab = (snap.short(dp) if agree else f"{a_} / {b_}")
            score = (f"{pct(sm)} · {sim(dm)}" if has_svm else sim(dm))
            out.append(Overlay(lf.track_id, box, lab, score, "known" if (sp or dp) else "unknown",
                               lf.track_id == selected))
    return out


class LiveBinding:
    """Conecta una CameraView (y opcionalmente CameraMeta) al servicio de cámara y al analizador
    mientras la pantalla está visible. Libera la cámara al salir."""

    def __init__(self, ctx: AppContext, owner: str, view: CameraView, meta: CameraMeta | None, on_analysis):
        self.ctx, self.owner, self.view, self.meta, self.on_analysis = ctx, owner, view, meta, on_analysis
        self.active = False
        self.last: LiveAnalysis | None = None

    def start(self) -> None:
        if self.active:
            return
        self.active = True
        cam = self.ctx.camera
        cam.frameReady.connect(self._frame)
        cam.stateChanged.connect(self._state)
        cam.darkChanged.connect(self._dark)
        self.ctx.analyzer.analysis.connect(self._analysis)
        if not self.ctx.models_ok:
            self.view.set_state("error", self.ctx.models_error)
            return
        self.ctx.acquire_camera(self.owner)
        self.view.set_state(cam.state if cam.state != "stopped" else "opening", cam.message)
        self._dark(cam.dark)
        self._meta(None)

    def stop(self) -> None:
        if not self.active:
            return
        self.active = False
        cam = self.ctx.camera
        for sig, slot in ((cam.frameReady, self._frame), (cam.stateChanged, self._state), (cam.darkChanged, self._dark),
                          (self.ctx.analyzer.analysis, self._analysis)):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self.ctx.release_camera(self.owner)
        self.view.set_state("stopped", "")

    def _frame(self, frame, _fid) -> None:
        self.view.set_frame(frame)

    def _dark(self, dark: bool) -> None:
        self.view.set_hint("La imagen está negra. Revisá que la cámara no esté tapada (obturador de privacidad) "
                           "y que haya luz, o elegí otra cámara en el selector." if dark else "")

    def _state(self, state: str, msg: str) -> None:
        self.view.set_state(state, msg)
        self._meta(self.last)

    def _analysis(self, a: LiveAnalysis) -> None:
        self.last = a
        self._meta(a)
        self.on_analysis(a)

    def _meta(self, a: LiveAnalysis | None) -> None:
        if self.meta is None:
            return
        cam = self.ctx.camera
        live = cam.state == "live"
        self.meta.update_info(camera_label(self.ctx.settings), cam.resolution if live else (0, 0),
                              cam.fps if live else 0.0, a.inference_ms if (a and live) else None,
                              a.inference_fps if (a and live) else None, len(a.faces) if (a and live) else None)


def largest(faces: list[LiveFace]) -> LiveFace | None:
    return max(faces, key=lambda f: f.face.detection.w * f.face.detection.h, default=None)


def camera_picker(ctx: AppContext):
    """Selector de cámara para las vistas en vivo: lista los dispositivos por nombre, cambia de cámara al
    instante (se guarda en la configuración) y permite recargarla (p. ej. después de conectar el celular a
    DroidCam/Camo). La lista se actualiza sola cuando Windows agrega o quita una cámara."""

    from dataclasses import replace

    from PySide6.QtCore import Qt
    from PySide6.QtMultimedia import QMediaDevices
    from PySide6.QtWidgets import QComboBox, QHBoxLayout, QInputDialog, QLineEdit, QWidget

    from ..widgets import Button

    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    combo = QComboBox()
    combo.setMinimumWidth(150)
    combo.setMaximumWidth(280)
    combo.setToolTip("Cámara en uso. Cambiarla se aplica enseguida y queda guardada.")
    reload_btn = Button("", "ghost", "sm", "refresh-cw")
    reload_btn.setToolTip("Recargar cámaras (volver a abrir la cámara y actualizar la lista)")
    edit_btn = Button("", "ghost", "sm", "settings-2")
    edit_btn.setToolTip("Cambiar la IP y el puerto de DroidCam")
    rot_btn = Button("", "ghost", "sm", "rotate-cw")
    h.addWidget(combo)
    h.addWidget(edit_btn)
    h.addWidget(rot_btn)
    h.addWidget(reload_btn)
    devices = QMediaDevices(w)  # se mantiene vivo con el widget

    def populate() -> None:
        s = ctx.settings
        names = [d.description() for d in QMediaDevices.videoInputs()]
        current = s.camera_index
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(f"DroidCam · {s.camera_ip}:{s.camera_port}", "ip")
        for i, n in enumerate(names):
            combo.addItem(f"{n}", i)
        if current >= len(names):  # índice configurado que Windows no lista (p. ej. desconectada)
            combo.addItem(f"Cámara {current} (no listada)", current)
        if not names and current < 1:
            combo.addItem("Cámara 0", 0)
        combo.setCurrentIndex(0 if s.camera_source == "ip" else max(0, combo.findData(current)))
        edit_btn.setVisible(s.camera_source == "ip")
        from ...config import camera_rotation
        rot_btn.setToolTip(f"Girar la imagen 90° (ahora {camera_rotation(s)}°). Útil con el celular en vertical.")
        combo.blockSignals(False)

    def changed(_i: int) -> None:
        data = combo.currentData()
        s = ctx.settings
        if data is None:
            return
        if data == "ip":
            if s.camera_source == "ip":
                return
            new = replace(s, camera_source="ip")
        else:
            if s.camera_source == "device" and data == s.camera_index:
                return
            new = replace(s, camera_source="device", camera_index=int(data))
        ctx.save_settings(new)  # reinicia la cámara
        ctx.toast("Cámara cambiada", combo.currentText(), "info")

    def edit_ip() -> None:
        s = ctx.settings
        text, ok = QInputDialog.getText(w, "DroidCam por IP", "IP y puerto del celular (como muestra DroidCam):",
                                        QLineEdit.Normal, f"{s.camera_ip}:{s.camera_port}")
        if not ok or not text.strip():
            return
        host, _, port = text.strip().replace("http://", "").split("/")[0].partition(":")
        try:
            new = replace(s, camera_source="ip", camera_ip=host.strip(), camera_port=int(port or s.camera_port))
        except ValueError:
            ctx.toast("Puerto inválido", text, "warn")
            return
        ctx.save_settings(new)
        ctx.toast("Cámara IP actualizada", f"{new.camera_ip}:{new.camera_port}", "info")

    def rotate() -> None:
        from ...config import camera_rotation, with_rotation
        new = with_rotation(ctx.settings, camera_rotation(ctx.settings) + 90)
        ctx.save_settings(new)  # gira en caliente, sin reabrir la cámara
        ctx.toast("Imagen girada", f"{camera_rotation(new)}° en sentido horario", "info")

    def reload() -> None:
        populate()
        ctx.camera.restart()
        ctx.toast("Cámara recargada", combo.currentText(), "info")

    combo.currentIndexChanged.connect(changed)
    reload_btn.clicked.connect(reload)
    edit_btn.clicked.connect(edit_ip)
    rot_btn.clicked.connect(rotate)
    devices.videoInputsChanged.connect(populate)
    ctx.settingsChanged.connect(populate)
    w.destroyed.connect(lambda *_: _safe_disconnect(ctx.settingsChanged, populate))
    populate()
    w.combo, w.reload_btn, w.rot_btn, w.populate = combo, reload_btn, rot_btn, populate
    combo.setFocusPolicy(Qt.StrongFocus)
    return w


def _safe_disconnect(sig, slot) -> None:
    try:
        sig.disconnect(slot)
    except (RuntimeError, TypeError):
        pass
