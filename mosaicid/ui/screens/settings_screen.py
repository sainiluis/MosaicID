"""Configuración — inspector de parámetros (config/settings.json)."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QMessageBox, QSlider, QSpinBox, QVBoxLayout,
                               QWidget)

from ...app_context import AppContext
from ...config import Settings
from ...services.camera import probe_cameras
from ...services.workers import keep_until_finished
from ..camera_view import camera_name
from ..theme import C
from ..widgets import Button, Divider, Page, PageHeader, Row, Segmented, clear_layout, label

RESOLUTIONS = [(1280, 720), (1920, 1080), (640, 480)]


class _Probe(QThread):
    found = Signal(list)

    def run(self):
        self.found.emit(probe_cameras(5))


class Field(QWidget):
    def __init__(self, title: str, hint: str, control: QWidget):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 8, 0, 8)
        h.setSpacing(32)
        v = QVBoxLayout()
        v.setSpacing(1)
        v.addWidget(label(title, "body"))
        if hint:
            v.addWidget(label(hint, "small", wrap=True))
        h.addLayout(v, 1)
        control.setFixedWidth(260)
        h.addWidget(control, 0, Qt.AlignVCenter)


def slider(value: float, lo: float, hi: float, step: float, decimals: int) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)
    s = QSlider(Qt.Horizontal)
    n = int(round((hi - lo) / step))
    s.setRange(0, n)
    s.setValue(int(round((value - lo) / step)))
    val = label("", "body")
    val.setFixedWidth(46)
    val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    w.get = lambda: round(lo + s.value() * step, decimals)
    s.valueChanged.connect(lambda _: val.setText(f"{w.get():.{decimals}f}"))
    val.setText(f"{w.get():.{decimals}f}")
    h.addWidget(s, 1)
    h.addWidget(val)
    w.slider = s
    return w


class SettingsScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        reset = Button("Restablecer valores", "ghost", "sm")
        reset.clicked.connect(lambda: self._load(Settings()))
        self.save_btn = Button("Guardar", "primary")
        self.save_btn.clicked.connect(self._save)
        self.add(PageHeader("Configuración", "Parámetros de cámara, detección, calidad y de los dos motores.",
                            [reset, self.save_btn]))
        self.form = QVBoxLayout()
        self.form.setSpacing(0)
        self.add(self.form)
        self.stretch()
        self._probe = None

    def _section(self, title: str, desc: str) -> None:
        if self.form.count():
            self.form.addSpacing(18)
            self.form.addWidget(Divider())
        self.form.addSpacing(22)
        t = label(title, "body")
        t.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.form.addWidget(t)
        self.form.addWidget(label(desc, "small"))
        self.form.addSpacing(6)

    def _load(self, s: Settings) -> None:
        clear_layout(self.form)
        self.w = {}
        self._section("Cámara", "Fuente de video para reconocimiento, registro y escaneo.")
        src = Segmented([("ip", "DroidCam por IP"), ("device", "Cámara de Windows")], s.camera_source, small=True)
        self.w["camera_source"] = lambda: src.value
        self.form.addWidget(Field("Fuente", "Por IP la app se conecta directo al celular (sin el cliente de DroidCam "
                                            "en la PC). DroidCam admite una sola conexión a la vez.", src))
        ip = QLineEdit(s.camera_ip)
        ip.setPlaceholderText("192.168.0.244")
        ip.setMaximumWidth(220)
        self.w["camera_ip"] = lambda: ip.text().strip()
        self.form.addWidget(Field("IP del celular", "La que muestra la app DroidCam («WiFi IP»).", ip))
        port = QSpinBox()
        port.setRange(1, 65535)
        port.setValue(s.camera_port)
        self.w["camera_port"] = port.value
        self.form.addWidget(Field("Puerto", "DroidCam usa 4747.", port))
        rots = [("0", "0°"), ("90", "90°"), ("180", "180°"), ("270", "270°")]
        rot_ip = Segmented(rots, str(s.camera_rotation_ip), small=True)
        self.w["camera_rotation_ip"] = lambda: int(rot_ip.value)
        self.form.addWidget(Field("Giro de la imagen (IP)", "Sentido horario. Con el celular en vertical suele ser 90°. "
                                  "También se gira con el botón ⟳ junto al selector de cámara.", rot_ip))
        cam = QComboBox()
        for i in range(6):
            cam.addItem(f"{i} · {camera_name(i)}", i)
        cam.setCurrentIndex(min(5, s.camera_index))
        self.w["camera_index"] = lambda: cam.currentData()
        probe = Button("Detectar cámaras", "ghost", "sm", "refresh-cw")
        probe.clicked.connect(lambda: self._run_probe(cam, probe))
        cw = QWidget()
        ch = QHBoxLayout(cw)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.addWidget(cam, 1)
        self.form.addWidget(Field("Dispositivo", "Para la fuente «Cámara de Windows»: índice de DirectShow.", cw))
        rot_dev = Segmented(rots, str(s.camera_rotation_device), small=True)
        self.w["camera_rotation_device"] = lambda: int(rot_dev.value)
        self.form.addWidget(Field("Giro de la imagen (Windows)", "Sentido horario.", rot_dev))
        self.form.addWidget(probe, 0, Qt.AlignRight)
        res = QComboBox()
        for w_, h_ in RESOLUTIONS:
            res.addItem(f"{w_} × {h_}", (w_, h_))
        idx = next((i for i, r in enumerate(RESOLUTIONS) if r == (s.width, s.height)), 0)
        res.setCurrentIndex(idx)
        self.w["res"] = lambda: res.currentData()
        self.form.addWidget(Field("Resolución pedida", "La cámara puede entregar otra; se informa la real.", res))
        fps = QSpinBox()
        fps.setRange(5, 60)
        fps.setValue(s.fps)
        self.w["fps"] = fps.value
        self.form.addWidget(Field("FPS deseados", "", fps))

        self._section("Detección", "YuNet: qué cuenta como rostro.")
        self._slider("yunet_score_threshold", "Puntaje mínimo", "Score threshold de YuNet.", s.yunet_score_threshold, 0.5, 0.99, 0.01, 2)
        self._slider("yunet_nms_threshold", "Supresión de no máximos", "NMS threshold.", s.yunet_nms_threshold, 0.1, 0.6, 0.05, 2)
        self._spin("min_face_size", "Tamaño mínimo de rostro (registro)", "Lado menor de la caja, en píxeles.", s.min_face_size, 40, 400)

        self._section("Calidad de muestra", "Controles del onboarding y del escaneo.")
        self._dspin("blur_threshold", "Nitidez mínima", "Varianza del Laplaciano sobre la cara alineada.", s.blur_threshold, 5, 1000, 5)
        self._dspin("min_brightness", "Brillo mínimo", "Media de gris (0–255) en el centro del rostro.", s.min_brightness, 0, 255, 5)
        self._dspin("max_brightness", "Brillo máximo", "", s.max_brightness, 0, 255, 5)
        self._spin("enrollment_sample_count", "Muestras por persona", "", s.enrollment_sample_count, 3, 12)

        self._section("Distancia", "Comparación coseno contra referencias almacenadas.")
        self._slider("distance_cosine_threshold", "Umbral de similitud", "Recomendado para SFace: 0,363.", s.distance_cosine_threshold, 0.2, 0.8, 0.001, 3)
        strat = Segmented([("best", "Mejor"), ("centroid", "Centroide"), ("topk", "Top-k")], s.distance_strategy, small=True)
        self.w["distance_strategy"] = lambda: strat.value
        self.form.addWidget(Field("Agregación por persona", "Mejor referencia individual (v1), centroide o promedio top-k.", strat))
        self._spin("distance_top_k", "k (para Top-k)", "", s.distance_top_k, 1, 12)

        self._section("Clasificador", "SVM sobre embeddings normalizados. Los cambios aplican al reentrenar.")
        kern = Segmented([("linear", "Lineal"), ("rbf", "RBF")], s.svm_kernel, small=True)
        self.w["svm_kernel"] = lambda: kern.value
        self.form.addWidget(Field("Kernel", "", kern))
        self._dspin("svm_c", "Regularización (C)", "", s.svm_c, 0.01, 100, 0.1)
        self._slider("svm_rejection_threshold", "Margen de rechazo",
                     "Margen sobre el azar (1/K). Umbral = 1/K + margen·(1−1/K); por debajo se informa Desconocido.",
                     s.svm_rejection_threshold, 0.0, 0.95, 0.01, 2)

        self._section("Rendimiento y estabilidad", "El video se muestra a la tasa de la cámara; la inferencia puede ir más lenta.")
        self._spin("inference_every_n_frames", "Inferencia cada N cuadros", "", s.inference_every_n_frames, 1, 6)
        self._spin("tracker_history", "Historial de votación", "Resultados recientes por rostro para estabilizar el nombre.",
                   s.tracker_history, 1, 15)

        self._section("Galería", "Indexación y búsqueda por rostro.")
        self._slider("gallery_match_threshold", "Umbral de búsqueda", "Similitud coseno mínima para considerar una coincidencia.",
                     s.gallery_match_threshold, 0.2, 0.8, 0.001, 3)
        self._spin("gallery_max_side", "Lado máximo para detectar", "Las fotos más grandes se reducen solo para detectar.",
                   s.gallery_max_side, 640, 4096)
        pol = Segmented([("mark", "Marcar faltantes"), ("delete", "Eliminar del índice")], s.gallery_missing_policy, small=True)
        self.w["gallery_missing_policy"] = lambda: pol.value
        self.form.addWidget(Field("Archivos borrados", "Qué hacer con fotos indexadas que ya no están en disco.", pol))
        self.form.addSpacing(8)
        self.form.addWidget(label("Carpetas indexadas", "small2"))
        for r in self.ctx.db.list_folders():
            row = Row(40, clickable=False)
            row.h.addWidget(label(r["path"], "mono"), 1)
            row.h.addWidget(label(r["source_type"], "small"))
            rm = Button("Quitar", "danger", "sm")
            rm.clicked.connect(lambda _=False, fid=r["id"], p=r["path"]: self._remove_folder(fid, p))
            row.h.addWidget(rm)
            self.form.addWidget(row)
        self.form.addSpacing(16)
        self.form.addWidget(label(f"Archivo: {self.ctx.paths.config_file}", "meta"))

    def _slider(self, key, title, hint, value, lo, hi, step, dec):
        w = slider(value, lo, hi, step, dec)
        self.w[key] = w.get
        self.form.addWidget(Field(title, hint, w))

    def _spin(self, key, title, hint, value, lo, hi):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(value))
        self.w[key] = s.value
        self.form.addWidget(Field(title, hint, s))

    def _dspin(self, key, title, hint, value, lo, hi, step):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setValue(float(value))
        self.w[key] = s.value
        self.form.addWidget(Field(title, hint, s))

    def _run_probe(self, combo: QComboBox, btn: Button) -> None:
        btn.set_loading(True, "Buscando…")
        self._probe = _Probe()
        keep_until_finished(self._probe)

        def found(idx):
            btn.set_loading(False, "Detectar cámaras")
            for i in range(combo.count()):
                ok = combo.itemData(i) in idx
                combo.setItemText(i, f"{combo.itemData(i)} · {camera_name(combo.itemData(i))}" + ("" if ok else " (no disponible)"))
            self.ctx.toast("Cámaras detectadas", ", ".join(str(i) for i in idx) or "Ninguna", "info")
        self._probe.found.connect(found)
        self._probe.start()

    def _remove_folder(self, fid: int, path: str) -> None:
        r = QMessageBox.question(self, "Quitar carpeta", f"¿Quitar {path} del índice?\n\nLas fotos originales no se tocan; "
                                 "solo se borran sus registros y rostros indexados.", QMessageBox.Yes | QMessageBox.Cancel)
        if r == QMessageBox.Yes:
            self.ctx.db.remove_folder(fid)
            self.ctx.galleryChanged.emit()
            self._load(self.ctx.settings)

    def _save(self) -> None:
        vals = {k: f() for k, f in self.w.items()}
        w_, h_ = vals.pop("res")
        s = replace(self.ctx.settings, width=w_, height=h_, **vals)
        self.ctx.save_settings(s)
        self.ctx.toast("Configuración guardada", "")

    def on_enter(self, **_):
        self._load(self.ctx.settings)
