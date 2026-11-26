"""Cómo funciona — pipeline académico y la diferencia entre los dos motores."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ...app_context import AppContext
from ..pipeline_widget import Pipeline
from ..theme import C
from ..widgets import Divider, LinkLabel, Page, PageHeader, icon_label, label, overline


def engine_block(icon_name: str, name: str, quote: str, steps: list[str], outcome: str, tone: str, note: str,
                 left: bool) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0 if left else 40, 0, 40 if left else 0, 0)
    v.setSpacing(0)
    h = QHBoxLayout()
    h.setSpacing(8)
    h.addWidget(icon_label(icon_name, C["accent"], 18))
    t = label(name, "body")
    t.setStyleSheet("font-size: 15px; font-weight: 600;")
    h.addWidget(t)
    h.addStretch(1)
    v.addLayout(h)
    v.addSpacing(6)
    v.addWidget(label(f"“{quote}”", "lead"))
    v.addSpacing(20)
    v.addWidget(label("Nueva persona", "meta"))
    v.addSpacing(6)
    for i, s in enumerate(steps):
        r = QHBoxLayout()
        r.setSpacing(10)
        n = label(str(i + 1), "small")
        n.setStyleSheet(f"color: {C['ink4']};")
        n.setFixedWidth(14)
        r.addWidget(n)
        r.addWidget(label(s, "small2"))
        r.addStretch(1)
        v.addLayout(r)
        v.addSpacing(6)
    o = label(outcome, "body", tone)
    o.setStyleSheet("font-weight: 600; padding-left: 24px;")
    v.addWidget(o)
    v.addSpacing(14)
    v.addWidget(label(note, "small", wrap=True))
    v.addStretch(1)
    return w


class HowItWorksScreen(Page):
    def __init__(self, ctx: AppContext):
        super().__init__()
        link = LinkLabel("Ver en vivo →", "small")
        link.setStyleSheet(f"color: {C['accent']}; font-weight: 500;")
        link.clicked.connect(lambda: ctx.navigate("compare"))
        self.add(PageHeader("Cómo funciona", "Del cuadro de la cámara a una identidad, en cinco etapas.", [link]))
        self.add(Divider())
        self.add(36)
        self.add(Pipeline([
            ("scan-face", "Detección", "YuNet encuentra cada rostro: caja y puntaje."),
            ("fingerprint", "Landmarks", "Ojos, nariz y comisuras: 5 puntos de referencia."),
            ("crosshair", "Alineamiento", "SFace.alignCrop() endereza y recorta a 112 × 112."),
            ("binary", "Embedding", "SFace.feature() lo resume en 128 valores."),
            ("user-check", "Identificación", "Un motor decide a quién pertenece el vector."),
        ], large=True))
        self.add(28)
        self.add(Divider())
        self.add(36)
        row = QHBoxLayout()
        row.setSpacing(48)
        intro = QVBoxLayout()
        intro.setSpacing(0)
        intro.addWidget(overline("Mismo embedding"))
        intro.addSpacing(4)
        t = label("Dos decisiones", "section")
        t.setStyleSheet("font-size: 22px;")
        intro.addWidget(t)
        intro.addSpacing(8)
        intro.addWidget(label("Cada persona se registra una sola vez. Sus muestras generan embeddings que alimentan "
                              "a los dos motores: son las referencias del motor por distancia y el dataset del SVM.",
                              "small2", wrap=True))
        intro.addSpacing(12)
        intro.addWidget(label("El pipeline se ejecuta una sola vez por rostro; solo cambia la decisión final.",
                              "small", wrap=True))
        intro.addStretch(1)
        iw = QWidget()
        iw.setLayout(intro)
        iw.setFixedWidth(260)
        row.addWidget(iw)
        two = QHBoxLayout()
        two.setSpacing(0)
        two.addWidget(engine_block("brain-circuit", "Clasificador SVM", "Aprende fronteras entre identidades.",
                                   ["Agregar muestras", "Generar embeddings", "Actualizar dataset", "Reentrenar SVM"],
                                   "Requiere reentrenar", "warn",
                                   "Closed-set: siempre elige una clase entrenada. Para informar «Desconocido» se "
                                   "aplica un umbral sobre predict_proba(), relativo al azar (1/K): con pocas personas "
                                   "el SVM rechaza peor a los desconocidos.", True), 1)
        two.addWidget(Divider(vertical=True))
        two.addWidget(engine_block("ruler", "Distancia", "Compara contra referencias almacenadas.",
                                   ["Capturar muestras", "Generar embeddings", "Guardar referencias"],
                                   "Disponible inmediatamente", "ok",
                                   "Similitud coseno (FR_COSINE) contra cada referencia. La mejor coincidencia sobre "
                                   "el umbral (0,363 por defecto) define la identidad.", False), 1)
        row.addLayout(two, 1)
        self.add(row)
        self.stretch()
