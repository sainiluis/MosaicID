"""Regresión: widgets quitados de un contenedor visible no deben abrirse como ventanas sueltas.

Qt agenda mostrar cada widget agregado a un contenedor visible; si se le quitaba el padre antes del siguiente
ciclo de eventos (Reconocimiento rehace la lista en cada cuadro), aparecían ventanitas con etiquetas sueltas.

Corre en un proceso aparte: otros tests crean una QCoreApplication y en el mismo proceso no se puede crear
después la QApplication que necesitan los widgets.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from mosaicid.ui.widgets import clear_layout, discard
app = QApplication([])
main = QWidget(); lay = QVBoxLayout(main); main.show(); app.processEvents()
for _ in range(5):  # varios "cuadros" seguidos sin pasar por el ciclo de eventos
    inner = QVBoxLayout(); inner.addWidget(QLabel("anidada")); lay.addLayout(inner)
    lay.addWidget(QLabel("Umbral 0.363 · último cuadro"))
    clear_layout(lay)
row = QLabel("fila"); lay.addWidget(row); discard(row)
app.processEvents(); app.processEvents()
strays = [type(w).__name__ for w in app.topLevelWidgets() if w.isVisible() and w is not main]
print("STRAYS", strays)
sys.exit(1 if strays else 0)
'''


def test_removed_widgets_never_become_windows():
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
