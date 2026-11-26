"""Recorrido automático de la interfaz (sin pantalla) que guarda una captura de cada vista.

Uso:
    set MOSAICID_HOME=<raíz con datos>   (p. ej. la que deja `tools/e2e_real.py --keep`)
    set MOSAICID_FAKE_CAMERA=<imagen>     (opcional; sin ella usa la webcam configurada)
    python tools/gui_smoke.py <carpeta_capturas> [--routes home,recognition,...]

Sirve para verificar que cada pantalla abre, pinta y no deja excepciones en logs/mosaicid.log.
Las capturas pueden contener caras: quedan solo en la carpeta indicada.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT = "home,recognition:compare,recognition:svm,compare,people,training,gallery,gallery_search,results,photo,how,settings"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--routes", default=DEFAULT)
    ap.add_argument("--wait", type=float, default=2.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from mosaicid.logging_setup import setup_logging
    from mosaicid.paths import Paths
    from mosaicid.ui import theme, widgets

    paths = Paths().ensure()
    setup_logging(paths.logs)
    app = QApplication([])
    widgets.ICONS_DIR = paths.assets / "icons"
    theme.load_fonts(paths.assets / "fonts")
    app.setStyleSheet(theme.stylesheet().replace("__CHEVRON__", str(paths.assets / "icons" / "chevron-down.svg").replace("\\", "/")))
    app.setFont(theme.font(14))
    errors: list[str] = []
    sys.excepthook = lambda t, v, tb: errors.append(f"{t.__name__}: {v}")

    from mosaicid.app_context import AppContext
    from mosaicid.ui.screens import create_screen
    from mosaicid.ui.shell import MainWindow

    ctx = AppContext(paths)
    win = MainWindow(ctx, create_screen)
    win.resize(1440, 900)
    win.show()

    def pump(sec: float) -> None:
        t = time.time()
        while time.time() - t < sec:
            app.processEvents()
            time.sleep(0.01)

    people = ctx.db.list_people()
    pid = people[0].id if people else None
    photos = ctx.db.list_photos()
    pump(1.0)
    for i, route in enumerate(a.routes.split(",")):
        key, _, arg = route.partition(":")
        try:
            if key == "recognition":
                win.go("recognition")
                scr = win.screens["recognition"]
                scr._set_engine(arg or "distance")
            elif key == "results":
                if pid is None:
                    continue
                win.go("gallery_results", person_id=pid)
            elif key == "photo":
                if not photos:
                    continue
                win.go("photo", photo_id=photos[0].id)
            elif key == "person":
                if pid is None:
                    continue
                win.go("person", person_id=pid)
            else:
                win.go(key)
            pump(a.wait if key in ("recognition", "compare", "enroll") else 1.2)
            name = f"{i:02d}_{key}{'_' + arg if arg else ''}.png"
            win.grab().save(str(a.out / name))
            strays = [w for w in app.topLevelWidgets() if w.isVisible() and w is not win and w.isWindow()
                      and type(w).__name__ not in ("QMenu", "QToolTip", "Toast")]
            for w in strays:
                txt = getattr(w, "text", lambda: "")()
                errors.append(f"{route}: widget suelto como ventana: {type(w).__name__} {txt!r}")
            print(f"[OK ] {route:<20} → {name}")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{route}: {type(e).__name__}: {e}")
            print(f"[FALLA] {route}: {e}")
    win.go("home")
    pump(1.0)
    ctx.shutdown()
    log = (paths.logs / "mosaicid.log").read_text(encoding="utf-8", errors="replace")
    tb = log.count("Traceback")
    print(f"\nExcepciones capturadas: {len(errors)} · Tracebacks en el log: {tb}")
    for e in errors:
        print("  ", e)
    return 1 if errors or tb else 0


if __name__ == "__main__":
    sys.exit(main())
