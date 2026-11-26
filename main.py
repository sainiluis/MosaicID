"""Punto de entrada de MosaicID (aplicación de escritorio local)."""
from __future__ import annotations

import logging
import os
import sys
import traceback

# Sin telemetría ni servicios externos: Qt no necesita red para nada de esto.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")  # DirectShow abre más rápido en Windows


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication, QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from mosaicid import APP_NAME
    from mosaicid.logging_setup import log_versions, setup_logging
    from mosaicid.paths import Paths

    paths = Paths().ensure()
    setup_logging(paths.logs)
    log = logging.getLogger("mosaicid")
    log.info("Inicio de %s (raíz %s)", APP_NAME, paths.root)
    log_versions()

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    from mosaicid.ui import theme, widgets

    widgets.ICONS_DIR = paths.assets / "icons"
    theme.load_fonts(paths.assets / "fonts")
    qss = theme.stylesheet().replace("__CHEVRON__", str(paths.assets / "icons" / "chevron-down.svg").replace("\\", "/"))
    app.setStyleSheet(qss)
    app.setFont(theme.font(14))
    icon_path = paths.assets / "mosaicid.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    def excepthook(t, v, tb):
        log.error("Excepción no controlada:\n%s", "".join(traceback.format_exception(t, v, tb)))

    sys.excepthook = excepthook

    try:
        from mosaicid.app_context import AppContext
        from mosaicid.ui.screens import create_screen
        from mosaicid.ui.shell import MainWindow

        ctx = AppContext(paths)
        win = MainWindow(ctx, create_screen)
    except Exception as e:
        log.exception("No se pudo iniciar la aplicación")
        QMessageBox.critical(None, APP_NAME, f"No se pudo iniciar MosaicID:\n{e}\n\nRevisá logs/mosaicid.log")
        return 1
    win.show()
    from PySide6.QtCore import QTimer
    from mosaicid.ui.screens.gallery import index_included_gallery
    QTimer.singleShot(300, lambda: index_included_gallery(ctx))
    code = app.exec()
    log.info("Cierre de %s", APP_NAME)
    return code


if __name__ == "__main__":
    sys.exit(main())
