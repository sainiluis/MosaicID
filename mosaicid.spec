# -*- mode: python ; coding: utf-8 -*-
# PyInstaller (onedir). Uso: build_exe.bat
# Recursos de solo lectura (models/, assets/) van dentro del bundle; data/, config/, cache/ y logs/
# se crean junto al .exe en la primera ejecución (ver mosaicid/paths.py).
block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[],
    datas=[('models/*.onnx', 'models'), ('assets', 'assets')],
    hiddenimports=['sklearn.svm', 'sklearn.model_selection', 'sklearn.utils._typedefs', 'sklearn.neighbors._partition_nodes',
                   'mosaicid.ui.screens.home', 'mosaicid.ui.screens.recognition', 'mosaicid.ui.screens.people',
                   'mosaicid.ui.screens.enroll', 'mosaicid.ui.screens.training', 'mosaicid.ui.screens.gallery',
                   'mosaicid.ui.screens.gallery_search', 'mosaicid.ui.screens.photo_viewer', 'mosaicid.ui.screens.compare',
                   'mosaicid.ui.screens.how', 'mosaicid.ui.screens.settings_screen'],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.Qt3DCore', 'PySide6.QtQuick3D', 'PySide6.QtPdf'],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MosaicID', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=False, icon='assets/mosaicid.ico')
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='MosaicID')
