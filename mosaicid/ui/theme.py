"""Tokens de diseño — traducción directa de tailwind.config.js del prototipo React."""
from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase

C = {
    "canvas": "#FAFAF9",
    "rail": "#F2F3F3",
    "sunken": "#EDEEEE",
    "surface": "#FFFFFF",
    "line": "rgba(20,32,43,0.09)",
    "line_strong": "rgba(20,32,43,0.16)",
    "ink": "#16212B",
    "ink2": "#46525E",
    "ink3": "#6A7580",
    "ink4": "#97A0A8",
    "accent": "#1F5A6E",
    "accent_hover": "#194B5C",
    "accent_press": "#143E4C",
    "accent_soft": "#E7EFF1",
    "ok": "#2E7D57",
    "ok_soft": "#E6F2EC",
    "warn": "#A86B12",
    "warn_soft": "#F8EEDC",
    "err": "#B23A30",
    "err_soft": "#F7E5E3",
    "stage": "#111417",
    "box_known": "#9BEBC3",
    "box_unknown": "#F5CB74",
}

LINE = QColor(20, 32, 43, 23)
LINE_STRONG = QColor(20, 32, 43, 41)


def qc(name: str) -> QColor:
    v = C[name]
    if v.startswith("rgba"):
        r, g, b, a = [float(x) for x in v[5:-1].split(",")]
        return QColor(int(r), int(g), int(b), int(a * 255))
    return QColor(v)


FAMILY = "Segoe UI"


def load_fonts(fonts_dir) -> str:
    """Registra Inter (local). Si falla, se usa Segoe UI."""
    global FAMILY
    fams = set()
    for f in sorted(fonts_dir.glob("Inter-*.ttf")):
        fid = QFontDatabase.addApplicationFont(str(f))
        if fid >= 0:
            fams.update(QFontDatabase.applicationFontFamilies(fid))
    if fams:
        FAMILY = "Inter" if "Inter" in fams else sorted(fams)[0]
    return FAMILY


def font(px: float, weight: int = 400) -> QFont:
    f = QFont(FAMILY)
    f.setPixelSize(int(round(px)))
    f.setWeight(QFont.Weight(weight))
    f.setHintingPreference(QFont.PreferNoHinting)
    return f


# Escala tipográfica (px): title 28, section 18, lead 15, body 14, small 13, meta 12
TYPE = {"title": 28, "section": 18, "lead": 15, "body": 14, "small": 13, "meta": 12, "over": 11}


def stylesheet() -> str:
    c = C
    return f"""
    * {{ font-family: "{FAMILY}"; font-size: 14px; color: {c['ink']}; }}
    QMainWindow, #Main {{ background: {c['canvas']}; }}
    QWidget#Page, QWidget#PageBody {{ background: {c['canvas']}; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QToolTip {{ background: #1B252E; color: white; border: none; padding: 4px 6px; border-radius: 4px; font-size: 12px; }}

    QLabel {{ background: transparent; }}
    QLabel[role="title"] {{ font-size: 28px; font-weight: 600; letter-spacing: -0.4px; }}
    QLabel[role="section"] {{ font-size: 18px; font-weight: 600; }}
    QLabel[role="lead"] {{ font-size: 15px; color: {c['ink2']}; }}
    QLabel[role="body"] {{ font-size: 14px; }}
    QLabel[role="muted"] {{ font-size: 14px; color: {c['ink3']}; }}
    QLabel[role="small"] {{ font-size: 13px; color: {c['ink3']}; }}
    QLabel[role="small2"] {{ font-size: 13px; color: {c['ink2']}; }}
    QLabel[role="meta"] {{ font-size: 12px; color: {c['ink3']}; }}
    QLabel[role="over"] {{ font-size: 11px; font-weight: 500; color: {c['ink3']}; letter-spacing: 0.6px; }}
    QLabel[role="mono"] {{ font-family: Consolas, "Cascadia Mono", monospace; font-size: 12px; color: {c['ink2']}; }}
    QLabel[tone="ok"] {{ color: {c['ok']}; }}
    QLabel[tone="warn"] {{ color: {c['warn']}; }}
    QLabel[tone="err"] {{ color: {c['err']}; }}
    QLabel[tone="accent"] {{ color: {c['accent']}; }}
    QLabel[link="true"] {{ color: {c['accent']}; font-weight: 500; }}
    QLabel[link="true"]:hover {{ text-decoration: underline; }}

    /* Botones */
    QPushButton {{ border: none; border-radius: 6px; padding: 0 12px; min-height: 32px; font-weight: 500; }}
    QPushButton[variant="primary"] {{ background: {c['accent']}; color: white; }}
    QPushButton[variant="primary"]:hover {{ background: {c['accent_hover']}; }}
    QPushButton[variant="primary"]:pressed {{ background: {c['accent_press']}; }}
    QPushButton[variant="primary"]:disabled {{ background: #9DB5BE; color: rgba(255,255,255,0.85); }}
    QPushButton[variant="secondary"] {{ background: {c['surface']}; border: 1px solid rgba(20,32,43,0.13); color: {c['ink']}; }}
    QPushButton[variant="secondary"]:hover {{ background: #F6F7F7; border-color: rgba(20,32,43,0.2); }}
    QPushButton[variant="secondary"]:pressed {{ background: {c['sunken']}; }}
    QPushButton[variant="secondary"]:disabled {{ color: {c['ink4']}; }}
    QPushButton[variant="ghost"] {{ background: transparent; color: {c['ink2']}; }}
    QPushButton[variant="ghost"]:hover {{ background: rgba(0,0,0,0.045); color: {c['ink']}; }}
    QPushButton[variant="ghost"]:pressed {{ background: rgba(0,0,0,0.07); }}
    QPushButton[variant="ghost"]:disabled {{ color: {c['ink4']}; }}
    QPushButton[variant="quiet"] {{ background: transparent; color: {c['accent']}; padding: 0 6px; }}
    QPushButton[variant="quiet"]:hover {{ background: {c['accent_soft']}; }}
    QPushButton[variant="danger"] {{ background: transparent; color: {c['err']}; }}
    QPushButton[variant="danger"]:hover {{ background: {c['err_soft']}; }}
    QPushButton[size="sm"] {{ min-height: 28px; padding: 0 10px; font-size: 13px; border-radius: 5px; }}
    QPushButton[size="lg"] {{ min-height: 36px; padding: 0 16px; border-radius: 7px; }}
    QPushButton:focus {{ outline: none; }}
    QPushButton[variant="dark"] {{ background: transparent; color: rgba(255,255,255,0.75); }}
    QPushButton[variant="dark"]:hover {{ background: rgba(255,255,255,0.1); color: white; }}
    QPushButton[variant="dark"]:disabled {{ color: rgba(255,255,255,0.3); }}

    /* Inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {c['surface']}; border: 1px solid rgba(20,32,43,0.14); border-radius: 6px;
        min-height: 32px; padding: 0 10px; selection-background-color: rgba(31,90,110,0.2); selection-color: {c['ink']};
    }}
    QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{ border-color: rgba(20,32,43,0.24); }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {c['accent']}; }}
    QLineEdit[size="lg"] {{ min-height: 36px; font-size: 15px; }}
    QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: none; }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox::down-arrow {{ image: url(__CHEVRON__); width: 14px; height: 14px; }}
    QComboBox QAbstractItemView {{ background: white; border: 1px solid rgba(20,32,43,0.12); selection-background-color: {c['accent_soft']}; selection-color: {c['ink']}; outline: none; padding: 4px; }}

    QSlider::groove:horizontal {{ height: 3px; background: {c['sunken']}; border-radius: 1px; }}
    QSlider::sub-page:horizontal {{ background: {c['accent']}; border-radius: 1px; }}
    QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: white; border: 1px solid rgba(20,32,43,0.25); }}
    QSlider::handle:horizontal:hover {{ border-color: {c['accent']}; }}

    QProgressBar {{ background: {c['sunken']}; border: none; border-radius: 1px; max-height: 3px; min-height: 3px; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 1px; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: rgba(20,32,43,0.16); border-radius: 3px; min-height: 30px; margin: 2px 3px; }}
    QScrollBar::handle:vertical:hover {{ background: rgba(20,32,43,0.28); }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ height: 0; }}

    QMessageBox {{ background: white; }}
    QMessageBox QLabel {{ font-size: 14px; }}
    QMenu {{ background: white; border: 1px solid rgba(20,32,43,0.12); padding: 4px; }}
    QMenu::item {{ padding: 6px 14px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {c['accent_soft']}; color: {c['ink']}; }}
    """
