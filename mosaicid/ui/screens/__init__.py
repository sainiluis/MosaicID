"""Factory de pantallas (rutas → clases). Se crean bajo demanda y se reutilizan."""
from __future__ import annotations

import importlib

from PySide6.QtWidgets import QWidget

ROUTES = {
    "home": ("home", "HomeScreen"),
    "recognition": ("recognition", "RecognitionScreen"),
    "people": ("people", "PeopleScreen"),
    "person": ("people", "PersonDetailScreen"),
    "enroll": ("enroll", "EnrollScreen"),
    "training": ("training", "TrainingScreen"),
    "gallery": ("gallery", "GalleryScreen"),
    "gallery_search": ("gallery_search", "GallerySearchScreen"),
    "gallery_results": ("gallery", "GalleryResultsScreen"),
    "photo": ("photo_viewer", "PhotoViewerScreen"),
    "compare": ("compare", "CompareScreen"),
    "how": ("how", "HowItWorksScreen"),
    "settings": ("settings_screen", "SettingsScreen"),
}


def create_screen(key: str, ctx) -> QWidget:
    mod, cls = ROUTES[key]
    m = importlib.import_module(f"{__name__}.{mod}")
    return getattr(m, cls)(ctx)
