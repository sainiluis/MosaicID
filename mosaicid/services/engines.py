"""Instantánea inmutable de los motores de identificación.

El analizador en vivo y las pantallas leen una `EngineSnapshot`; cuando cambian las
personas, el SVM o la configuración se construye una nueva y se reemplaza de forma
atómica (asignación de referencia). Nunca se modifica una instantánea en uso.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..core.matching import DistanceMatcher, DistanceResult
from ..core.svm import Dataset, SvmEngine, SvmPrediction


@dataclass
class FaceDecision:
    """Ambos motores sobre el MISMO embedding."""

    svm: SvmPrediction | None  # None si no hay modelo entrenado
    distance: DistanceResult


@dataclass
class EngineSnapshot:
    matcher: DistanceMatcher
    svm: SvmEngine
    svm_threshold: float
    names: dict[int, str] = field(default_factory=dict)
    svm_dirty: bool = False
    svm_pending: set[int] = field(default_factory=set)
    dataset: Dataset | None = None

    def decide(self, embedding: np.ndarray) -> FaceDecision:
        svm = self.svm.predict(embedding, self.svm_threshold) if self.svm.trained else None
        if svm is not None and svm.person_id is not None and svm.person_id not in self.names:
            svm.person_id = None  # clase de una persona ya eliminada: no se informa como identidad
        return FaceDecision(svm, self.matcher.identify(embedding))

    def name(self, pid: int | None, unknown: str = "Desconocido") -> str:
        if pid is None:
            return unknown
        return self.names.get(pid, f"#{pid}")

    def short(self, pid: int | None, unknown: str = "Desconocido") -> str:
        return self.name(pid, unknown).split(" ")[0]


def build_snapshot(X, y, sids, names: dict[int, str], svm: SvmEngine, s: Settings) -> EngineSnapshot:
    from ..core.svm import build_dataset

    ds = build_dataset(X, y, sids)
    matcher = DistanceMatcher(X, y, threshold=s.distance_cosine_threshold, strategy=s.distance_strategy,
                              top_k=s.distance_top_k)
    return EngineSnapshot(matcher, svm, s.svm_rejection_threshold, dict(names), svm.is_dirty(ds),
                          svm.pending_person_ids(ds) if len(ds.y) else set(), ds)
