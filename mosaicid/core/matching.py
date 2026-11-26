"""Motor B — matching por distancia entre embeddings (sin entrenamiento).

Cada persona conserva VARIAS referencias (no se promedian al guardar).
La similitud es coseno (equivalente a FaceRecognizerSF.match con FR_COSINE).
La estrategia de agregación por persona es intercambiable:

- best     : mejor referencia individual (v1, por defecto)
- centroid : similitud contra el centroide normalizado de la persona
- topk     : promedio de las k mejores referencias

Búsqueda brute-force: correcta y suficiente para decenas de personas.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embeddings import as_embedding, l2_normalize


@dataclass
class DistanceResult:
    person_id: int | None  # None = desconocido
    candidate_id: int | None  # mejor candidato aunque no supere el umbral
    similarity: float
    threshold: float
    ranking: list[tuple[int, float]]  # (person_id, similitud) ordenado desc

    @property
    def known(self) -> bool:
        return self.person_id is not None


def _aggregate(sims: np.ndarray, labels: np.ndarray, person_ids: np.ndarray, strategy: str, k: int,
               refs: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.empty(len(person_ids), dtype=np.float32)
    for i, pid in enumerate(person_ids):
        mask = labels == pid
        if strategy == "centroid":
            c = l2_normalize(refs[mask].mean(axis=0))
            out[i] = float(np.dot(c, q))
        elif strategy == "topk":
            s = np.sort(sims[mask])[::-1][: max(1, k)]
            out[i] = float(s.mean())
        else:
            out[i] = float(sims[mask].max())
    return out


class DistanceMatcher:
    def __init__(self, embeddings: np.ndarray | None = None, labels: np.ndarray | None = None,
                 threshold: float = 0.363, strategy: str = "best", top_k: int = 3):
        self.threshold = float(threshold)
        self.strategy = strategy
        self.top_k = top_k
        if embeddings is None or len(embeddings) == 0:
            self.refs = np.zeros((0, 128), dtype=np.float32)
            self.labels = np.zeros((0,), dtype=np.int64)
        else:
            self.refs = l2_normalize(np.asarray(embeddings, dtype=np.float32))
            self.labels = np.asarray(labels, dtype=np.int64)
        self.person_ids = np.unique(self.labels)

    @property
    def empty(self) -> bool:
        return len(self.refs) == 0

    def similarities(self, query: np.ndarray) -> list[tuple[int, float]]:
        if self.empty:
            return []
        q = l2_normalize(as_embedding(query))
        sims = self.refs @ q
        agg = _aggregate(sims, self.labels, self.person_ids, self.strategy, self.top_k, self.refs, q)
        order = np.argsort(-agg)
        return [(int(self.person_ids[i]), float(agg[i])) for i in order]

    def identify(self, query: np.ndarray) -> DistanceResult:
        ranking = self.similarities(query)
        if not ranking:
            return DistanceResult(None, None, 0.0, self.threshold, [])
        pid, sim = ranking[0]
        return DistanceResult(pid if sim >= self.threshold else None, pid, sim, self.threshold, ranking)
