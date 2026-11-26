"""Estabilización temporal simple para el video en vivo.

Estrategia (deliberadamente pequeña):
1. Asociación por IoU: cada detección nueva se asigna al track existente con mayor
   superposición (IoU ≥ 0.3), de forma greedy. Si no hay, se crea un track nuevo.
2. Historial corto por track (N últimos resultados, configurable) para CADA motor.
3. Votación: la identidad mostrada es la mayoritaria del historial si alcanza
   mayoría simple; si no, se mantiene la anterior (histéresis) → el nombre no parpadea.
4. La métrica mostrada es el promedio de la métrica de ese motor en los votos ganadores.
5. Un track sin detecciones durante `max_missed` inferencias se descarta.

El embedding y los resultados crudos del último frame se conservan para Comparar.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Vote:
    person_id: int | None
    metric: float


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    history: int
    votes: dict[str, deque] = field(default_factory=dict)
    stable: dict[str, int | None] = field(default_factory=dict)
    missed: int = 0
    age: int = 0
    last: object = None  # último FaceResult crudo

    def add(self, engine: str, person_id: int | None, metric: float) -> None:
        dq = self.votes.setdefault(engine, deque(maxlen=self.history))
        dq.append(Vote(person_id, metric))
        counts = Counter(v.person_id for v in dq)
        winner, n = counts.most_common(1)[0]
        if engine not in self.stable or n * 2 > len(dq):
            self.stable[engine] = winner

    def label(self, engine: str) -> tuple[int | None, float]:
        dq = self.votes.get(engine)
        if not dq:
            return None, 0.0
        pid = self.stable.get(engine)
        vals = [v.metric for v in dq if v.person_id == pid] or [dq[-1].metric]
        return pid, float(np.mean(vals))


class Tracker:
    def __init__(self, history: int = 7, iou_threshold: float = 0.3, max_missed: int = 4):
        self.history = history
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, boxes: list[tuple[float, float, float, float]]) -> list[Track]:
        """Asocia cajas a tracks. Devuelve un track por caja (mismo orden)."""
        pairs = sorted(((iou(t.box, b), ti, bi) for ti, t in enumerate(self.tracks) for bi, b in enumerate(boxes)),
                       reverse=True)
        used_t, assigned = set(), {}
        for score, ti, bi in pairs:
            if score < self.iou_threshold or ti in used_t or bi in assigned:
                continue
            used_t.add(ti)
            assigned[bi] = self.tracks[ti]
        out = []
        for bi, b in enumerate(boxes):
            t = assigned.get(bi)
            if t is None:
                t = Track(self._next, b, self.history)
                self._next += 1
                self.tracks.append(t)
            t.box, t.missed = b, 0
            t.age += 1
            out.append(t)
        live = {id(t) for t in out}
        for t in self.tracks:
            if id(t) not in live:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return out
