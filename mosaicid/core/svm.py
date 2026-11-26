"""Motor A — clasificador supervisado SVM sobre embeddings SFace.

- Dataset: embedding (L2-normalizado) → id de persona. Se reconstruye completo en cada entrenamiento.
- SVC(C, kernel, probability=True). Persistencia con joblib + metadata JSON.
- Closed-set: el SVM siempre elige una clase conocida. Para informar "Desconocido" se usa
  predict_proba() y un umbral de rechazo (svm_rejection_threshold). La probabilidad del SVM
  NO es una similitud coseno y nunca se mezcla con ella.
- Estado DIRTY: si el conjunto de muestras actual difiere del usado para entrenar, el
  clasificador queda pendiente de reentrenamiento.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from .. import PIPELINE_VERSION
from .embeddings import as_embedding, l2_normalize

log = logging.getLogger(__name__)


@dataclass
class Dataset:
    X: np.ndarray  # (n, 128) normalizado
    y: np.ndarray  # (n,) person_id
    sample_ids: np.ndarray

    @property
    def classes(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.y))

    def counts(self) -> dict[int, int]:
        u, c = np.unique(self.y, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}

    def fingerprint(self) -> str:
        return dataset_fingerprint(self.y, self.sample_ids)


def dataset_fingerprint(person_ids: np.ndarray, sample_ids: np.ndarray) -> str:
    pairs = sorted(zip((int(s) for s in sample_ids), (int(p) for p in person_ids)))
    return hashlib.sha1(json.dumps(pairs).encode()).hexdigest()


def build_dataset(embeddings: np.ndarray, person_ids: np.ndarray, sample_ids: np.ndarray) -> Dataset:
    X = l2_normalize(np.asarray(embeddings, np.float32)) if len(embeddings) else np.zeros((0, 128), np.float32)
    return Dataset(X, np.asarray(person_ids, np.int64), np.asarray(sample_ids, np.int64))


@dataclass
class SvmPrediction:
    person_id: int | None  # None = desconocido (por rechazo)
    candidate_id: int | None
    probability: float  # probabilidad de la clase candidata (predict_proba)
    threshold: float
    probabilities: list[tuple[int, float]] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.person_id is not None


class TrainingError(RuntimeError):
    pass


def can_train(ds: Dataset) -> tuple[bool, str]:
    counts = ds.counts()
    if len(counts) < 2:
        return False, "Se necesitan al menos 2 personas registradas para entrenar un clasificador."
    if min(counts.values()) < 3:
        return False, "Cada persona necesita al menos 3 muestras."
    return True, ""


class SvmEngine:
    def __init__(self, model_path: Path, meta_path: Path):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.model = None
        self.meta: dict = {}
        self.load()

    # ── persistencia ─────────────────────────────────────────────────────
    def load(self) -> None:
        self.model, self.meta = None, {}
        if not (self.model_path.exists() and self.meta_path.exists()):
            return
        try:
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if self.meta.get("pipeline_version") != PIPELINE_VERSION:
                log.warning("Modelo SVM de otra versión de pipeline (%s); se ignora", self.meta.get("pipeline_version"))
                self.meta = {}
                return
            self.model = joblib.load(self.model_path)
        except Exception as e:  # archivo corrupto → sin modelo, no rompe la app
            log.error("No se pudo cargar el modelo SVM: %s", e)
            self.model, self.meta = None, {}

    @property
    def trained(self) -> bool:
        return self.model is not None

    @property
    def trained_person_ids(self) -> set[int]:
        return {int(p) for p in self.meta.get("classes", [])}

    def is_dirty(self, ds: Dataset) -> bool:
        """True si el dataset actual no coincide con el usado en el último entrenamiento."""
        if not self.trained:
            return len(ds.y) > 0
        return self.meta.get("dataset_fingerprint") != ds.fingerprint()

    def pending_person_ids(self, ds: Dataset) -> set[int]:
        """Personas cuyas muestras actuales no son exactamente las entrenadas."""
        trained = {int(k): set(v) for k, v in self.meta.get("sample_ids_by_person", {}).items()}
        out = set()
        for pid in ds.classes:
            current = set(int(s) for s in ds.sample_ids[ds.y == pid])
            if trained.get(pid) != current:
                out.add(pid)
        return out

    # ── entrenamiento ────────────────────────────────────────────────────
    def train(self, ds: Dataset, C: float = 1.0, kernel: str = "linear", progress=None) -> dict:
        """Entrena, evalúa (si hay datos suficientes) y guarda. `progress(stage)` informa la etapa."""
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.svm import SVC

        ok, why = can_train(ds)
        if not ok:
            raise TrainingError(why)
        t0 = time.perf_counter()
        progress and progress("prepare")
        counts = ds.counts()

        progress and progress("train")
        with warnings.catch_warnings():
            # sklearn 1.9 depreca probability=True (se elimina en 1.11); versión fijada en requirements.txt
            warnings.simplefilter("ignore", FutureWarning)
            model = SVC(C=C, kernel=kernel, probability=True, random_state=0)
            model.fit(ds.X, ds.y)

        progress and progress("evaluate")
        min_count = min(counts.values())
        evaluation: dict
        if min_count >= 3:
            k = min(5, min_count)
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
            scores = cross_val_score(SVC(C=C, kernel=kernel), ds.X, ds.y, cv=cv)
            evaluation = {
                "method": f"Validación cruzada estratificada ({k} particiones)",
                "accuracy_mean": float(scores.mean()),
                "accuracy_std": float(scores.std()),
                "note": "Las muestras de cada persona provienen de una misma sesión de captura: "
                        "la precisión es optimista y no mide desempeño sobre fotos nuevas.",
            }
        else:
            evaluation = {"method": None, "note": "No hay suficientes muestras por persona para validar."}

        progress and progress("save")
        sample_ids_by_person = {str(pid): [int(s) for s in ds.sample_ids[ds.y == pid]] for pid in ds.classes}
        meta = {
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "pipeline_version": PIPELINE_VERSION,
            "classes": ds.classes,
            "samples": int(len(ds.y)),
            "samples_per_person": {str(k): v for k, v in counts.items()},
            "sample_ids_by_person": sample_ids_by_person,
            "dataset_fingerprint": ds.fingerprint(),
            "params": {"C": C, "kernel": kernel, "probability": True},
            "evaluation": evaluation,
            "duration_s": round(time.perf_counter() - t0, 3),
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".tmp")
        joblib.dump(model, tmp)
        tmp.replace(self.model_path)
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        self.model, self.meta = model, meta
        log.info("SVM entrenado: %d clases, %d muestras, %.2fs", len(ds.classes), len(ds.y), meta["duration_s"])
        return meta

    # ── predicción ───────────────────────────────────────────────────────
    @property
    def n_classes(self) -> int:
        return len(self.model.classes_) if self.model is not None else 0

    def predict(self, embedding: np.ndarray, margin: float) -> SvmPrediction | None:
        """Rechazo por `predict_proba` con umbral relativo al azar (ver `effective_threshold`)."""
        if self.model is None:
            return None
        q = l2_normalize(as_embedding(embedding)).reshape(1, -1)
        proba = self.model.predict_proba(q)[0]
        classes = [int(c) for c in self.model.classes_]
        order = np.argsort(-proba)
        probs = [(classes[i], float(proba[i])) for i in order]
        cand, p = probs[0]
        thr = effective_threshold(margin, len(classes))
        return SvmPrediction(cand if p >= thr else None, cand, p, thr, probs)


def effective_threshold(margin: float, n_classes: int) -> float:
    """Umbral de probabilidad según cuántas clases conoce el modelo.

    Con K clases la probabilidad "al azar" es 1/K, así que una probabilidad de 0,60 significa cosas muy
    distintas con 2 personas (apenas sobre el azar) o con 6 (muy por encima). El umbral se define como un
    margen sobre el azar:  t = 1/K + margen · (1 − 1/K).  margen 0 → acepta todo; margen 1 → exige 100 %.
    Calibrado con fotos reales: calibration/CALIBRACION.md.
    """
    if n_classes <= 1:
        return float(margin)
    base = 1.0 / n_classes
    return float(base + margin * (1.0 - base))
