"""Embeddings SFace: 128 valores float32. Serialización a BLOB sin pickle."""
from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 128
EMBEDDING_DTYPE = np.float32
EMBEDDING_BYTES = EMBEDDING_DIM * np.dtype(EMBEDDING_DTYPE).itemsize  # 512


class EmbeddingError(ValueError):
    pass


def as_embedding(x) -> np.ndarray:
    """Valida y devuelve un vector (128,) float32 contiguo."""
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.shape != (EMBEDDING_DIM,):
        raise EmbeddingError(f"Embedding con forma {a.shape}; se esperaba ({EMBEDDING_DIM},)")
    if not np.issubdtype(a.dtype, np.floating):
        raise EmbeddingError(f"Embedding con dtype {a.dtype}; se esperaba float")
    a = np.ascontiguousarray(a, dtype=EMBEDDING_DTYPE)
    if not np.all(np.isfinite(a)):
        raise EmbeddingError("Embedding con valores no finitos")
    return a


def serialize_embedding(x) -> bytes:
    return as_embedding(x).tobytes(order="C")


def deserialize_embedding(blob: bytes) -> np.ndarray:
    if blob is None or len(blob) != EMBEDDING_BYTES:
        raise EmbeddingError(f"BLOB de {0 if blob is None else len(blob)} bytes; se esperaban {EMBEDDING_BYTES}")
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).copy()


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Equivalente a FaceRecognizerSF.match(..., FR_COSINE)."""
    return float(np.dot(l2_normalize(a), l2_normalize(b)))
