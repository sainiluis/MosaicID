import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mosaicid.db import Database  # noqa: E402


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    yield d
    d.close()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def person_cluster(rng, center, n, noise=0.15):
    """Embeddings sintéticos alrededor de un centro (solo para tests de lógica)."""
    return (center + rng.normal(0, noise, size=(n, 128))).astype(np.float32)


@pytest.fixture
def models_dir():
    return ROOT / "models"
