import json
import os
import time

import numpy as np
import pytest

from conftest import person_cluster
from mosaicid.config import Settings, load_settings, save_settings
from mosaicid.core.embeddings import (EmbeddingError, cosine_similarity, deserialize_embedding, l2_normalize,
                                      serialize_embedding)
from mosaicid.core.gallery import plan_index, search_faces, source_from_path, thumbnail_path
from mosaicid.core.matching import DistanceMatcher
from mosaicid.core.svm import SvmEngine, TrainingError, build_dataset, can_train
from mosaicid.core.tracker import Tracker


# ── serialización de embeddings ──────────────────────────────────────────
def test_embedding_roundtrip(rng):
    e = rng.normal(size=128).astype(np.float32)
    b = serialize_embedding(e)
    assert len(b) == 512
    assert np.array_equal(deserialize_embedding(b), e)


def test_embedding_accepts_opencv_shape(rng):
    e = rng.normal(size=(1, 128)).astype(np.float32)
    assert deserialize_embedding(serialize_embedding(e)).shape == (128,)


@pytest.mark.parametrize("bad", [np.zeros(127, np.float32), np.zeros((2, 128), np.float32), np.zeros(128, np.int32),
                                 np.full(128, np.nan, np.float32)])
def test_embedding_validation(bad):
    with pytest.raises(EmbeddingError):
        serialize_embedding(bad)


def test_deserialize_wrong_length():
    with pytest.raises(EmbeddingError):
        deserialize_embedding(b"\x00" * 100)


def test_float64_is_converted(rng):
    e = rng.normal(size=128)
    assert deserialize_embedding(serialize_embedding(e)).dtype == np.float32


# ── matching por distancia ───────────────────────────────────────────────
def _two_people(rng):
    a, b = rng.normal(size=128), rng.normal(size=128)
    X = np.vstack([person_cluster(rng, a, 8), person_cluster(rng, b, 8)])
    y = np.array([1] * 8 + [2] * 8)
    return a, b, X, y


def test_distance_identifies_known(rng):
    a, b, X, y = _two_people(rng)
    m = DistanceMatcher(X, y, threshold=0.363)
    r = m.identify(a + rng.normal(0, 0.15, 128))
    assert r.person_id == 1 and r.similarity > 0.9
    assert [p for p, _ in r.ranking] == [1, 2]


def test_distance_threshold_unknown(rng):
    _, _, X, y = _two_people(rng)
    m = DistanceMatcher(X, y, threshold=0.363)
    r = m.identify(rng.normal(size=128))  # vector aleatorio ~ ortogonal
    assert r.person_id is None and r.candidate_id in (1, 2)
    assert r.similarity < 0.363


def test_distance_threshold_boundary():
    ref = np.zeros(128, np.float32)
    ref[0] = 1
    m = DistanceMatcher(ref[None], np.array([7]), threshold=0.5)
    q = np.zeros(128, np.float32)
    q[0], q[1] = 0.5, np.sqrt(0.75)  # coseno = 0.5 exacto
    assert m.identify(q).person_id == 7
    m.threshold = 0.5001
    assert m.identify(q).person_id is None


@pytest.mark.parametrize("strategy", ["best", "centroid", "topk"])
def test_distance_strategies(rng, strategy):
    a, _, X, y = _two_people(rng)
    m = DistanceMatcher(X, y, threshold=0.3, strategy=strategy, top_k=3)
    assert m.identify(a).person_id == 1


def test_distance_empty():
    r = DistanceMatcher().identify(np.ones(128, np.float32))
    assert r.person_id is None and r.ranking == []


def test_cosine_matches_opencv(rng, models_dir):
    cv2 = pytest.importorskip("cv2")
    from mosaicid.core.pipeline import FacePipeline
    if not (models_dir / "face_recognition_sface_2021dec.onnx").exists():
        pytest.skip("modelos no disponibles")
    p = FacePipeline(models_dir / "face_detection_yunet_2023mar.onnx", models_dir / "face_recognition_sface_2021dec.onnx")
    a, b = rng.normal(size=128).astype(np.float32), rng.normal(size=128).astype(np.float32)
    assert abs(p.match(a, b) - cosine_similarity(a, b)) < 1e-5


# ── SVM ──────────────────────────────────────────────────────────────────
def _ds(rng, people=2, n=8):
    centers = [rng.normal(size=128) for _ in range(people)]
    X = np.vstack([person_cluster(rng, c, n) for c in centers])
    y = np.repeat(np.arange(1, people + 1), n)
    return centers, build_dataset(X, y, np.arange(1, len(y) + 1))


def test_dataset_building(rng):
    _, ds = _ds(rng, 3, 8)
    assert ds.X.shape == (24, 128)
    assert np.allclose(np.linalg.norm(ds.X, axis=1), 1, atol=1e-5)
    assert ds.counts() == {1: 8, 2: 8, 3: 8}
    assert ds.classes == [1, 2, 3]


def test_cannot_train_single_person(rng):
    _, ds = _ds(rng, 1, 8)
    ok, _ = can_train(ds)
    assert not ok


def test_svm_dirty_state(tmp_path, rng):
    centers, ds = _ds(rng, 2, 8)
    eng = SvmEngine(tmp_path / "m.joblib", tmp_path / "m.json")
    assert not eng.trained and eng.is_dirty(ds)
    meta = eng.train(ds, C=1.0, kernel="linear")
    assert not eng.is_dirty(ds) and meta["classes"] == [1, 2]
    assert meta["evaluation"]["method"] is not None
    # nueva persona → dirty; solo la nueva figura pendiente
    X3 = np.vstack([ds.X, person_cluster(rng, rng.normal(size=128), 6)])
    y3 = np.concatenate([ds.y, np.full(6, 3)])
    ds3 = build_dataset(X3, y3, np.arange(1, len(y3) + 1))
    assert eng.is_dirty(ds3)
    assert eng.pending_person_ids(ds3) == {3}
    # persistencia: recargar mantiene el estado
    eng2 = SvmEngine(tmp_path / "m.joblib", tmp_path / "m.json")
    assert eng2.trained and not eng2.is_dirty(ds)


def test_svm_rejection_threshold(tmp_path, rng):
    centers, ds = _ds(rng, 3, 8)
    eng = SvmEngine(tmp_path / "m.joblib", tmp_path / "m.json")
    eng.train(ds)
    p = eng.predict(centers[0], margin=0.0)
    assert p.candidate_id == 1 and p.person_id == 1
    assert abs(sum(v for _, v in p.probabilities) - 1) < 1e-6
    p2 = eng.predict(centers[0], margin=1.01)  # umbral imposible → desconocido
    assert p2.person_id is None and p2.candidate_id == 1


def test_svm_training_error(tmp_path, rng):
    _, ds = _ds(rng, 1, 8)
    with pytest.raises(TrainingError):
        SvmEngine(tmp_path / "m.joblib", tmp_path / "m.json").train(ds)


# ── base de datos ────────────────────────────────────────────────────────
def test_database_people_and_samples(db, rng):
    assert db.schema_version == 1
    pid = db.create_person("Luis Saini")
    with pytest.raises(Exception):
        db.create_person("luis saini")  # único sin distinguir mayúsculas
    embs = rng.normal(size=(8, 128)).astype(np.float32)
    db.add_samples(pid, [dict(embedding=e, kind="Frontal", quality_score=0.9) for e in embs])
    X, y, sid = db.all_embeddings()
    assert X.shape == (8, 128) and set(y) == {pid} and np.allclose(X, embs)
    assert db.list_people()[0].sample_count == 8
    db.delete_person(pid)
    assert db.all_embeddings()[0].shape == (0, 128)  # cascada


def test_database_photos(db, rng):
    fid = db.add_folder("C:/fotos", "real")
    meta = dict(path="C:/fotos/a.jpg", filename="a.jpg", mtime=1.0, size=10, source_type="real")
    pid = db.upsert_photo(fid, meta, [dict(x=1, y=2, w=3, h=4, score=0.9, embedding=rng.normal(size=128))])
    assert db.gallery_counts() == (1, 1)
    db.set_photo_source(pid, "test")
    db.upsert_photo(fid, meta, [])  # reindexar no pisa la clasificación manual
    assert db.get_photo(pid).source_type == "test"
    assert db.face_matrix(only_real=True)[0].shape[0] == 0
    db.mark_missing([pid], delete=False)
    assert db.list_photos() == [] and len(db.list_photos(include_missing=True)) == 1
    with pytest.raises(ValueError):
        db.set_photo_source(pid, "fake")


# ── estado del índice de imágenes ────────────────────────────────────────
def test_image_index_state(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"22")
    st = a.stat()
    known = {str(a): (st.st_mtime, st.st_size, "ok", 1), str(tmp_path / "gone.jpg"): (1.0, 1, "ok", 9)}
    plan = plan_index([a, b], known)
    assert plan.to_index == [b] and plan.unchanged == 1 and plan.missing_ids == [9]
    time.sleep(0.01)
    a.write_bytes(b"333")  # cambió → reindexar
    os.utime(a, (st.st_mtime + 5, st.st_mtime + 5))
    assert a in plan_index([a], known).to_index


def test_thumbnail_key_changes_with_file(tmp_path):
    p = tmp_path / "x.jpg"
    assert thumbnail_path(tmp_path, p, 1.0, 10) != thumbnail_path(tmp_path, p, 2.0, 10)


def test_source_from_folder(tmp_path):
    root = tmp_path
    assert source_from_path(root / "real" / "2019" / "a.jpg", root, "unknown") == "real"
    assert source_from_path(root / "viaje" / "a.jpg", root, "test") == "test"
    assert source_from_path(root / "Generated" / "a.jpg", root, "real") == "generated"


def test_search_faces(rng):
    q = rng.normal(size=128)
    F = np.vstack([q + rng.normal(0, 0.1, 128), rng.normal(size=128), q + rng.normal(0, 0.1, 128)]).astype(np.float32)
    hits = search_faces(q, F, np.array([10, 11, 12]), np.array([1, 1, 2]), threshold=0.363)
    assert {h.photo_id for h in hits} == {1, 2}
    assert all(h.face_id in (10, 12) for h in hits)


# ── settings ─────────────────────────────────────────────────────────────
def test_settings_roundtrip_and_validation(tmp_path):
    p = tmp_path / "settings.json"
    s = load_settings(p)  # crea con defaults
    assert p.exists() and s.distance_cosine_threshold == 0.363 and s.svm_rejection_threshold == 0.40
    s.camera_index = 2
    save_settings(p, s)
    assert load_settings(p).camera_index == 2
    raw = json.loads(p.read_text())
    raw.update(distance_strategy="nope", width="abc", unknown_key=1, svm_rejection_threshold=3)
    p.write_text(json.dumps(raw))
    s2 = load_settings(p)
    assert s2.distance_strategy == "best" and s2.width == Settings().width and s2.svm_rejection_threshold == 1.0


# ── tracker ──────────────────────────────────────────────────────────────
def test_tracker_vote_stability():
    t = Tracker(history=5)
    box = (100, 100, 80, 80)
    for pid in [1, 1, 1, None, 2, 1]:
        tr = t.update([box])[0]
        tr.add("distance", pid, 0.7)
    assert tr.label("distance")[0] == 1  # un frame aislado no cambia el nombre
    tr2 = t.update([(105, 102, 80, 80), (400, 100, 80, 80)])
    assert tr2[0].id == tr.id and tr2[1].id != tr.id


def test_svm_threshold_is_relative_to_chance():
    from mosaicid.core.svm import effective_threshold
    assert effective_threshold(0.0, 2) == pytest.approx(0.5)      # margen 0 = azar
    assert effective_threshold(1.0, 6) == pytest.approx(1.0)
    assert effective_threshold(0.4, 2) == pytest.approx(0.70)     # con 2 clases exige más probabilidad…
    assert effective_threshold(0.4, 6) == pytest.approx(0.50)     # …que con 6
    assert effective_threshold(0.4, 1) == pytest.approx(0.4)


def test_camera_source_settings():
    from mosaicid.config import camera_label, camera_url
    s = Settings()
    assert s.camera_source == "ip" and camera_url(s) == "http://192.168.0.244:4747/video"
    assert camera_label(s) == "DroidCam · 192.168.0.244:4747"
    s.camera_source, s.camera_port, s.camera_ip = "otra", 99999, "  "
    s.validate()
    assert s.camera_source == "device" and s.camera_port == 65535 and s.camera_ip == "192.168.0.244"


def test_camera_rotation_per_source():
    from mosaicid.config import Settings, camera_rotation, with_rotation

    s = Settings(camera_source="ip")
    assert camera_rotation(s) == 0
    s = with_rotation(s, 90)
    assert camera_rotation(s) == 90 and s.camera_rotation_device == 0  # la webcam no se gira
    s = with_rotation(s, camera_rotation(s) + 270)
    assert camera_rotation(s) == 0
    assert Settings(camera_rotation_ip=450).validate().camera_rotation_ip == 90
