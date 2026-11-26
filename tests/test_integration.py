"""Integración con los modelos ONNX reales sobre las fotos SIMULADAS del prototipo.

No son fotos reales del grupo: sirven para verificar que el pipeline, ambos motores, la
persistencia y la galería funcionan de punta a punta. La calibración con fotos reales es la
fase 10.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from mosaicid.config import Settings
from mosaicid.core.gallery import search_faces
from mosaicid.core.indexer import index_folder
from mosaicid.core.matching import DistanceMatcher
from mosaicid.core.pipeline import FacePipeline
from mosaicid.core.quality import check_sample
from mosaicid.core.svm import SvmEngine, build_dataset

ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT.parent / "public" / "photos"
MODELS = ROOT / "models"

pytestmark = pytest.mark.skipif(not (MODELS / "face_recognition_sface_2021dec.onnx").exists() or not PHOTOS.exists(),
                                reason="modelos o fotos simuladas no disponibles")


@pytest.fixture(scope="module")
def pipe():
    return FacePipeline(MODELS / "face_detection_yunet_2023mar.onnx", MODELS / "face_recognition_sface_2021dec.onnx")


def read(name):
    return cv2.imread(str(PHOTOS / name))


def variants(img):
    """Pequeñas variaciones (brillo, escala, rotación leve) para simular 8 capturas."""
    out = []
    h, w = img.shape[:2]
    for i, (a, b, ang, sc) in enumerate([(1, 0, 0, 1), (1.1, 5, -4, 1), (0.9, -5, 4, 1), (1, 10, 0, 0.9),
                                         (1, 0, 2, 1.1), (1.05, -8, -2, 0.95), (0.95, 8, 0, 1.05), (1, 0, -6, 1)]):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        out.append(cv2.convertScaleAbs(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT), alpha=a, beta=b))
    return out


def test_pipeline_real_models(pipe):
    faces = pipe.process(read("pedro.jpg"), max_side=1200)
    assert len(faces) == 1
    f = faces[0]
    assert f.aligned.shape == (112, 112, 3)
    assert f.embedding.shape == (128,) and f.embedding.dtype == np.float32
    assert f.detection.landmarks.shape == (5, 2) and f.detection.score > 0.9


def test_multiple_faces(pipe):
    assert len(pipe.process(read("cafe.jpg"))) >= 2


def test_quality_rejects_two_faces(pipe):
    img = read("cafe.jpg")
    rep, _ = check_sample(pipe.process(img), img.shape, Settings())
    assert not rep.ok and rep.code == "multiple"


def test_engines_end_to_end(pipe, db, tmp_path):
    s = Settings()
    people = {}
    for name, file in (("Pedro", "pedro.jpg"), ("Agustín", "agustin.jpg")):
        pid = db.create_person(name)
        embs = [pipe.process(v, max_side=1200)[0].embedding for v in variants(read(file))]
        db.add_samples(pid, [dict(embedding=e, kind="test") for e in embs])
        people[name] = pid
    X, y, sids = db.all_embeddings()
    m = DistanceMatcher(X, y, s.distance_cosine_threshold)
    # Distancia: la misma persona en otra variante se reconoce; otra cara no
    q_pedro = pipe.process(cv2.flip(read("pedro.jpg"), 1), max_side=1200)[0].embedding
    r = m.identify(q_pedro)
    assert r.person_id == people["Pedro"] and r.similarity > s.distance_cosine_threshold
    other = pipe.process(read("p23.jpg"), max_side=1200)
    assert other, "la foto de control debería tener un rostro"
    assert m.identify(other[0].embedding).person_id is None
    # SVM: mismo dataset, mismos embeddings
    eng = SvmEngine(tmp_path / "svm.joblib", tmp_path / "svm.json")
    ds = build_dataset(X, y, sids)
    eng.train(ds)
    p = eng.predict(q_pedro, s.svm_rejection_threshold)
    assert p.candidate_id == people["Pedro"]
    # tercera persona → dataset cambia → SVM pendiente, Distancia ya la reconoce
    pid3 = db.create_person("Sofía")
    db.add_samples(pid3, [dict(embedding=e.embedding) for e in [pipe.process(v, max_side=1200)[0] for v in variants(read("p23.jpg"))]])
    X3, y3, s3 = db.all_embeddings()
    assert eng.is_dirty(build_dataset(X3, y3, s3))
    assert DistanceMatcher(X3, y3, s.distance_cosine_threshold).identify(other[0].embedding).person_id == pid3


def test_gallery_index_incremental_and_search(pipe, db, tmp_path):
    import shutil
    folder = tmp_path / "fotos" / "synthetic"
    folder.mkdir(parents=True)
    for n in ("pedro.jpg", "agustin.jpg", "cafe.jpg", "p23.jpg"):
        shutil.copy(PHOTOS / n, folder / n)
    fid = db.add_folder(str(tmp_path / "fotos"), "unknown")
    s = Settings()
    rep = index_folder(db, pipe, fid, s, tmp_path / "thumbs")
    assert rep.indexed == 4 and not rep.errors
    assert all(p.source_type == "synthetic" for p in db.list_photos())  # por nombre de subcarpeta
    assert len(list((tmp_path / "thumbs").rglob("*.jpg"))) == 4
    rep2 = index_folder(db, pipe, fid, s, tmp_path / "thumbs")
    assert rep2.indexed == 0 and rep2.unchanged == 4  # no reprocesa
    (folder / "p23.jpg").unlink()
    rep3 = index_folder(db, pipe, fid, s, tmp_path / "thumbs")
    assert rep3.missing == 1 and len(db.list_photos()) == 3
    # búsqueda por rostro: Pedro aparece en su foto
    q = pipe.process(cv2.flip(read("pedro.jpg"), 1), max_side=1200)[0].embedding
    F, fids, pids = db.face_matrix()
    hits = search_faces(q, F, fids, pids, s.gallery_match_threshold)
    found = {db.get_photo(h.photo_id).filename for h in hits}
    assert "pedro.jpg" in found and "agustin.jpg" not in found
    assert db.face_matrix(only_real=True)[0].shape[0] == 0  # nada marcado como real


def test_rotated_face_detected_only_with_rotation_pass(pipe):
    """Una selfie inclinada ~40°: sin la pasada rotada YuNet no la ve; con ella sí, con la caja en su lugar
    y un embedding equivalente al de la cara derecha (alignCrop corrige el giro)."""
    img = read("pedro.jpg")
    upright = pipe.process(img, max_side=1600, multiscale=True)
    assert len(upright) == 1
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 40, 1.0)
    tilted = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    without = pipe.process(tilted, max_side=1600, multiscale=True)
    with_rot = pipe.process(tilted, max_side=1600, multiscale=True, rotations=(-30.0, 30.0))
    assert len(without) == 0 and len(with_rot) == 1
    # el rostro de Pedro aparece en la imagen inclinada, en la posición rotada de la caja original
    cx, cy = upright[0].detection.x + upright[0].detection.w / 2, upright[0].detection.y + upright[0].detection.h / 2
    ex, ey = m @ np.array([cx, cy, 1.0])
    best = max(with_rot, key=lambda f: float(f.embedding @ upright[0].embedding))
    bx, by = best.detection.x + best.detection.w / 2, best.detection.y + best.detection.h / 2
    assert abs(bx - ex) < upright[0].detection.w * 0.35 and abs(by - ey) < upright[0].detection.h * 0.35
    assert pipe.match(best.embedding, upright[0].embedding) > 0.6


def test_detector_change_forces_full_reindex(pipe, db, tmp_path):
    import shutil
    folder = tmp_path / "fotos"
    folder.mkdir()
    for n in ("pedro.jpg", "cafe.jpg"):
        shutil.copy(PHOTOS / n, folder / n)
    fid = db.add_folder(str(folder), "test")
    s = Settings()
    assert index_folder(db, pipe, fid, s, tmp_path / "t").indexed == 2
    assert index_folder(db, pipe, fid, s, tmp_path / "t").indexed == 0
    s.gallery_rotated_detection = False  # cambia la detección → se reprocesa todo
    rep = index_folder(db, pipe, fid, s, tmp_path / "t")
    assert rep.indexed == 2 and rep.missing == 0
    assert index_folder(db, pipe, fid, s, tmp_path / "t").indexed == 0


def test_included_gallery_registered_once(db, tmp_path):
    """La carpeta galeria_incluida/ se registra sola una vez; si se quita no vuelve; si la app se mueve, se actualiza."""
    from types import SimpleNamespace

    from mosaicid.paths import Paths
    from mosaicid.ui.screens.gallery import included_gallery_folder

    paths = Paths(tmp_path / "app")
    ctx = SimpleNamespace(paths=paths, db=db)
    assert included_gallery_folder(ctx) is None  # sin carpeta
    paths.included_gallery.mkdir(parents=True)
    cv2.imwrite(str(paths.included_gallery / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    fid = included_gallery_folder(ctx)
    assert fid is not None and db.list_folders()[0]["source_type"] == "real"
    assert included_gallery_folder(ctx) == fid  # todavía sin indexar → se reintenta
    moved = Paths(tmp_path / "otra")
    moved.included_gallery.mkdir(parents=True)
    cv2.imwrite(str(moved.included_gallery / "a.jpg"), np.zeros((20, 20, 3), np.uint8))
    assert included_gallery_folder(SimpleNamespace(paths=moved, db=db)) == fid
    assert Path(db.list_folders()[0]["path"]) == moved.included_gallery
    db.remove_folder(fid)
    assert included_gallery_folder(ctx) is None and not db.list_folders()


def test_thread_survives_dropping_last_reference(qapp=None):
    """Soltar la última referencia a un QThread que todavía corre no debe abortar el proceso."""
    import gc
    import time

    from PySide6.QtCore import QCoreApplication, QThread
    from mosaicid.services import workers

    app = QCoreApplication.instance() or QCoreApplication([])

    class Slow(QThread):
        def run(self):
            time.sleep(0.3)

    t = Slow()
    workers.keep_until_finished(t)
    t.start()
    del t
    gc.collect()
    assert len(workers._retired) == 1
    end = time.time() + 3
    while workers._retired and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    assert not workers._retired


def test_camera_frames_are_rotated_live(tmp_path, monkeypatch):
    """La rotación se aplica a cada frame (lo que ven detector y registro) y cambia sin reabrir la cámara."""
    import time

    from PySide6.QtCore import QCoreApplication
    from mosaicid.config import Settings
    from mosaicid.services.camera import CameraService

    img = np.zeros((120, 200, 3), np.uint8)
    path = tmp_path / "f.png"
    cv2.imwrite(str(path), img)
    monkeypatch.setenv("MOSAICID_FAKE_CAMERA", str(path))
    app = QCoreApplication.instance() or QCoreApplication([])
    s = Settings(camera_source="device", width=200, height=120, camera_rotation_device=90)
    cam = CameraService(lambda: s)
    shapes = []
    cam.frameReady.connect(lambda f, _i: shapes.append(f.shape[:2]))
    cam.acquire("t")

    def pump(sec):
        end = time.time() + sec
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)
    pump(1.0)
    assert shapes and shapes[-1][0] > shapes[-1][1]  # vertical
    worker = cam._worker
    cam.set_rotation(0)
    shapes.clear()
    pump(0.6)
    assert cam._worker is worker and shapes[-1][0] < shapes[-1][1]
    cam.shutdown()
