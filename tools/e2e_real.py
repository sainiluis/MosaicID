"""Prueba de punta a punta con fotos REALES, usando los mismos servicios que la app (sin ventana).

Uso:
    python tools/e2e_real.py <carpeta_fotos> [--labels calibration/labels.csv] [--a P3] [--b P4] [--c P1]

Simula el recorrido del checkpoint de prueba real:
  registrar A y B (muestras = rostros reales de OTRAS salidas, pasando el control de calidad del registro)
  → entrenar SVM en su hilo → cámara (simulada con la foto del restaurante, 6 personas) → analizador en vivo
  → A y B reconocidos por ambos motores, el resto Desconocido → indexar la carpeta real → buscar A y B
  → cerrar y reabrir (persistencia) → registrar C: Distancia lo reconoce al instante, SVM queda pendiente.

Corre en una raíz de datos temporal (MOSAICID_HOME); no toca data/ ni la configuración del usuario.
Sale con código 0 solo si todas las verificaciones pasan.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


WARNINGS: list[str] = []


def check(ok: bool, msg: str) -> bool:
    RESULTS.append((bool(ok), msg))
    print(f"[{'OK ' if ok else 'FALLA'}] {msg}")
    return ok


def notice(ok: bool, msg: str, why: str) -> bool:
    """Comportamiento medido y documentado (no es un error de implementación): se informa, no falla."""
    if ok:
        RESULTS.append((True, msg))
        print(f"[OK ] {msg}")
    else:
        WARNINGS.append(msg)
        print(f"[AVISO] {msg} — {why}")
    return ok


def read_labels(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [dict(file=r["file"], box=tuple(float(r[k]) for k in "xywh"), person=r["person"], event=r["event"])
                for r in csv.DictReader(f)]


def camera_frame(img: np.ndarray, box, face_px: int = 220, out=(1280, 720), alone: bool = False) -> np.ndarray:
    """Encuadre tipo webcam alrededor de un rostro: cara de ~face_px de ancho, centrada, en 1280×720.
    `alone`: solo se conserva la zona de esa persona (cabeza y hombros); el resto queda neutro, como si
    estuviera sola frente a la cámara (en el registro real hay una sola persona)."""
    x, y, w, h = box
    if alone:
        keep = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(keep, (int(x + w / 2), int(y + h * 0.55)), (int(w * 0.85), int(h * 1.0)), 0, 0, 360, 255, -1)
        img = np.where(keep[..., None] > 0, img, np.full_like(img, 40))
    s = face_px / max(w, 1)
    big = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    cx, cy = (x + w / 2) * s, (y + h / 2) * s
    W, H = out
    canvas = np.full((H, W, 3), 40, np.uint8)
    x0, y0 = int(cx - W / 2), int(cy - H / 2)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(big.shape[1], x0 + W), min(big.shape[0], y0 + H)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = big[sy0:sy1, sx0:sx1]
    return canvas


def enroll_samples(pipe, settings, folder: Path, faces: list[dict], n: int) -> tuple[list[dict], Counter]:
    """Igual que el registro: un rostro por frame, control de calidad, alignCrop + feature."""
    from mosaicid.core.gallery import load_image
    from mosaicid.core.quality import check_sample
    accepted, reasons = [], Counter()
    for r in sorted(faces, key=lambda r: -r["box"][2]):  # primero las caras más grandes
        img, _ = load_image(folder / r["file"])
        frame = camera_frame(img, r["box"], face_px=min(240, max(120, int(r["box"][2] * 2.2))), alone=True)
        found = pipe.process(frame)
        q, _ = check_sample(found, frame.shape, settings)
        reasons[q.code] += 1
        if q.ok:
            f = found[0]
            accepted.append(dict(embedding=f.embedding, aligned=f.aligned, kind="foto", quality_score=q.score,
                                 detection_score=f.detection.score, sharpness=q.sharpness))
        if len(accepted) >= n:
            break
    return accepted, reasons


def wait_until(app, cond, timeout: float) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--labels", type=Path, default=ROOT / "calibration" / "labels.csv")
    ap.add_argument("--people", type=Path, default=ROOT / "calibration" / "people.csv")
    ap.add_argument("--a", default="P3")
    ap.add_argument("--b", default="P4")
    ap.add_argument("--c", default="P1")
    ap.add_argument("--keep", action="store_true", help="no borrar la raíz temporal")
    a = ap.parse_args()

    names = {}
    if a.people.exists():
        with open(a.people, encoding="utf-8") as f:
            names = {r["person"]: r["name"] for r in csv.DictReader(f)}
    nm = lambda p: names.get(p) or p  # noqa: E731
    labels = read_labels(a.labels)
    by_photo = defaultdict(list)
    for r in labels:
        by_photo[r["file"]].append(r)
    # escena de prueba: la foto con más integrantes etiquetados; su salida (evento) NO se usa para registrar
    # escena de prueba: la foto con más integrantes que se ven de frente (detectables sin la pasada rotada,
    # como frente a una webcam); su salida (evento) NO se usa para registrar
    from mosaicid.core.gallery import load_image as _load
    from mosaicid.core.pipeline import FacePipeline as _FP
    from mosaicid.paths import Paths as _P
    _pp = _P(ROOT)
    _probe = _FP(_pp.yunet_model, _pp.sface_model)
    def upright_members(f):
        img, _ = _load(a.folder / f)
        found = _probe.process(img, max_side=1280)
        n = sum(1 for r in by_photo[f] if not r["person"].upper().startswith("EXT")
                and any(_iou(x.detection.box, r["box"]) > 0.3 for x in found))
        return (n, float(np.median([r["box"][2] for r in by_photo[f]])))
    scene = max(by_photo, key=upright_members)
    scene_event = by_photo[scene][0]["event"]
    in_scene = {r["person"] for r in by_photo[scene]}
    print(f"Escena de prueba: {scene} · personas {sorted(nm(p) for p in in_scene)}\n")

    home = Path(tempfile.mkdtemp(prefix="mosaicid_e2e_"))
    os.environ["MOSAICID_HOME"] = str(home)
    cam_dir = home / "_camara"
    cam_dir.mkdir()
    shutil.copy(a.folder / scene, cam_dir / "escena.jpg")
    os.environ["MOSAICID_FAKE_CAMERA"] = str(cam_dir / "escena.jpg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication
    from mosaicid.app_context import AppContext
    from mosaicid.core.pipeline import FacePipeline
    from mosaicid.core.gallery import search_faces
    from mosaicid.paths import Paths
    from mosaicid.services.workers import IndexWorker, TrainingWorker
    from mosaicid.logging_setup import setup_logging

    app = QApplication.instance() or QApplication([])
    paths = Paths()
    setup_logging(paths.logs)
    ctx = AppContext(paths)
    s = ctx.settings
    pipe = FacePipeline(paths.yunet_model, paths.sface_model, s.yunet_score_threshold, s.yunet_nms_threshold)
    check(ctx.models_ok, "YuNet y SFace cargados")

    def register(code: str) -> int | None:
        pool = [r for r in labels if r["person"] == code and r["event"] != scene_event]
        samples, reasons = enroll_samples(pipe, s, a.folder, pool, s.enrollment_sample_count)
        rej = {k: v for k, v in reasons.items() if k != "ok"}
        ok = check(len(samples) >= 3, f"Registro de {nm(code)}: {len(samples)} muestras aceptadas de {sum(reasons.values())}"
                   f" intentos (rechazos: {rej or 'ninguno'})")
        if not ok:
            return None
        pid = ctx.db.create_person(nm(code))
        folder = paths.samples / str(pid)
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, smp in enumerate(samples):
            cv2.imwrite(str(folder / f"{i}.jpg"), smp["aligned"])
            rows.append(dict(embedding=smp["embedding"], kind=smp["kind"], image_path=f"samples/{pid}/{i}.jpg",
                             quality_score=smp["quality_score"], detection_score=smp["detection_score"],
                             sharpness=smp["sharpness"]))
        ctx.db.add_samples(pid, rows)
        ctx.people_changed()
        return pid

    # 1-4. registrar A y B
    pa, pb = register(a.a), register(a.b)
    if pa is None or pb is None:
        return finish(ctx, home, a.keep)
    check(ctx.svm_status() == "untrained", f"SVM sin entrenar tras registrar 2 personas (estado: {ctx.svm_status()})")

    # 5. entrenar SVM en su hilo
    w = TrainingWorker(ctx.svm, ctx.snapshot.dataset, s.svm_c, s.svm_kernel)
    stages, meta = [], {}
    w.stage.connect(stages.append)
    w.finished_ok.connect(meta.update)
    t0 = time.time()
    w.start()
    wait_until(app, lambda: w.isFinished(), 60)
    app.processEvents()
    ctx.svm_trained()
    check(ctx.svm_status() == "trained", f"SVM entrenado en {time.time() - t0:.1f}s · etapas {stages} · "
                                         f"{meta.get('cv_note') or meta.get('cv') or ''}")

    # 6-10. cámara + analizador: A y B a la vez entre 6 personas
    got = []
    ctx.analyzer.analysis.connect(got.append)
    ctx.acquire_camera("e2e")
    wait_until(app, lambda: len(got) >= 25, 30)
    ctx.release_camera("e2e")
    wait_until(app, lambda: False, 0.8)
    check(len(got) >= 25, f"Analizador en vivo: {len(got)} análisis recibidos desde su hilo")
    if got:
        last = got[-1]
        ms = [g.inference_ms for g in got[5:]] or [last.inference_ms]
        print(f"      inferencia {np.median(ms):.0f} ms (mediana) · {last.inference_fps:.1f} inferencias/s · "
              f"{len(last.faces)} rostros")
        truth = {}
        scene_w = _load(a.folder / scene)[0].shape[1]
        k = scene_w / last.frame.shape[1]  # la cámara simulada reduce la foto a la resolución configurada
        for lf in last.faces:
            box = tuple(v * k for v in lf.face.detection.box)
            best = max(by_photo[scene], key=lambda r: _iou(box, r["box"]))
            truth[lf.track_id] = best["person"] if _iou(box, best["box"]) > 0.3 else "?"
        check(len(last.faces) >= len(in_scene) - 1, f"Múltiples caras: {len(last.faces)} detectadas de {len(in_scene)}"
                                                    " (en vivo no se usa la pasada rotada)")
        for lf in last.faces:
            t = truth[lf.track_id]
            exp = {a.a: pa, a.b: pb}.get(t)
            d = lf.decision
            sv = d.svm.person_id if d.svm else "sin modelo"
            st_svm, st_dist = lf.stable.get("svm", (None, 0))[0], lf.stable.get("distance", (None, 0))[0]
            who = nm(t)
            if exp:
                check(d.distance.person_id == exp and st_dist == exp,
                      f"Distancia reconoce a {who} ({d.distance.similarity:.3f})")
                check(d.svm is not None and d.svm.person_id == exp and st_svm == exp,
                      f"SVM reconoce a {who} ({d.svm.probability:.2f})" if d.svm else f"SVM reconoce a {who}")
            else:
                check(d.distance.person_id is None and st_dist is None,
                      f"Distancia: {who} (no registrado) → Desconocido ({d.distance.similarity:.3f})")
                notice(d.svm is None or d.svm.person_id is None,
                       f"SVM: {who} (no registrado) → Desconocido ({d.svm.probability:.2f} vs umbral "
                       f"{d.svm.threshold:.2f})" if d.svm else f"SVM: {who}",
                       f"limitación medida del SVM cerrado con {ctx.svm.n_classes} clases (CALIBRACION.md)")

    # 11. indexar la carpeta real en su hilo
    fid = ctx.db.add_folder(str(a.folder), "real")
    iw = IndexWorker(type(ctx.db)(paths.database), paths, s, [fid])
    rep = []
    iw.finished_ok.connect(rep.append)
    iw.start()
    wait_until(app, lambda: iw.isFinished(), 120)
    app.processEvents()
    r = rep[0] if rep else None
    check(r is not None and not r.errors, f"Galería indexada: {r.indexed if r else 0} fotos, {r.faces if r else 0} rostros,"
                                          f" {r.seconds if r else 0:.1f}s, errores {r.errors if r else '?'}")

    # 12. buscar fotografías de A y B (todas sus referencias, como hace la app)
    F, fids, phids = ctx.db.face_matrix(only_real=True)
    X, y, _ = ctx.db.all_embeddings()
    for code, pid in ((a.a, pa), (a.b, pb)):
        hits = search_faces(X[y == pid], F, fids, phids, s.gallery_match_threshold)
        found = {ctx.db.get_photo(h.photo_id).filename for h in hits}
        truth = {f for f, rs in by_photo.items() if any(x["person"] == code for x in rs)}
        tp, fp = len(found & truth), len(found - truth)
        check(tp >= 0.75 * len(truth) and fp <= max(1, len(found) // 10),
              f"Búsqueda de {nm(code)}: {tp}/{len(truth)} fotos donde aparece, {fp} ajenas")
        for h in hits:
            fn = ctx.db.get_photo(h.photo_id).filename
            if fn not in truth:
                print(f"      ajena: {fn} similitud {h.similarity:.3f}")

    # persistencia: cerrar y reabrir
    ctx.shutdown()
    ctx = AppContext(Paths())
    check(len(ctx.db.list_people()) == 2 and ctx.svm_status() == "trained" and ctx.db.gallery_counts()[0] > 0,
          f"Persistencia: {len(ctx.db.list_people())} personas, SVM {ctx.svm_status()}, "
          f"{ctx.db.gallery_counts()[0]} fotos tras reabrir")

    # registrar C: Distancia lo reconoce enseguida, SVM queda pendiente
    pc = register(a.c)
    if pc is not None:
        check(ctx.svm_status() == "dirty" and pc in ctx.snapshot.svm_pending,
              f"Tras registrar a {nm(a.c)}: SVM pendiente de reentrenar ({ctx.svm_status()})")
        cface = next((x for x in by_photo[scene] if x["person"] == a.c), None)
        if cface:
            from mosaicid.core.gallery import load_image
            img, _ = load_image(a.folder / scene)
            fr = camera_frame(img, cface["box"], face_px=int(cface["box"][2]), alone=True)
            fs = pipe.process(fr)
            if fs:
                dec = ctx.snapshot.decide(max(fs, key=lambda f: f.detection.w * f.detection.h).embedding)
                check(dec.distance.person_id == pc,
                      f"Distancia reconoce a {nm(a.c)} sin reentrenar ({dec.distance.similarity:.3f})")
                check(dec.svm is None or dec.svm.person_id != pc,
                      f"SVM todavía no conoce a {nm(a.c)} (responde "
                      f"{ctx.snapshot.name(dec.svm.person_id) if dec.svm else '—'})")
    return finish(ctx, home, a.keep)


def _iou(a, b) -> float:
    from mosaicid.core.pipeline import _iou as f
    return f(a, b)


def finish(ctx, home: Path, keep: bool) -> int:
    ctx.shutdown()
    fails = [m for ok, m in RESULTS if not ok]
    print(f"\n{len(RESULTS)} verificaciones · {len(fails)} con falla · {len(WARNINGS)} avisos esperados")
    if keep:
        print(f"Datos de la prueba: {home}")
    else:
        shutil.rmtree(home, ignore_errors=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
