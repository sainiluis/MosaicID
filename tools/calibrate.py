"""Calibración de umbrales con fotos reales etiquetadas (100 % local).

Uso:
    python tools/calibrate.py <carpeta_fotos> [--labels calibration/labels.csv] [--out calibration/report.json]

`labels.csv` tiene una fila por rostro: file, x, y, w, h, person, event, note.
- `person`: identidad verdadera (P1…; "?" = no usar; "EXT…" = persona ajena al grupo: nunca se usa como
  referencia ni para entrenar, y siempre debe salir "Desconocido").
- `event`: fotos de una misma ráfaga/salida comparten evento. Los pares del MISMO evento no se usan
  como evidencia genuina (dos fotos de una ráfaga son casi la misma imagen e inflarían los aciertos).

Protocolos (todos con el pipeline real de la app, la misma detección que la galería):
1. Verificación: pares genuinos (misma persona, eventos distintos) vs impostores (personas distintas).
2. Búsqueda en galería, dejando un evento afuera: referencias = rostros de la persona en OTROS eventos;
   se busca en las fotos del evento retenido. Precisión / exhaustividad por foto.
3. Conjunto abierto, dejando una persona afuera: se entrena con las demás y los rostros de la persona
   excluida deben quedar como "Desconocido" (Distancia y SVM). Con los mismos modelos se mide el acierto
   sobre las personas conocidas (evento retenido).

No elige umbrales "a medida" de una foto: informa las curvas y un punto de operación con criterio fijo.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaicid.config import load_settings  # noqa: E402
from mosaicid.core.gallery import load_image  # noqa: E402
from mosaicid.core.indexer import ROTATIONS  # noqa: E402
from mosaicid.core.matching import DistanceMatcher  # noqa: E402
from mosaicid.core.pipeline import FacePipeline, _iou  # noqa: E402
from mosaicid.core.svm import effective_threshold  # noqa: E402
from mosaicid.paths import Paths  # noqa: E402

GRID = np.round(np.arange(0.20, 0.701, 0.01), 3)


def extract(folder: Path, labels: list[dict], pipe: FacePipeline, settings) -> tuple[np.ndarray, list[dict], dict]:
    """Corre el pipeline de galería sobre cada foto y empareja cada rostro con su etiqueta por IoU."""
    by_file = defaultdict(list)
    for r in labels:
        by_file[r["file"]].append(r)
    embs, faces, photos = [], [], {}
    for fn in sorted({r["file"] for r in labels} | {p.name for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}):
        p = folder / fn
        if not p.exists():
            continue
        img, _ = load_image(p)
        det = pipe.process(img, max_side=settings.gallery_max_side, multiscale=True,
                           rotations=ROTATIONS if settings.gallery_rotated_detection else ())
        ev = next((r["event"] for r in by_file.get(fn, [])), f"solo:{fn}")
        photos[fn] = dict(event=ev, faces=[], labeled=len(by_file.get(fn, [])))
        for f in det:
            box = f.detection.box
            best = max(by_file.get(fn, []), key=lambda r: _iou(box, r["box"]), default=None)
            person = best["person"] if best and _iou(box, best["box"]) >= 0.3 else "?"
            idx = len(faces)
            faces.append(dict(file=fn, event=ev, person=person, box=box))
            embs.append(f.embedding / np.linalg.norm(f.embedding))
            photos[fn]["faces"].append(idx)
    return np.array(embs, np.float32), faces, photos


def rates(gen: np.ndarray, imp: np.ndarray) -> list[dict]:
    return [dict(t=float(t), far=float((imp >= t).mean()) if len(imp) else 0.0,
                 frr=float((gen < t).mean()) if len(gen) else 0.0) for t in GRID]


def verification(E, F) -> dict:
    gen, imp = [], []
    n = len(F)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = F[i], F[j]
            if "?" in (a["person"], b["person"]):
                continue
            s = float(E[i] @ E[j])
            if a["person"] == b["person"]:
                if a["event"] != b["event"]:
                    gen.append(s)
            else:
                imp.append(s)
    gen, imp = np.array(gen), np.array(imp)
    r = rates(gen, imp)
    eer = min(r, key=lambda x: abs(x["far"] - x["frr"]))
    return dict(genuine=len(gen), impostor=len(imp),
                genuine_pct=_pcts(gen), impostor_pct=_pcts(imp), eer=eer, curve=r,
                impostor_max=float(imp.max()) if len(imp) else None)


def _pcts(x):
    return {f"p{q}": float(np.percentile(x, q)) for q in (1, 5, 25, 50, 75, 95, 99)} if len(x) else {}


def gallery_search(E, F, photos) -> dict:
    """Leave-one-event-out: por persona, ¿qué fotos del evento retenido la contienen?"""
    people = sorted(p for p in {f["person"] for f in F} - {"?"} if not _ext(p))
    events = sorted({f["event"] for f in F})
    out = {float(t): dict(tp=0, fp=0, fn=0) for t in GRID}
    for ev in events:
        ev_photos = [p for p, d in photos.items() if d["event"] == ev]
        for person in people:
            refs = [i for i, f in enumerate(F) if f["person"] == person and f["event"] != ev]
            if not refs:
                continue
            R = E[refs]
            for p in ev_photos:
                idx = photos[p]["faces"]
                truth = any(F[i]["person"] == person for i in idx)
                score = max((float((R @ E[i]).max()) for i in idx), default=-1.0)
                for t in GRID:
                    hit = score >= t
                    c = out[float(t)]
                    c["tp"] += hit and truth
                    c["fp"] += hit and not truth
                    c["fn"] += (not hit) and truth
    res = []
    for t, c in out.items():
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 1.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        res.append(dict(t=t, precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0, **c))
    return dict(curve=res)


def open_set(E, F, strategy: str, top_k: int) -> dict:
    """Leave-one-person-out (desconocidos) + leave-one-event-out (conocidos), Distancia y SVM."""
    from sklearn.svm import SVC
    people = sorted(p for p in {f["person"] for f in F} - {"?"} if not _ext(p))
    lab = np.array([f["person"] for f in F])
    ext = np.array([_ext(x) for x in lab])
    ext_d, ext_p = [], []           # personas ajenas reales (no simuladas)
    ev = np.array([f["event"] for f in F])
    unk_d, unk_p = [], []           # mejor similitud / probabilidad de un desconocido
    kn_d, kn_ok_d, kn_p, kn_ok_p = [], [], [], []
    for out_p in people:
        train_mask = (lab != out_p) & (lab != "?") & ~ext
        for e in sorted(set(ev)):
            tr = train_mask & (ev != e)
            te_known = train_mask & (ev == e)
            te_ext = ext & (ev == e) if out_p == people[0] else np.zeros_like(ext)  # una vez por evento
            te_unk = (lab == out_p) & (ev == e)
            if not (te_known.any() or te_unk.any() or te_ext.any()):
                continue
            classes = sorted(set(lab[tr]))
            if len(classes) < 2:
                continue
            pid = {c: k for k, c in enumerate(classes)}  # la app usa ids enteros
            m = DistanceMatcher(E[tr], [pid[x] for x in lab[tr]], threshold=0.0, strategy=strategy, top_k=top_k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                svm = SVC(C=1.0, kernel="linear", probability=True, random_state=0).fit(E[tr], lab[tr])
            for i in np.where(te_ext)[0]:
                d = m.identify(E[i])
                ext_d.append(d.similarity)
                ext_p.append(float(svm.predict_proba(E[i:i + 1])[0].max()))
            for i in np.where(te_known | te_unk)[0]:
                d = m.identify(E[i])
                pr = svm.predict_proba(E[i:i + 1])[0]
                k = int(pr.argmax())
                if te_unk[i]:
                    unk_d.append(d.similarity)
                    unk_p.append(float(pr[k]))
                else:
                    kn_d.append(d.similarity)
                    kn_ok_d.append(classes[d.candidate_id] == lab[i])
                    kn_p.append(float(pr[k]))
                    kn_ok_p.append(svm.classes_[k] == lab[i])
    kn_d, kn_ok_d, kn_p, kn_ok_p = map(np.array, (kn_d, kn_ok_d, kn_p, kn_ok_p))
    unk_d, unk_p = np.array(unk_d), np.array(unk_p)

    def curve(score, ok, unk, grid):
        rows = []
        for t in grid:
            acc = float(((score >= t) & ok).mean()) if len(score) else 0.0     # conocido aceptado y bien
            wrong = float(((score >= t) & ~ok).mean()) if len(score) else 0.0  # conocido aceptado como otro
            rej = float((score < t).mean()) if len(score) else 0.0              # conocido rechazado
            far = float((unk >= t).mean()) if len(unk) else 0.0                 # desconocido aceptado
            rows.append(dict(t=float(t), correct=acc, wrong=wrong, rejected=rej, unknown_accepted=far))
        return rows

    return dict(
        known_queries=len(kn_d), unknown_queries=len(unk_d),
        closed_set_top1_distance=float(kn_ok_d.mean()) if len(kn_ok_d) else None,
        closed_set_top1_svm=float(kn_ok_p.mean()) if len(kn_ok_p) else None,
        distance=curve(kn_d, kn_ok_d, unk_d, GRID),
        # SVM: el umbral se aplica como margen sobre el azar con K = clases entrenadas en cada pliegue
        svm=[dict(r, t=float(m)) for m, r in ((m, curve(kn_p, kn_ok_p, unk_p, [effective_threshold(m, len(people) - 1)])[0])
                                             for m in np.round(np.arange(0.0, 0.951, 0.01), 3))],
        unknown_distance_pct=_pcts(unk_d), unknown_svm_pct=_pcts(unk_p),
        external=dict(n=len(ext_d), distance_best=ext_d, svm_best_prob=ext_p),
    )


def svm_by_classes(E, F, margins=(0.2, 0.3, 0.35, 0.4, 0.5)) -> dict:
    """Con K personas registradas (todas las combinaciones de K integrantes), ¿cómo se comporta el rechazo?
    Muestra por qué el umbral es relativo al azar: la escala de predict_proba depende de K."""
    import itertools
    from sklearn.svm import SVC
    lab = np.array([f["person"] for f in F])
    ev = np.array([f["event"] for f in F])
    people = sorted(p for p in set(lab) - {"?"} if not _ext(p))
    out = {}
    for K in range(2, len(people)):
        kp, ku, ok = [], [], []
        for sub in itertools.combinations(people, K):
            for e in sorted(set(ev)):
                tr = np.isin(lab, sub) & (ev != e)
                te = ev == e
                if len(set(lab[tr])) < K or not te.any():
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SVC(kernel="linear", C=1.0, probability=True, random_state=0).fit(E[tr], lab[tr])
                P = m.predict_proba(E[te])
                pm, pc = P.max(1), m.classes_[P.argmax(1)]
                known = np.isin(lab[te], sub)
                kp += list(pm[known]); ok += list(pc[known] == lab[te][known]); ku += list(pm[~known])  # noqa: E702
        kp, ku, ok = map(np.array, (kp, ku, ok))
        out[K] = dict(known_p50=float(np.median(kp)), unknown_p50=float(np.median(ku)), rows=[
            dict(margin=m, t=effective_threshold(m, K), correct=float(((kp >= effective_threshold(m, K)) & ok).mean()),
                 unknown_accepted=float((ku >= effective_threshold(m, K)).mean())) for m in margins])
    return out


def _ext(person: str) -> bool:
    return person.upper().startswith("EXT")


def pick(curve, key_bad, key_good, max_bad):
    """Punto de operación: el umbral más bajo con tasa de error ≤ max_bad (criterio fijo, no por foto)."""
    ok = [r for r in curve if r[key_bad] <= max_bad]
    return min(ok, key=lambda r: r["t"]) if ok else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--labels", type=Path, default=ROOT / "calibration" / "labels.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "calibration" / "report.json")
    a = ap.parse_args()

    labels = []
    with open(a.labels, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels.append(dict(file=r["file"], box=tuple(float(r[k]) for k in "xywh"),
                               person=r["person"].strip() or "?", event=r["event"].strip()))
    paths = Paths()
    s = load_settings(paths.config_file)
    pipe = FacePipeline(paths.yunet_model, paths.sface_model, s.yunet_score_threshold, s.yunet_nms_threshold)
    E, F, photos = extract(a.folder, labels, pipe, s)
    unmatched = sum(1 for f in F if f["person"] == "?")
    counts = defaultdict(int)
    for f in F:
        counts[f["person"]] += 1

    ver = verification(E, F)
    gal = gallery_search(E, F, photos)
    ops = open_set(E, F, s.distance_strategy, s.distance_top_k)
    ops["svm_by_classes"] = svm_by_classes(E, F)

    rep = dict(photos=len(photos), faces=len(F), unlabeled_faces=unmatched, per_person=dict(counts),
               events=len({f["event"] for f in F}), current=dict(distance=s.distance_cosine_threshold,
                                                                 gallery=s.gallery_match_threshold,
                                                                 svm=s.svm_rejection_threshold),
               verification=ver, gallery=gal, open_set=ops)
    rep["suggested"] = dict(
        distance=pick(ops["distance"], "unknown_accepted", "correct", 0.01),
        svm=pick(ops["svm"], "unknown_accepted", "correct", 0.05),
        gallery=pick(gal["curve"], "fp", "recall", 0),
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    print(f"{rep['photos']} fotos · {rep['faces']} rostros · {rep['events']} eventos · sin etiqueta {unmatched}")
    print("por persona:", dict(sorted(counts.items())))
    print(f"\nVERIFICACIÓN  genuinos {ver['genuine']} (entre eventos) · impostores {ver['impostor']}")
    print("  genuinos   ", {k: round(v, 3) for k, v in ver["genuine_pct"].items()})
    print("  impostores ", {k: round(v, 3) for k, v in ver["impostor_pct"].items()}, "máx", round(ver["impostor_max"], 3))
    print(f"  EER ≈ {ver['eer']['far']:.1%} en t={ver['eer']['t']}")
    for t in (0.30, 0.363, 0.40, 0.45):
        r = min(ver["curve"], key=lambda x: abs(x["t"] - t))
        print(f"  t={r['t']:.2f}  FAR {r['far']:.2%}  FRR {r['frr']:.2%}")
    print("\nGALERÍA (evento retenido)")
    for t in (0.30, 0.363, 0.40, 0.45):
        r = min(gal["curve"], key=lambda x: abs(x["t"] - t))
        print(f"  t={r['t']:.2f}  precisión {r['precision']:.1%}  exhaustividad {r['recall']:.1%}  (tp {r['tp']} fp {r['fp']} fn {r['fn']})")
    print(f"\nCONJUNTO ABIERTO  conocidos {ops['known_queries']} · desconocidos {ops['unknown_queries']}")
    print(f"  top-1 cerrado: Distancia {ops['closed_set_top1_distance']:.1%} · SVM {ops['closed_set_top1_svm']:.1%}")
    for name, key, ts in (("Distancia", "distance", (0.30, 0.363, 0.40, 0.45)), ("SVM margen", "svm", (0.3, 0.4, 0.5))):
        for t in ts:
            r = min(ops[key], key=lambda x: abs(x["t"] - t))
            print(f"  {name:<9} t={r['t']:.2f}  correcto {r['correct']:.1%}  confundido {r['wrong']:.1%}  "
                  f"rechazado {r['rejected']:.1%}  desconocido aceptado {r['unknown_accepted']:.1%}")
    ex = ops["external"]
    if ex["n"]:
        print(f"  Ajenos reales ({ex['n']}): mejor similitud Distancia {[round(x, 3) for x in ex['distance_best']]}"
              f" · prob. SVM {[round(x, 2) for x in ex['svm_best_prob']]}")
    print("\n  SVM según cantidad de personas registradas (K):")
    for K, d in ops["svm_by_classes"].items():
        row = " · ".join(f"m{r['margin']:.2f}→t{r['t']:.2f}: {r['correct']:.0%} ok / {r['unknown_accepted']:.0%} desc." for r in d["rows"])
        print(f"   K={K} (prob. conocido {d['known_p50']:.2f}, desconocido {d['unknown_p50']:.2f})  {row}")
    print("\nSUGERIDO (criterio fijo):", json.dumps({k: (v and v["t"]) for k, v in rep["suggested"].items()}))
    print(f"Reporte: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
