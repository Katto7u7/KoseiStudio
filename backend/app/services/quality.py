"""Evaluación de las imágenes sintéticas contra el split de entrenamiento real.

Nunca usa val ni test. Ninguna métrica se presenta como prueba única de calidad: cada
una viene con lo que mide y con avisos cuando el tamaño de muestra no la sostiene.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from .. import db
from ..jobs import JobContext, register
from . import features, imaging, storage
from . import fidelity as fidelity_mod

log = logging.getLogger("kosei.quality")

MIN_FID_SAMPLES = 2048


def _inception_scores(ctx: JobContext, real: list[dict], synth: list[dict]) -> dict[str, Any]:
    """FID y KID con los pesos Inception estandar, comparables con la literatura."""
    import torch
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    subset = min(len(real), len(synth), 50)
    fid = FrechetInceptionDistance(feature=2048, normalize=False)
    kid = KernelInceptionDistance(subset_size=subset, subsets=100, normalize=False)

    def feed(rows: list[dict], is_real: bool, label: str) -> None:
        for start in range(0, len(rows), 8):
            ctx.check_cancel()
            ctx.progress(start, len(rows), f"Inception: {label}")
            batch = []
            for row in rows[start:start + 8]:
                image = Image.open(storage.resolve(row["normalized_path"])).convert("RGB").resize((299, 299), Image.Resampling.BICUBIC)
                batch.append(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))
            tensor = torch.from_numpy(np.stack(batch))
            fid.update(tensor, real=is_real)
            kid.update(tensor, real=is_real)

    feed(real, True, "reales")
    feed(synth, False, "sintéticas")
    kid_mean, kid_std = kid.compute()
    return {
        "fid": round(float(fid.compute()), 3),
        "kid_mean": round(float(kid_mean), 5),
        "kid_std": round(float(kid_std), 5),
        "kid_subset_size": subset,
    }


def improved_precision_recall(real: np.ndarray, fake: np.ndarray, k: int = 3) -> dict[str, float]:
    """Precision y recall k-NN de Kynkaanniemi et al. sobre embeddings CLIP."""
    def radii(x: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        return np.sort(distances, axis=1)[:, min(k, len(x) - 1) - 1]

    cross = np.linalg.norm(real[:, None] - fake[None, :], axis=-1)
    precision = float((cross <= radii(real)[:, None]).any(axis=0).mean())
    recall = float((cross <= radii(fake)[None, :]).any(axis=1).mean())
    return {"precision": round(precision, 4), "recall": round(recall, 4), "k": k}


def diversity(vectors: np.ndarray) -> float:
    """Distancia coseno media entre pares: 0 si todas son iguales."""
    if len(vectors) < 2:
        return 0.0
    sims = vectors @ vectors.T
    upper = sims[np.triu_indices(len(vectors), k=1)]
    return round(float(1 - upper.mean()), 4)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values)
    return {
        "mean": round(float(arr.mean()), 3), "p10": round(float(np.percentile(arr, 10)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3), "p90": round(float(np.percentile(arr, 90)), 3),
    }


@register("quality", "cpu")
def run_quality(ctx: JobContext) -> dict[str, Any]:
    project_id = ctx.project_id
    report_id = ctx.params["report_id"]
    real = db.query(
        "SELECT * FROM images WHERE project_id=? AND origin='real' AND split='train' AND status!='rejected'",
        [project_id],
    )
    synth = db.query(
        "SELECT * FROM images WHERE project_id=? AND origin='synthetic' AND status NOT IN ('rejected','pending')",
        [project_id],
    )
    if len(real) < 2 or len(synth) < 2:
        raise ValueError("Se necesitan al menos 2 imágenes reales en train y 2 sintéticas")

    warnings: list[str] = []
    if min(len(real), len(synth)) < MIN_FID_SAMPLES:
        warnings.append(
            f"FID con {len(real)} reales y {len(synth)} sintéticas no es fiable: se recomiendan "
            f"{MIN_FID_SAMPLES} o mas por grupo. Usalo solo para comparar corridas de igual tamaño; KID es menos sesgado."
        )
    if len(real) < 50:
        warnings.append("Con menos de 50 imágenes reales, precisión y recall tienen alta varianza.")

    scores = _inception_scores(ctx, real, synth)
    known = features.embed_images(ctx, real + synth)
    real_v = np.stack([known[r["id"]] for r in real])
    synth_v = np.stack([known[s["id"]] for s in synth])

    per_class: dict[str, Any] = {}
    classes = {c["id"]: c for c in db.query("SELECT * FROM classes WHERE project_id=?", [project_id])}
    for class_id, cls in classes.items():
        members = [i for i, s in enumerate(synth) if s["class_id"] == class_id]
        if not members:
            continue
        text = features.embed_text([f"a photo of a {cls['detector_prompt'] or cls['name']}"])[0]
        clip_scores = synth_v[members] @ text
        per_class[cls["name"]] = {
            "count": len(members),
            "clip_consistency": _distribution(clip_scores.tolist()),
            "real_baseline": _distribution((real_v @ text).tolist()),
        }

    nearest = (synth_v @ real_v.T).max(axis=1)
    memorized = int((nearest > 0.98).sum())
    if memorized:
        warnings.append(f"{memorized} sintéticas son casi identicas a una real de entrenamiento: posible memorización del LoRA.")

    phashes = [s["phash"] for s in synth]
    near_duplicates = sum(
        1 for i in range(len(phashes)) if any(imaging.hamming(phashes[i], phashes[j]) <= 4 for j in range(i))
    )

    all_flags: dict[str, int] = {}
    for row in synth:
        for flag in (row["qc"] or {}).get("flags", []):
            all_flags[flag] = all_flags.get(flag, 0) + 1

    status_rows = db.query(
        "SELECT status, COUNT(*) AS n FROM images WHERE project_id=? AND origin='synthetic' GROUP BY status",
        [project_id],
    )
    statuses = {r["status"]: r["n"] for r in status_rows}
    labeled = sum(v for k, v in statuses.items() if k != "pending")
    confidences = [
        r["confidence"] for r in db.query(
            "SELECT a.confidence FROM annotations a JOIN images i ON i.id=a.image_id "
            "WHERE i.project_id=? AND i.origin='synthetic' AND a.confidence IS NOT NULL",
            [project_id],
        )
    ]
    boxes = db.query(
        "SELECT a.w, a.h, i.origin FROM annotations a JOIN images i ON i.id=a.image_id WHERE i.project_id=?",
        [project_id],
    )

    scored = [(s["qc"] or {}).get("fidelity") for s in synth if (s["qc"] or {}).get("fidelity")]
    ref = fidelity_mod.reference(project_id)
    fidelity_report = {
        **fidelity_mod.summarize(scored),
        "threshold": fidelity_mod.SAME_PERSON_THRESHOLD,
        "real_baseline": ref["baseline_face"] if ref else None,
        "train_faces": ref["train_faces"] if ref else 0,
    }

    report = {
        "fidelity": fidelity_report,
        "samples": {"real_train": len(real), "synthetic": len(synth)},
        "inception": scores,
        "precision_recall": improved_precision_recall(real_v, synth_v),
        "diversity": {"synthetic": diversity(synth_v), "real": diversity(real_v)},
        "similarity_to_real": {**_distribution(nearest.tolist()), "memorized": memorized},
        "duplicates": {"near_duplicates": near_duplicates, "rate": round(near_duplicates / len(synth), 4)},
        "per_class": per_class,
        "visual": {
            "sharpness_synthetic": _distribution([(s["qc"] or {}).get("sharpness", 0) for s in synth]),
            "sharpness_real": _distribution([(r["qc"] or {}).get("sharpness", 0) for r in real]),
            "brightness_synthetic": _distribution([(s["qc"] or {}).get("brightness", 0) for s in synth]),
            "brightness_real": _distribution([(r["qc"] or {}).get("brightness", 0) for r in real]),
            "artifact_flags": all_flags,
        },
        "labeling": {
            "statuses": statuses,
            "no_detection_rate": round(statuses.get("no_detection", 0) / labeled, 4) if labeled else None,
            "confidence": _distribution(confidences),
            "box_area_real": _distribution([b["w"] * b["h"] for b in boxes if b["origin"] == "real"]),
            "box_area_synthetic": _distribution([b["w"] * b["h"] for b in boxes if b["origin"] == "synthetic"]),
        },
        "warnings": warnings,
        "omitted": {
            "inception_score": "No se calcula: mide diversidad entre clases de ImageNet y no tiene sentido para un dataset de una sola clase.",
        },
    }
    db.update("quality_reports", report_id, {"report": report, "status": "ready"})
    return {"report_id": report_id}
