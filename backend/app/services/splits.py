"""Particion train/val/test controlada y deteccion de fuga de datos.

Los casi duplicados van siempre al mismo split: separarlos filtraria informacion de
prueba hacia entrenamiento. Al bloquear, se guarda un hash que deja constancia del reparto.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from .. import db
from . import imaging

NEAR_BITS = 4


def _groups(images: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Agrupa casi duplicados por pHash con union-find."""
    parent = list(range(len(images)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(images):
        for j in range(i + 1, len(images)):
            if imaging.hamming(a["phash"], images[j]["phash"]) <= NEAR_BITS:
                parent[find(i)] = find(j)
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, image in enumerate(images):
        buckets[find(i)].append(image)
    return list(buckets.values())


def assign(project_id: str, train: int, val: int, test: int, seed: int) -> dict[str, Any]:
    project = db.one("SELECT * FROM projects WHERE id=?", [project_id])
    if project["split_locked_at"]:
        raise ValueError("La particion esta bloqueada")
    images = db.query(
        "SELECT id, phash, class_id FROM images WHERE project_id=? AND origin='real' AND status!='rejected' ORDER BY sha256",
        [project_id],
    )
    total = train + val + test
    if not images:
        raise ValueError("No hay imagenes reales")
    if total <= 0 or min(train, val, test) < 0:
        raise ValueError("Cantidades invalidas")

    rng = random.Random(seed)
    # Reparto estratificado por clase, proporcional a lo pedido, por grupos de casi duplicados.
    by_class: dict[Any, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in _groups(images):
        by_class[group[0]["class_id"]].append(group)
    assignment: dict[str, str] = {}
    fractions = {"train": train / total, "val": val / total, "test": test / total}
    for groups in by_class.values():
        rng.shuffle(groups)
        n = sum(len(g) for g in groups)
        targets = {k: round(v * n) for k, v in fractions.items()}
        counts = {k: 0 for k in fractions}
        for group in groups:
            split = max(targets, key=lambda k: targets[k] - counts[k])
            counts[split] += len(group)
            for image in group:
                assignment[image["id"]] = split
    with db.transaction():
        db.execute("UPDATE images SET split=NULL WHERE project_id=? AND origin='real'", [project_id])
        for image_id, split in assignment.items():
            db.update("images", image_id, {"split": split})
        db.update("projects", project_id, {
            "split_config": {"train": train, "val": val, "test": test, "seed": seed},
            "updated_at": db.now(),
        })
    return leakage_report(project_id)


def split_hash(project_id: str) -> str:
    rows = db.query(
        "SELECT split, sha256 FROM images WHERE project_id=? AND origin='real' AND split IS NOT NULL ORDER BY split, sha256",
        [project_id],
    )
    return hashlib.sha256("\n".join(f"{r['split']}:{r['sha256']}" for r in rows).encode()).hexdigest()


def lock(project_id: str) -> dict[str, Any]:
    report = leakage_report(project_id)
    if report["counts"]["unassigned"]:
        raise ValueError(f"Hay {report['counts']['unassigned']} imagenes sin particion")
    if not report["counts"]["train"] or not report["counts"]["test"]:
        raise ValueError("Se necesitan imagenes en train y test para bloquear")
    if report["cross_split_near_duplicates"]:
        raise ValueError("Hay casi duplicados entre particiones: repite la particion")
    digest = split_hash(project_id)
    db.update("projects", project_id, {"split_locked_at": db.now(), "split_hash": digest, "updated_at": db.now()})
    return {"split_hash": digest, **report}


def leakage_report(project_id: str) -> dict[str, Any]:
    rows = db.query(
        "SELECT id, filename, split, phash, sha256 FROM images WHERE project_id=? AND origin='real' AND status!='rejected'",
        [project_id],
    )
    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    for row in rows:
        counts[row["split"] or "unassigned"] += 1
    cross: list[dict[str, Any]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["split"] and b["split"] and a["split"] != b["split"]:
                distance = imaging.hamming(a["phash"], b["phash"])
                if distance <= NEAR_BITS:
                    cross.append({"a": a["filename"], "a_split": a["split"], "b": b["filename"], "b_split": b["split"], "distance": distance})

    # Un LoRA entrenado con imagenes de val o test contaminaria el experimento.
    protected = {r["sha256"] for r in rows if r["split"] in ("val", "test")}
    lora_leaks = []
    for lora in db.query("SELECT id, name, train_hashes FROM loras WHERE project_id=? AND train_hashes IS NOT NULL", [project_id]):
        overlap = protected & set(lora["train_hashes"] or [])
        if overlap:
            lora_leaks.append({"lora": lora["name"], "images": len(overlap)})
    return {"counts": counts, "cross_split_near_duplicates": cross, "lora_leaks": lora_leaks}
