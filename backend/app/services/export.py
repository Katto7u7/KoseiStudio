"""Constructor reproducible del dataset extendido, con exportación YOLO verificada.

Reglas: las sintéticas solo entran a train; val y test son solo reales; solo se exporta
lo aprobado por una persona; y el ZIP se abre y se valida antes de darlo por bueno.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image

from .. import db
from ..jobs import JobContext, register
from . import ingest, storage

log = logging.getLogger("kosei.export")

TASKS = ("detect", "segment", "classify")


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def select_images(project_id: str, n_synthetic: Optional[int], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Real por partición y sintéticas aprobadas, con subconjuntos anidados según la seed."""
    real = db.query(
        "SELECT * FROM images WHERE project_id=? AND origin='real' AND status='approved' AND split IS NOT NULL ORDER BY sha256",
        [project_id],
    )
    synth = db.query(
        "SELECT * FROM images WHERE project_id=? AND origin='synthetic' AND status='approved' ORDER BY sha256",
        [project_id],
    )
    random.Random(seed).shuffle(synth)
    if n_synthetic is not None:
        synth = synth[:n_synthetic]
    return {
        "train": [i for i in real if i["split"] == "train"] + synth,
        "val": [i for i in real if i["split"] == "val"],
        "test": [i for i in real if i["split"] == "test"],
    }


def yolo_lines(image_id: str, task: str, class_index: dict[int, int]) -> Optional[list[str]]:
    """Lineas de etiqueta. None si la imagen no puede representarse en esa tarea."""
    rows = db.query("SELECT * FROM annotations WHERE image_id=? ORDER BY id", [image_id])
    lines: list[str] = []
    for row in rows:
        cls = class_index[row["class_id"]]
        if task == "detect":
            lines.append(f"{cls} {row['cx']:.6f} {row['cy']:.6f} {row['w']:.6f} {row['h']:.6f}")
        else:
            polygon = row["polygon"]
            if not polygon or len(polygon) < 6:
                return None
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in polygon))
    return lines


def write_tree(root: Path, selection: dict[str, list[dict[str, Any]]], classes: list[dict[str, Any]], tasks: list[str], ctx: Optional[JobContext] = None) -> dict[str, Any]:
    class_index = {c["id"]: c["idx"] for c in classes}
    names = {c["idx"]: c["name"] for c in sorted(classes, key=lambda c: c["idx"])}
    summary: dict[str, Any] = {}
    manifest_items: list[dict[str, Any]] = []
    for task in tasks:
        task_dir = root / task
        counts = {split: {"real": 0, "synthetic": 0, "instances": 0, "skipped": 0} for split in selection}
        for split, images in selection.items():
            for image in images:
                if ctx:
                    ctx.check_cancel()
                src = storage.resolve(image["normalized_path"])
                name = f"{image['id']}.png"
                if task == "classify":
                    cls_name = names.get(class_index.get(image["class_id"]))
                    if cls_name is None:
                        counts[split]["skipped"] += 1
                        continue
                    _link(src, task_dir / split / cls_name / name)
                else:
                    lines = yolo_lines(image["id"], task, class_index)
                    if lines is None:
                        counts[split]["skipped"] += 1
                        continue
                    _link(src, task_dir / "images" / split / name)
                    label = task_dir / "labels" / split / f"{image['id']}.txt"
                    label.parent.mkdir(parents=True, exist_ok=True)
                    label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    counts[split]["instances"] += len(lines)
                counts[split][image["origin"]] += 1
        if task != "classify":
            (task_dir / "data.yaml").write_text(yaml.safe_dump({
                "path": ".", "train": "images/train", "val": "images/val", "test": "images/test",
                "names": names,
            }, sort_keys=False, allow_unicode=True), encoding="utf-8")
        summary[task] = counts
    for split, images in selection.items():
        for image in images:
            manifest_items.append({
                "id": image["id"], "file": f"{image['id']}.png", "split": split, "origin": image["origin"],
                "sha256_original": image["sha256"], "source_filename": image["filename"],
                "status": image["status"], "class_source": image["class_source"],
                "lora_id": image["lora_id"], "generation": image["generation"],
                "annotation_sources": sorted({
                    r["source"] for r in db.query("SELECT source FROM annotations WHERE image_id=?", [image["id"]])
                }),
            })
    return {"counts": summary, "items": manifest_items}


def validate_zip(path: Path, tasks: list[str]) -> dict[str, Any]:
    """Abre lo exportado de verdad: imagen legible, etiqueta correspondiente y formato correcto."""
    problems: list[str] = []
    checked = 0
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad:
            problems.append(f"Entrada corrupta: {bad}")
        names = set(zf.namelist())
        if "manifest.json" not in names:
            problems.append("Falta manifest.json")
        for task in tasks:
            if task == "classify":
                images = [n for n in names if n.startswith("classify/") and n.endswith(".png")]
                for name in images:
                    with Image.open(io.BytesIO(zf.read(name))) as image:
                        image.verify()
                    checked += 1
                continue
            yaml_name = f"{task}/data.yaml"
            if yaml_name not in names:
                problems.append(f"Falta {yaml_name}")
                continue
            n_classes = len(yaml.safe_load(zf.read(yaml_name))["names"])
            for name in [n for n in names if n.startswith(f"{task}/images/") and n.endswith(".png")]:
                label = name.replace("/images/", "/labels/", 1)[:-4] + ".txt"
                if label not in names:
                    problems.append(f"Sin etiqueta: {name}")
                    continue
                with Image.open(io.BytesIO(zf.read(name))) as image:
                    image.verify()
                for line in zf.read(label).decode("utf-8").splitlines():
                    parsed = ingest.parse_yolo_line(line)
                    if parsed is None or parsed[0] >= n_classes:
                        problems.append(f"Etiqueta inválida en {label}: {line[:60]}")
                    elif task == "detect" and len(parsed[1]) != 4:
                        problems.append(f"Se esperaban 4 valores en {label}")
                    elif task == "segment" and len(parsed[1]) < 6:
                        problems.append(f"Poligono inválido en {label}")
                checked += 1
    return {"ok": not problems, "checked_images": checked, "problems": problems[:50]}


@register("build", "cpu")
def run_build(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    build_id, project_id = p["build_id"], ctx.project_id
    project = db.one("SELECT * FROM projects WHERE id=?", [project_id])
    root = storage.project_dir(project_id) / "builds" / build_id
    if root.exists():
        shutil.rmtree(root)
    tree = root / "tree"
    tree.mkdir(parents=True)
    try:
        classes = db.query("SELECT * FROM classes WHERE project_id=? ORDER BY idx", [project_id])
        tasks = [t for t in p["tasks"] if t in TASKS]
        ctx.progress(0, 3, "Seleccionando imágenes")
        selection = select_images(project_id, p.get("n_synthetic"), int(p["seed"]))
        if not selection["train"]:
            raise ValueError("No hay imágenes aprobadas para train")
        if not selection["val"]:
            raise ValueError("No hay imágenes reales aprobadas en val")
        ctx.progress(1, 3, "Escribiendo imágenes y etiquetas")
        written = write_tree(tree, selection, classes, tasks, ctx)
        manifest = {
            "project": project["name"], "build_id": build_id, "name": p["name"], "created_at": db.now(),
            "config": {k: p[k] for k in ("n_synthetic", "seed", "tasks") if k in p},
            "split_hash": project["split_hash"],
            "classes": {c["idx"]: c["name"] for c in classes},
            "rules": "Sintéticas solo en train. Val y test solo reales. Solo imágenes aprobadas por una persona.",
            "counts": written["counts"], "items": written["items"],
        }
        (tree / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

        ctx.progress(2, 3, "Comprimiendo")
        safe_name = storage.safe_filename(p["name"], "dataset")
        zip_path = root / f"{safe_name}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for path in sorted(tree.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(tree).as_posix())
        validation = validate_zip(zip_path, tasks)
        digest = storage.sha256_file(zip_path)
        summary = {
            "counts": written["counts"], "validation": validation, "zip": zip_path.name,
            "size_bytes": zip_path.stat().st_size, "sha256": digest,
        }
        status = "ready" if validation["ok"] else "invalid"
        db.update("builds", build_id, {"summary": summary, "status": status})
        if not validation["ok"]:
            raise RuntimeError("La validación del ZIP encontro problemas: " + "; ".join(validation["problems"][:5]))
        return summary
    except Exception:
        row = db.one("SELECT status FROM builds WHERE id=?", [build_id])
        if row and row["status"] != "invalid":
            db.update("builds", build_id, {"status": "failed"})
        raise


def fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(path.name.encode())
    return h.hexdigest()
