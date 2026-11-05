"""Etiquetado automatico: clase por condicionamiento, deteccion YOLO, mascara SAM.

Corre en CPU para no competir por VRAM con el motor de generacion. Las etiquetas que
produce son candidatas: solo pasan a verificadas cuando una persona las aprueba.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .. import config, db, jobs
from ..jobs import JobContext, register
from . import storage

log = logging.getLogger("kosei.labeling")

_lock = threading.RLock()
_models: dict[str, Any] = {}

# Nombres en espanol frecuentes para las clases COCO.
COCO_ALIASES = {
    "persona": "person", "gente": "person", "auto": "car", "coche": "car", "carro": "car",
    "perro": "dog", "gato": "cat", "bicicleta": "bicycle", "camion": "truck", "autobus": "bus",
}


def _yolo(name: str):
    with _lock:
        if name not in _models:
            from ultralytics import YOLO
            _models[name] = YOLO(str(config.YOLO_DIR / name))
        return _models[name]


def _sam():
    with _lock:
        if "sam" not in _models:
            from ultralytics import SAM
            _models["sam"] = SAM(str(config.YOLO_DIR / "mobile_sam.pt"))
        return _models["sam"]


def _world(prompt: str):
    """YOLO-World con vocabulario abierto, para clases que COCO no conoce."""
    with _lock:
        key = f"world::{prompt}"
        if key not in _models:
            from ultralytics import YOLOWorld
            model = YOLOWorld(str(config.YOLO_DIR / "yolov8s-worldv2.pt"))
            model.set_classes([prompt])
            _models[key] = model
        return _models[key]


def coco_name(cls: dict[str, Any]) -> Optional[str]:
    """Nombre COCO de la clase, o None si COCO no la conoce."""
    wanted = (cls.get("detector_prompt") or cls["name"]).strip().lower()
    wanted = COCO_ALIASES.get(wanted, wanted)
    names = {str(v).lower() for v in _yolo("yolo11m.pt").names.values()}
    return wanted if wanted in names else None


def detect(image_path: Path, cls: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Devuelve las cajas de ESTA clase. Lo que el detector vea de otra clase se descarta."""
    use_world = cls.get("detector") == "world"
    target = None if use_world else coco_name(cls)
    if not use_world and target is None:
        use_world = True
    if use_world:
        prompt = (cls.get("detector_prompt") or cls["name"]).strip()
        with _lock:
            result = _world(prompt).predict(str(image_path), conf=0.05, device="cpu", verbose=False)[0]
        source = "yolo-world"
    else:
        with _lock:
            result = _yolo("yolo11m.pt").predict(str(image_path), conf=config.LABEL_MIN_CONF, device="cpu", verbose=False)[0]
        source = "yolo11m-coco"
    boxes: list[dict[str, Any]] = []
    if result.boxes is not None:
        for xyxy, conf, cid in zip(result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            if not use_world and str(result.names[int(cid)]).lower() != target:
                continue
            x0, y0, x1, y1 = xyxy
            boxes.append({
                "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0,
                "confidence": round(float(conf), 4),
            })
    boxes.sort(key=lambda b: -b["confidence"])
    return boxes, source


def mask_to_polygon(mask: np.ndarray, max_points: int = 80) -> Optional[list[float]]:
    """Contorno exterior mayor de una mascara, como poligono YOLO normalizado."""
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    height, width = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 16:
        return None
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    while len(approx) > max_points:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    out: list[float] = []
    for x, y in approx.reshape(-1, 2).tolist():
        out += [round(min(max(x / width, 0.0), 1.0), 6), round(min(max(y / height, 0.0), 1.0), 6)]
    return out


def segment(image_path: Path, boxes: list[dict[str, Any]], width: int, height: int) -> list[Optional[list[float]]]:
    """SAM con cada caja como prompt, como describe la tesis."""
    if not boxes:
        return []
    bboxes = [
        [(b["cx"] - b["w"] / 2) * width, (b["cy"] - b["h"] / 2) * height,
         (b["cx"] + b["w"] / 2) * width, (b["cy"] + b["h"] / 2) * height]
        for b in boxes
    ]
    with _lock:
        result = _sam().predict(str(image_path), bboxes=bboxes, device="cpu", verbose=False)[0]
    polygons: list[Optional[list[float]]] = [None] * len(boxes)
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data.cpu().numpy()[: len(boxes)]):
            polygons[i] = mask_to_polygon(mask)
    return polygons


def decide_status(boxes: list[dict[str, Any]], polygons: list[Optional[list[float]]], flags: list[str]) -> tuple[str, list[str]]:
    """Reglas de aceptacion. Nada se aprueba solo: `candidate` aun exige revision o aprobacion en lote."""
    if not boxes:
        return "no_detection", ["sin detecciones de la clase"]
    reasons: list[str] = []
    if boxes[0]["confidence"] < config.LABEL_ACCEPT_CONF:
        reasons.append(f"confianza {boxes[0]['confidence']:.2f} menor a {config.LABEL_ACCEPT_CONF:.2f}")
    if len(boxes) > 1:
        reasons.append(f"{len(boxes)} instancias detectadas")
    if any(b["w"] * b["h"] < 0.01 for b in boxes):
        reasons.append("caja muy pequena")
    if any(b["w"] * b["h"] > 0.97 for b in boxes):
        reasons.append("caja ocupa toda la imagen")
    if any(p is None for p in polygons):
        reasons.append("sin mascara valida")
    reasons += [f"imagen {f}" for f in flags]
    return ("needs_review" if reasons else "candidate"), reasons


def _class_for(image: dict[str, Any], classes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if image["class_id"]:
        return next((c for c in classes if c["id"] == image["class_id"]), None)
    return classes[0] if len(classes) == 1 else None


def label_image(image: dict[str, Any], classes: list[dict[str, Any]]) -> str:
    cls = _class_for(image, classes)
    qc = image["qc"] or {}
    if cls is None:
        db.update("images", image["id"], {"status": "needs_review", "qc": {**qc, "label_reasons": ["sin clase asignada"]}})
        return "needs_review"
    path = storage.resolve(image["normalized_path"])
    boxes, source = detect(path, cls)
    polygons = segment(path, boxes, image["width"], image["height"])
    status, reasons = decide_status(boxes, polygons, qc.get("flags") or [])
    if qc.get("near_duplicate_of"):
        status = "needs_review"
        reasons.append("casi duplicada de otra imagen")
    with db.transaction():
        db.execute("DELETE FROM annotations WHERE image_id=? AND verified=0", [image["id"]])
        for box, polygon in zip(boxes, polygons):
            db.insert("annotations", {
                "image_id": image["id"], "class_id": cls["id"],
                "cx": box["cx"], "cy": box["cy"], "w": box["w"], "h": box["h"],
                "polygon": polygon, "confidence": box["confidence"], "source": source,
                "mask_source": "mobile-sam" if polygon else None, "verified": 0,
                "created_at": db.now(),
            })
        class_source = image["class_source"] or ("condicionamiento" if image["origin"] == "synthetic" else "proyecto")
        db.update("images", image["id"], {
            "status": status, "class_id": cls["id"], "class_source": class_source,
            "qc": {**qc, "label_reasons": reasons},
        })
    return status


def ensure_label_job(project_id: str) -> Optional[str]:
    """Una sola tarea de etiquetado por proyecto; se encarga de todo lo pendiente."""
    active = jobs.active_job(project_id, "label")
    return active["id"] if active else jobs.enqueue("label", project_id, {})


@register("label", "cpu")
def run_label(ctx: JobContext) -> dict[str, Any]:
    project_id = ctx.project_id
    classes = db.query("SELECT * FROM classes WHERE project_id=? ORDER BY idx", [project_id])
    counts: dict[str, int] = {}
    done = 0
    while True:
        ctx.check_cancel()
        batch = db.query(
            "SELECT * FROM images WHERE project_id=? AND status='pending' ORDER BY created_at LIMIT 8",
            [project_id],
        )
        if not batch:
            break
        remaining = int(db.scalar("SELECT COUNT(*) FROM images WHERE project_id=? AND status='pending'", [project_id]))
        for image in batch:
            ctx.check_cancel()
            ctx.progress(done, done + remaining, image["filename"])
            try:
                status = label_image(image, classes)
            except Exception as exc:
                log.exception("Fallo al etiquetar %s", image["id"])
                ctx.log(f"{image['id']}: {exc}")
                db.update("images", image["id"], {
                    "status": "needs_review",
                    "qc": {**(image["qc"] or {}), "label_reasons": [f"error del etiquetador: {exc}"]},
                })
                status = "needs_review"
            counts[status] = counts.get(status, 0) + 1
            done += 1
            remaining -= 1
    return {"labeled": done, "by_status": counts}


def annotations_for(image_id: str) -> list[dict[str, Any]]:
    return db.query(
        "SELECT a.*, c.name AS class_name, c.idx AS class_idx FROM annotations a "
        "JOIN classes c ON c.id=a.class_id WHERE a.image_id=? ORDER BY a.id",
        [image_id],
    )


def set_boxes(image: dict[str, Any], boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Guarda la correccion humana. SAM recalcula la mascara de cada caja corregida."""
    path = storage.resolve(image["normalized_path"])
    polygons = segment(path, boxes, image["width"], image["height"])
    with db.transaction():
        db.execute("DELETE FROM annotations WHERE image_id=?", [image["id"]])
        for box, polygon in zip(boxes, polygons):
            db.insert("annotations", {
                "image_id": image["id"], "class_id": box["class_id"],
                "cx": box["cx"], "cy": box["cy"], "w": box["w"], "h": box["h"],
                "polygon": polygon, "confidence": None, "source": "humano",
                "mask_source": "mobile-sam" if polygon else None, "verified": 1,
                "created_at": db.now(),
            })
    return annotations_for(image["id"])
