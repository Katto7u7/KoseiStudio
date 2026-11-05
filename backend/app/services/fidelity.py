"""Fidelidad al sujeto: cuanto se parece lo generado al sujeto real de entrenamiento.

Dos medidas complementarias, siempre contra el split de train:

- Rostro (SFace): compara identidad facial. Cuando hay rostro es la evidencia mas directa
  de "es la misma persona". Se promedian los 3 pares mas parecidos, lo que sube los valores
  frente al umbral publicado para un solo par (0.363). Por eso se usa 0.45, calibrado con una
  muestra pequeña: 10 fotos reales nunca vistas dieron 0.57 a 0.89 y 7 personas ajenas 0.06 a 0.36.
- CLIP: parecido visual general. Sirve también para objetos, pero no prueba identidad.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .. import config, db
from . import storage

log = logging.getLogger("kosei.fidelity")

SAME_PERSON_THRESHOLD = 0.45
TOP_K = 3

_lock = threading.RLock()
_detector = None
_recognizer = None
_references: dict[str, dict[str, Any]] = {}


def face_models_available() -> bool:
    return (config.WEIGHTS_DIR / "face/face_detection_yunet_2023mar.onnx").exists() and (
        config.WEIGHTS_DIR / "face/face_recognition_sface_2021dec.onnx").exists()


def _face_models():
    global _detector, _recognizer
    with _lock:
        if _detector is None:
            _detector = cv2.FaceDetectorYN.create(
                str(config.WEIGHTS_DIR / "face/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.7, 0.3, 50,
            )
            _recognizer = cv2.FaceRecognizerSF.create(
                str(config.WEIGHTS_DIR / "face/face_recognition_sface_2021dec.onnx"), "",
            )
        return _detector, _recognizer


def detect_faces(bgr: np.ndarray) -> np.ndarray:
    """Filas YuNet: x, y, w, h, puntos faciales, confianza. Vacio si no hay rostro."""
    detector, _ = _face_models()
    with _lock:
        height, width = bgr.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(bgr)
    return faces if faces is not None else np.empty((0, 15))


def largest_face(bgr: np.ndarray) -> Optional[np.ndarray]:
    faces = detect_faces(bgr)
    if not len(faces):
        return None
    return faces[np.argmax(faces[:, 2] * faces[:, 3])]


def face_embedding(bgr: np.ndarray) -> Optional[np.ndarray]:
    face = largest_face(bgr)
    if face is None:
        return None
    _, recognizer = _face_models()
    with _lock:
        aligned = recognizer.alignCrop(bgr, face)
        feature = recognizer.feature(aligned).flatten()
    norm = np.linalg.norm(feature)
    return feature / norm if norm else None


def _top_k_mean(similarities: np.ndarray) -> float:
    k = min(TOP_K, len(similarities))
    return float(np.sort(similarities)[-k:].mean())


def _train_images(project_id: str) -> list[dict[str, Any]]:
    return db.query(
        "SELECT id, normalized_path, sha256 FROM images WHERE project_id=? AND origin='real' AND split='train' "
        "AND status!='rejected' ORDER BY sha256",
        [project_id],
    )


def reference(project_id: str) -> Optional[dict[str, Any]]:
    """Embeddings del split de train. Se cachean mientras ese conjunto no cambie."""
    from . import features

    images = _train_images(project_id)
    if not images:
        return None
    key = ",".join(i["sha256"][:12] for i in images)
    with _lock:
        cached = _references.get(project_id)
        if cached and cached["key"] == key:
            return cached
    clip = features.clip_vectors([storage.resolve(i["normalized_path"]) for i in images])
    faces: list[np.ndarray] = []
    if face_models_available():
        for image in images:
            bgr = cv2.imread(str(storage.resolve(image["normalized_path"])))
            if bgr is not None:
                embedding = face_embedding(bgr)
                if embedding is not None:
                    faces.append(embedding)
    baseline = None
    if len(faces) >= 2:
        # Parecido tipico entre fotos reales del mismo sujeto: el techo razonable del LoRA.
        matrix = np.stack(faces)
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -1)
        baseline = round(float(np.mean([_top_k_mean(row[row > -1]) for row in sims])), 4)
    ref = {
        "key": key, "clip": clip, "faces": np.stack(faces) if faces else None,
        "baseline_face": baseline, "train_images": len(images), "train_faces": len(faces),
    }
    with _lock:
        _references[project_id] = ref
    return ref


def score(project_id: str, path: Path) -> Optional[dict[str, Any]]:
    """Fidelidad de una imagen. None si aún no hay imágenes reales de referencia."""
    from . import features

    ref = reference(project_id)
    if ref is None:
        return None
    out: dict[str, Any] = {"clip": round(_top_k_mean(ref["clip"] @ features.clip_vectors([path])[0]), 4), "face": None}
    if ref["faces"] is not None:
        bgr = cv2.imread(str(path))
        if bgr is not None:
            embedding = face_embedding(bgr)
            if embedding is not None:
                out["face"] = round(_top_k_mean(ref["faces"] @ embedding), 4)
                out["same_person"] = bool(out["face"] >= SAME_PERSON_THRESHOLD)
    return out


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    faces = [s["face"] for s in scores if s.get("face") is not None]
    clips = [s["clip"] for s in scores if s.get("clip") is not None]
    return {
        "images": len(scores),
        "clip_mean": round(float(np.mean(clips)), 4) if clips else None,
        "face_found": len(faces),
        "face_mean": round(float(np.mean(faces)), 4) if faces else None,
        "face_p90": round(float(np.percentile(faces, 90)), 4) if faces else None,
        "same_person_rate": round(float(np.mean([f >= SAME_PERSON_THRESHOLD for f in faces])), 4) if faces else None,
    }
