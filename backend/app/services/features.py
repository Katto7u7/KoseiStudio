"""Extractor de caracteristicas. Cada salida tiene un uso concreto:

- Captions BLIP: guia semantica del entrenamiento del LoRA.
- Embeddings CLIP: casi duplicados semanticos, similitud sintético-real y mapa de cobertura.
- Medidas de imagen: control de calidad, ya calculadas al importar.
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .. import db
from ..jobs import JobContext, register
from . import storage

log = logging.getLogger("kosei.features")

BLIP_MODEL = "Salesforce/blip-image-captioning-large"
CLIP_MODEL = "openai/clip-vit-base-patch32"

_lock = threading.Lock()
_cache: dict[str, Any] = {}


def _blip():
    with _lock:
        if "blip" not in _cache:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            _cache["blip"] = (
                BlipProcessor.from_pretrained(BLIP_MODEL),
                BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).eval(),
            )
        return _cache["blip"]


def _clip():
    with _lock:
        if "clip" not in _cache:
            from transformers import CLIPModel, CLIPProcessor
            _cache["clip"] = (
                CLIPProcessor.from_pretrained(CLIP_MODEL),
                CLIPModel.from_pretrained(CLIP_MODEL).eval(),
            )
        return _cache["clip"]


# Muletillas de BLIP que no describen nada y compiten con el token.
_FILLER = re.compile(r"^(there (is|are)|arafed|araffe|someone is|this is|it is)\s+", re.IGNORECASE)


def clean_caption(text: str) -> str:
    text = text.strip()
    while True:
        cleaned = _FILLER.sub("", text)
        if cleaned == text:
            return re.sub(r"\s+t - shirt", " t-shirt", text)
        text = cleaned


def caption_image(path: Path) -> str:
    import torch
    processor, model = _blip()
    with Image.open(path) as image:
        inputs = processor(image.convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        ids = model.generate(**inputs, max_length=60, min_length=8, num_beams=3)
    return clean_caption(processor.decode(ids[0], skip_special_tokens=True))


@register("caption", "cpu")
def run_captions(ctx: JobContext) -> dict[str, Any]:
    """Solo describe el split de entrenamiento: es el único que ve el LoRA."""
    overwrite = bool(ctx.params.get("overwrite"))
    token = (ctx.params.get("token") or "").strip()
    images = db.query(
        "SELECT id, filename, normalized_path, caption FROM images "
        "WHERE project_id=? AND origin='real' AND split='train' AND status!='rejected' ORDER BY filename",
        [ctx.project_id],
    )
    if not images:
        raise ValueError("Primero define la partición: las captions se generan solo para train")
    if not overwrite:
        images = [i for i in images if not i["caption"]]
    for n, image in enumerate(images, 1):
        ctx.check_cancel()
        ctx.progress(n, len(images), image["filename"])
        caption = caption_image(storage.resolve(image["normalized_path"]))
        db.update("images", image["id"], {"caption": f"{token} {caption}".strip()})
    return {"captioned": len(images)}


def embeddings_path(project_id: str) -> Path:
    return storage.project_dir(project_id) / "features" / "clip.npz"


def load_embeddings(project_id: str) -> dict[str, np.ndarray]:
    path = embeddings_path(project_id)
    if not path.exists():
        return {}
    with np.load(path) as data:
        return dict(zip(data["ids"].tolist(), data["vectors"]))


def clip_vectors(paths: list[Path], batch: int = 16) -> np.ndarray:
    """Embeddings CLIP normalizados, sin guardarlos en disco."""
    import torch
    processor, model = _clip()
    out = []
    for start in range(0, len(paths), batch):
        pics = [Image.open(p).convert("RGB") for p in paths[start:start + batch]]
        inputs = processor(images=pics, return_tensors="pt")
        with _lock, torch.inference_mode():
            vectors = model.get_image_features(**inputs)
        out.append(torch.nn.functional.normalize(vectors, dim=-1).numpy())
    return np.concatenate(out)


def embed_images(ctx: JobContext, images: list[dict[str, Any]], batch: int = 16) -> dict[str, np.ndarray]:
    """Calcula solo lo que falta y guarda incrementalmente."""
    import torch
    known = load_embeddings(ctx.project_id)
    todo = [i for i in images if i["id"] not in known]
    if todo:
        processor, model = _clip()
    for start in range(0, len(todo), batch):
        ctx.check_cancel()
        chunk = todo[start:start + batch]
        ctx.progress(start + len(chunk), len(todo), "Calculando embeddings CLIP")
        pics = [Image.open(storage.resolve(i["normalized_path"])).convert("RGB") for i in chunk]
        inputs = processor(images=pics, return_tensors="pt")
        with torch.inference_mode():
            vectors = model.get_image_features(**inputs)
        vectors = torch.nn.functional.normalize(vectors, dim=-1).numpy()
        for image, vector in zip(chunk, vectors):
            known[image["id"]] = vector
    if todo:
        path = embeddings_path(ctx.project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        ids = list(known)
        np.savez_compressed(path, ids=np.array(ids), vectors=np.stack([known[i] for i in ids]))
    return known


def embed_text(prompts: list[str]) -> np.ndarray:
    import torch
    processor, model = _clip()
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        vectors = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(vectors, dim=-1).numpy()


def pca_2d(vectors: np.ndarray) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


@register("features", "cpu")
def run_features(ctx: JobContext) -> dict[str, Any]:
    images = db.query(
        "SELECT id, filename, normalized_path, origin, split, phash FROM images "
        "WHERE project_id=? AND status!='rejected' ORDER BY created_at",
        [ctx.project_id],
    )
    if not images:
        raise ValueError("El proyecto no tiene imágenes")
    known = embed_images(ctx, images)
    ids = [i["id"] for i in images]
    matrix = np.stack([known[i] for i in ids])
    points = pca_2d(matrix) if len(ids) >= 3 else np.zeros((len(ids), 2))
    scale = float(np.abs(points).max() or 1.0)

    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1)
    semantic = []
    for i, image in enumerate(images):
        j = int(sims[i].argmax())
        if sims[i, j] > 0.985 and i < j:
            semantic.append({
                "a": image["filename"], "b": images[j]["filename"],
                "a_split": image["split"], "b_split": images[j]["split"],
                "similarity": round(float(sims[i, j]), 4),
            })
    return {
        "count": len(ids),
        "points": [
            {"id": i["id"], "x": round(float(x / scale), 4), "y": round(float(y / scale), 4), "origin": i["origin"], "split": i["split"]}
            for i, (x, y) in zip(images, points)
        ],
        "semantic_near_duplicates": semantic[:100],
    }
