"""Pruebas con pesos REALES en CPU. Requieren los volumenes:

docker compose run --rm --no-deps backend python -m pytest tests/test_real_models.py -m real
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import config
from app.services import features, fidelity, labeling, provenance

pytestmark = pytest.mark.real

HAS_WEIGHTS = (config.YOLO_DIR / "yolo11m.pt").exists() and (config.YOLO_DIR / "mobile_sam.pt").exists()
skip = pytest.mark.skipif(not HAS_WEIGHTS, reason="Faltan los pesos en los volumenes")


@pytest.fixture(scope="module")
def street() -> Path:
    """Imagen real incluida en ultralytics, con personas y un autobus."""
    import ultralytics
    path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    assert path.exists()
    return path


@skip
def test_detector_filters_to_requested_class(street):
    person = {"name": "persona", "detector": "coco", "detector_prompt": "person"}
    boxes, source = labeling.detect(street, person)
    assert source == "yolo11m-coco"
    assert len(boxes) >= 3
    assert all(0 <= b["cx"] <= 1 and 0 < b["w"] <= 1 for b in boxes)
    assert boxes == sorted(boxes, key=lambda b: -b["confidence"])

    bus = {"name": "autobus", "detector": "coco", "detector_prompt": "bus"}
    boxes_bus, _ = labeling.detect(street, bus)
    assert len(boxes_bus) == 1, "Debe descartar a las personas al pedir la clase autobus"


@skip
def test_detector_finds_nothing_in_empty_image(tmp_path):
    path = tmp_path / "vacia.png"
    Image.new("RGB", (640, 640), (128, 128, 128)).save(path)
    boxes, _ = labeling.detect(path, {"name": "persona", "detector": "coco", "detector_prompt": "person"})
    assert boxes == []


@skip
def test_alias_in_spanish_resolves_to_coco():
    assert labeling.coco_name({"name": "persona", "detector_prompt": ""}) == "person"
    assert labeling.coco_name({"name": "flipper", "detector_prompt": ""}) is None


@skip
def test_sam_produces_valid_polygon_inside_box(street):
    with Image.open(street) as image:
        width, height = image.size
    boxes, _ = labeling.detect(street, {"name": "persona", "detector": "coco", "detector_prompt": "person"})
    polygons = labeling.segment(street, boxes[:1], width, height)
    polygon = polygons[0]
    assert polygon and len(polygon) >= 6 and len(polygon) % 2 == 0
    box = boxes[0]
    xs, ys = polygon[0::2], polygon[1::2]
    margin = 0.05
    assert min(xs) >= box["cx"] - box["w"] / 2 - margin and max(xs) <= box["cx"] + box["w"] / 2 + margin
    assert min(ys) >= box["cy"] - box["h"] / 2 - margin and max(ys) <= box["cy"] + box["h"] / 2 + margin


def _blip_available() -> bool:
    return (config.WEIGHTS_DIR / "hf/hub/models--Salesforce--blip-image-captioning-large").exists()


@pytest.mark.skipif(not _blip_available(), reason="Falta BLIP")
def test_blip_describes_a_real_image(street):
    caption = features.caption_image(street)
    assert len(caption.split()) >= 3
    assert not caption.lower().startswith(("there is", "arafed"))


def test_clean_caption():
    assert features.clean_caption("there is a man with glasses") == "a man with glasses"
    assert features.clean_caption("arafed man in a t - shirt") == "man in a t-shirt"


@pytest.mark.skipif(not fidelity.face_models_available(), reason="Faltan los pesos faciales")
def test_face_embedding_is_stable_and_absent_without_face(street, tmp_path):
    import cv2
    bgr = cv2.imread(str(street))
    first, second = fidelity.face_embedding(bgr), fidelity.face_embedding(bgr)
    assert first is not None and float(first @ second) > 0.999
    empty = np.full((400, 400, 3), 127, dtype=np.uint8)
    assert fidelity.face_embedding(empty) is None


def test_watermark_roundtrip_and_no_false_positive(street):
    base = Image.open(street).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    marked = provenance.embed_watermark(base)
    assert provenance.read_watermark(marked) is True
    assert provenance.read_watermark(base) is False
    mse = np.mean((np.asarray(marked, dtype=np.float64) - np.asarray(base, dtype=np.float64)) ** 2)
    psnr = 10 * np.log10(255**2 / mse)
    print(f"PSNR de la marca: {psnr:.1f} dB")
    assert psnr > 38, f"Marca demasiado visible: {psnr:.1f} dB"
