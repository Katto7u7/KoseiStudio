"""Procedencia: registro estructurado en el PNG y marca de agua invisible verificable."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("kosei.provenance")

WATERMARK_TEXT = b"KoseiSyn"
WATERMARK_BITS = len(WATERMARK_TEXT) * 8


def record(generation: dict[str, Any]) -> dict[str, Any]:
    return {"synthetic": True, "software": "KoseiStudio", "generation": generation}


def embed_watermark(image: Image.Image, _project_id: str = "") -> Image.Image:
    """Marca DWT-DCT. Si la imagen es demasiado pequena para el metodo, se deja sin marca."""
    try:
        from imwatermark import WatermarkEncoder
        if min(image.size) < 256:
            return image
        encoder = WatermarkEncoder()
        encoder.set_watermark("bytes", WATERMARK_TEXT)
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        marked = encoder.encode(bgr, "dwtDct")
        return Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    except Exception as exc:
        log.warning("No se pudo aplicar la marca de agua: %s", exc)
        return image


def read_watermark(image: Image.Image) -> bool:
    """True si se recupera la marca exacta."""
    try:
        from imwatermark import WatermarkDecoder
        decoder = WatermarkDecoder("bytes", WATERMARK_BITS)
        bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return bytes(decoder.decode(bgr, "dwtDct")) == WATERMARK_TEXT
    except Exception as exc:
        log.warning("No se pudo leer la marca de agua: %s", exc)
        return False


def read_record(image: Image.Image) -> Optional[dict[str, Any]]:
    raw = getattr(image, "text", {}).get("kosei_provenance") if hasattr(image, "text") else image.info.get("kosei_provenance")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None
