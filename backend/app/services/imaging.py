"""Lectura validada, normalizacion sRGB, miniaturas y medidas basicas de imagen."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .. import config

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_SRGB = ImageCms.createProfile("sRGB")


class InvalidImage(ValueError):
    pass


def open_verified(path: Path) -> Image.Image:
    """Abre la imagen comprobando su contenido real, no su extension."""
    try:
        with Image.open(path) as probe:
            fmt = probe.format
            probe.verify()
        if fmt not in ALLOWED_FORMATS:
            raise InvalidImage(f"Formato no admitido: {fmt}")
        image = Image.open(path)
        image.load()
    except InvalidImage:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImage("Imagen demasiado grande") from exc
    except Exception as exc:
        raise InvalidImage(f"Archivo corrupto o ilegible: {exc}") from exc
    image.format = fmt
    return image


def to_srgb(image: Image.Image) -> Image.Image:
    """Aplica la orientacion EXIF y convierte a RGB en espacio sRGB."""
    icc = image.info.get("icc_profile")
    oriented = ImageOps.exif_transpose(image)
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(oriented.convert("RGB"), src, _SRGB, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError):
            pass
    return oriented.convert("RGB")


def resize_square(image: Image.Image, size: int) -> Image.Image:
    """Lleva a size x size sin deformar, recortando al centro."""
    return ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)


def save_thumb(image: Image.Image, path: Path, max_side: int = 384) -> None:
    thumb = image.convert("RGB")
    thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(path, "WEBP", quality=82, method=4)


def phash(image: Image.Image) -> str:
    """pHash DCT de 64 bits: detecta casi duplicados aunque cambie el formato o el tamano."""
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8]
    bits = (low > np.median(low[1:, 1:])).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def measures(image: Image.Image) -> dict[str, float]:
    """Nitidez, exposicion y contraste para control de calidad."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "dark_ratio": round(float((gray < 16).mean()), 4),
        "bright_ratio": round(float((gray > 240).mean()), 4),
    }


def artifact_flags(m: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if m["sharpness"] < 25:
        flags.append("borrosa")
    if m["dark_ratio"] > 0.7:
        flags.append("subexpuesta")
    if m["bright_ratio"] > 0.7:
        flags.append("sobreexpuesta")
    if m["contrast"] < 12:
        flags.append("sin contraste")
    return flags
