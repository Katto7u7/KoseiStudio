"""Configuracion por variables de entorno. Rutas por defecto para desarrollo local."""
from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def _path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).resolve()


FORGE_URL = os.getenv("FORGE_URL", "http://127.0.0.1:7860").rstrip("/")
DATA_DIR = _path("DATA_DIR", BASE / "_data")
LORA_DIR = _path("LORA_DIR", BASE.parent / "volumes/models/Lora")
CHECKPOINT_DIR = _path("CHECKPOINT_DIR", BASE.parent / "volumes/models/Stable-diffusion")
YOLO_DIR = _path("YOLO_DIR", BASE.parent / "volumes/yolo")
WEIGHTS_DIR = _path("WEIGHTS_DIR", Path("/weights") if Path("/weights").exists() else BASE.parent / "volumes/weights")

KOHYA_DIR = _path("KOHYA_DIR", Path("/opt/sd-scripts"))
KOHYA_PYTHON = os.getenv("KOHYA_PYTHON", "/opt/kohya/bin/python")

DB_PATH = DATA_DIR / "kosei.db"
PROJECTS_DIR = DATA_DIR / "projects"
TMP_DIR = DATA_DIR / "tmp"

# Limites de seguridad para lo que entra por la API.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024**3)))
MAX_ZIP_ENTRIES = int(os.getenv("MAX_ZIP_ENTRIES", "20000"))
MAX_ZIP_UNCOMPRESSED = int(os.getenv("MAX_ZIP_UNCOMPRESSED", str(20 * 1024**3)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(64_000_000)))

# Umbrales del etiquetado. 0.80 es el umbral que describe la tesis.
LABEL_ACCEPT_CONF = float(os.getenv("LABEL_ACCEPT_CONF", "0.80"))
LABEL_MIN_CONF = float(os.getenv("LABEL_MIN_CONF", "0.25"))

FORGE_TIMEOUT = float(os.getenv("FORGE_TIMEOUT", "1800"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, PROJECTS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
