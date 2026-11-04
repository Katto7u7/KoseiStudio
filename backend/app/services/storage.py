"""Rutas seguras y utilidades de archivo. Todo lo persistente vive bajo DATA_DIR."""
from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .. import config

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
LORA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def project_dir(project_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", project_id):
        raise ValueError("Identificador de proyecto invalido")
    return config.PROJECTS_DIR / project_id


def rel(path: Path) -> str:
    return path.resolve().relative_to(config.DATA_DIR).as_posix()


def resolve(relative: str) -> Path:
    """Convierte una ruta relativa guardada en absoluta y rechaza escapes del directorio."""
    full = (config.DATA_DIR / relative).resolve()
    if config.DATA_DIR != full and config.DATA_DIR not in full.parents:
        raise ValueError("Ruta fuera del directorio de datos")
    return full


def safe_filename(name: str, fallback: str = "archivo") -> str:
    base = Path(name.replace("\\", "/")).name
    cleaned = SAFE_NAME.sub("_", base).strip("._")
    return cleaned[:128] or fallback


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remove_tree(path: Path) -> None:
    resolved = path.resolve()
    if config.DATA_DIR not in resolved.parents:
        raise ValueError("Solo se puede borrar dentro del directorio de datos")
    shutil.rmtree(resolved, ignore_errors=True)
