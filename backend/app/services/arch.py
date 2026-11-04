"""Identifica la arquitectura de checkpoints y LoRAs leyendo sus tensores, no el nombre."""
from __future__ import annotations

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

ARCH_LABEL = {"sd15": "SD 1.x", "sdxl": "SDXL", "flux": "Flux", "unknown": "Desconocida"}


def read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        if size > 100_000_000:
            raise ValueError("Cabecera safetensors invalida")
        return json.loads(fh.read(size))


def _checkpoint_arch(keys: list[str]) -> str:
    if any("double_blocks" in k for k in keys):
        return "flux"
    if any(k.startswith("conditioner.embedders.1") for k in keys):
        return "sdxl"
    if any(k.startswith("cond_stage_model.") for k in keys):
        return "sd15"
    return "unknown"


@lru_cache(maxsize=256)
def _checkpoint_cached(path: str, mtime: float) -> str:
    p = Path(path)
    if p.suffix == ".safetensors":
        return _checkpoint_arch([k for k in read_header(p) if k != "__metadata__"])
    # Los .ckpt no se abren: cargar pickle ejecuta codigo. Los de este tamano son SD 1.x.
    return "sd15" if p.stat().st_size < 6 * 1024**3 else "unknown"


def checkpoint_arch(path: Path) -> str:
    try:
        return _checkpoint_cached(str(path), path.stat().st_mtime)
    except (OSError, ValueError, struct.error):
        return "unknown"


def lora_info(path: Path) -> dict[str, Any]:
    """Arquitectura y metadatos de entrenamiento de un LoRA."""
    header = read_header(path)
    meta = header.get("__metadata__") or {}
    keys = [k for k in header if k != "__metadata__"]
    version = (meta.get("ss_base_model_version") or "").lower()
    if "flux" in version or any("double_blocks" in k for k in keys):
        arch = "flux"
    elif "sdxl" in version or any(k.startswith(("lora_te1_", "lora_te2_")) or "input_blocks" in k for k in keys):
        arch = "sdxl"
    elif any(k.startswith(("lora_te_", "lora_unet_down_blocks", "lora_unet_mid_block", "lora_unet_up_blocks")) for k in keys):
        arch = "sd15"
    else:
        arch = "unknown"
    return {
        "arch": arch,
        "base_model": meta.get("ss_sd_model_name"),
        "output_name": meta.get("ss_output_name"),
        "network_dim": meta.get("ss_network_dim"),
        "steps": meta.get("ss_max_train_steps"),
    }
