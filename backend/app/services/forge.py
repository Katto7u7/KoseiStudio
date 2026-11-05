"""Cliente del motor de generacion (Stable Diffusion WebUI Forge)."""
from __future__ import annotations

import base64
import io
import json
from typing import Any, Optional

import httpx
from PIL import Image

from .. import config
from . import arch as arch_mod

# Valores de partida por arquitectura. Flux ignora el negativo y guia con distilled_cfg.
ARCH_DEFAULTS: dict[str, dict[str, Any]] = {
    "sd15": {"width": 512, "height": 512, "steps": 28, "cfg_scale": 6.5, "sampler": "DPM++ 2M", "scheduler": "karras", "batch_size": 4},
    "sdxl": {"width": 1024, "height": 1024, "steps": 30, "cfg_scale": 5.0, "sampler": "DPM++ 2M", "scheduler": "karras", "batch_size": 1},
    "flux": {"width": 1024, "height": 1024, "steps": 24, "cfg_scale": 1.0, "distilled_cfg_scale": 3.5, "sampler": "Euler", "scheduler": "simple", "batch_size": 1},
}
ARCH_DEFAULTS["unknown"] = ARCH_DEFAULTS["sd15"]

FORGE_PRESET = {"sd15": "sd", "sdxl": "xl", "flux": "flux", "unknown": "all"}


class ForgeError(RuntimeError):
    pass


class ForgeUnavailable(ForgeError):
    pass


def _client(timeout: float = 15.0) -> httpx.Client:
    return httpx.Client(base_url=config.FORGE_URL, timeout=timeout)


def _get(path: str, timeout: float = 15.0) -> Any:
    try:
        with _client(timeout) as c:
            r = c.get(path)
            r.raise_for_status()
            return r.json()
    except httpx.TransportError as exc:
        raise ForgeUnavailable("El motor de generacion aun no responde") from exc
    except httpx.HTTPStatusError as exc:
        raise ForgeError(f"Forge respondio {exc.response.status_code} en {path}") from exc


def _post(path: str, payload: Optional[dict] = None, timeout: float = 60.0) -> Any:
    try:
        with _client(timeout) as c:
            r = c.post(path, json=payload or {})
            if r.status_code >= 400:
                raise ForgeError(f"Forge respondio {r.status_code}: {r.text[:800]}")
            return r.json() if r.content else {}
    except httpx.TransportError as exc:
        raise ForgeUnavailable("Se perdio la conexion con el motor de generacion") from exc


def is_ready() -> bool:
    try:
        _get("/sdapi/v1/sd-models", timeout=4)
        return True
    except ForgeError:
        return False


def models() -> list[dict[str, Any]]:
    """Checkpoints que Forge ve, con la arquitectura leida de sus tensores."""
    out = []
    for item in _get("/sdapi/v1/sd-models"):
        filename = item.get("filename") or ""
        path = config.CHECKPOINT_DIR / filename.rsplit("/", 1)[-1]
        detected = arch_mod.checkpoint_arch(path)
        out.append({
            "title": item["title"],
            "name": item["model_name"],
            "arch": detected,
            "arch_label": arch_mod.ARCH_LABEL[detected],
            "defaults": ARCH_DEFAULTS[detected],
        })
    return out


def samplers() -> list[str]:
    return [s["name"] for s in _get("/sdapi/v1/samplers")]


def schedulers() -> list[str]:
    return [s["name"] for s in _get("/sdapi/v1/schedulers")]


def refresh() -> None:
    _post("/sdapi/v1/refresh-checkpoints")
    _post("/sdapi/v1/refresh-loras")


def current_checkpoint() -> str:
    return _get("/sdapi/v1/options").get("sd_model_checkpoint") or ""


def ensure_checkpoint(title: str, arch: str) -> None:
    """Forge guarda solo el nombre del archivo o el titulo completo segun la ruta usada."""
    current = current_checkpoint()
    short = title.split(" [")[0]
    if current == title or current.split(" [")[0] == short:
        return
    _post("/sdapi/v1/options", {
        "sd_model_checkpoint": title,
        "forge_preset": FORGE_PRESET.get(arch, "all"),
        "forge_additional_modules": [],
    }, timeout=config.FORGE_TIMEOUT)


def txt2img(payload: dict[str, Any]) -> tuple[list[Image.Image], dict[str, Any]]:
    """Genera sin guardar en el disco de Forge. Devuelve imagenes y el `info` estructurado."""
    body = {
        "save_images": False,
        "send_images": True,
        "do_not_save_samples": True,
        "do_not_save_grid": True,
        **payload,
    }
    data = _post("/sdapi/v1/txt2img", body, timeout=config.FORGE_TIMEOUT)
    info = data.get("info") or "{}"
    info = json.loads(info) if isinstance(info, str) else info
    images = [Image.open(io.BytesIO(base64.b64decode(b.split(",", 1)[-1]))) for b in data.get("images", [])]
    for image in images:
        image.load()
    return images, info


def img2img(payload: dict[str, Any], init: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    init.convert("RGB").save(buf, "PNG")
    body = {
        "init_images": [base64.b64encode(buf.getvalue()).decode()],
        "save_images": False, "send_images": True,
        "do_not_save_samples": True, "do_not_save_grid": True,
        **payload,
    }
    data = _post("/sdapi/v1/img2img", body, timeout=config.FORGE_TIMEOUT)
    image = Image.open(io.BytesIO(base64.b64decode(data["images"][0].split(",", 1)[-1])))
    image.load()
    return image


def interrupt() -> None:
    _post("/sdapi/v1/interrupt", timeout=10)


def progress() -> dict[str, Any]:
    return _get("/sdapi/v1/progress?skip_current_image=false", timeout=8)


def unload() -> None:
    """Saca el modelo de la VRAM para dejar sitio a un entrenamiento."""
    _post("/sdapi/v1/unload-checkpoint", timeout=120)


def memory() -> dict[str, Any]:
    return _get("/sdapi/v1/memory", timeout=8)
