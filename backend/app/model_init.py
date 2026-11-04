"""Descarga lo que falte y termina. No sobrescribe nada de lo que ya existe.

Se ejecuta como servicio de una sola vez en `docker compose up`. Sin internet solo falla
si falta un archivo obligatorio. Lo opcional que no se pueda descargar queda como aviso.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

MODELS = Path(os.getenv("MODELS_DIR", "/models"))
YOLO = Path(os.getenv("YOLO_DIR", "/yolo"))
WEIGHTS = Path(os.getenv("WEIGHTS_DIR", "/weights"))
HF = "https://huggingface.co"
ULTRA = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

FILES: list[tuple[Path, str, bool]] = [
    (MODELS / "Stable-diffusion/v1-5-pruned-emaonly.safetensors",
     f"{HF}/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors", True),
    (MODELS / "Stable-diffusion/RealVisXL_V5.0_fp16.safetensors",
     f"{HF}/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors", True),
    (YOLO / "yolo11m.pt", f"{ULTRA}/yolo11m.pt", True),
    (YOLO / "yolo11n.pt", f"{ULTRA}/yolo11n.pt", True),
    (YOLO / "mobile_sam.pt", f"{ULTRA}/mobile_sam.pt", True),
    (YOLO / "yolov8s-worldv2.pt", f"{ULTRA}/yolov8s-worldv2.pt", False),
    (WEIGHTS / "clip/ViT-B-32.pt",
     "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt", False),
    (WEIGHTS / "torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth",
     "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth", True),
]

if os.getenv("DOWNLOAD_FLUX", "0") == "1":
    FILES += [
        (MODELS / "Stable-diffusion/flux1-dev-bnb-nf4-v2.safetensors",
         f"{HF}/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors", False),
    ]

HF_SNAPSHOTS = [
    ("Salesforce/blip-image-captioning-large", True),
    ("openai/clip-vit-base-patch32", True),
]


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if resp.status == 206 else "wb"
        total = int(resp.headers.get("Content-Length") or 0) + (offset if mode == "ab" else 0)
        written = offset if mode == "ab" else 0
        last = time.time()
        with part.open(mode) as fh:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                fh.write(chunk)
                written += len(chunk)
                if time.time() - last > 15:
                    pct = f"{100 * written / total:.0f} %" if total else f"{written >> 20} MiB"
                    print(f"  {dest.name}: {pct}", flush=True)
                    last = time.time()
    part.rename(dest)


def main() -> int:
    missing_required: list[str] = []
    for dest, url, required in FILES:
        if dest.exists():
            print(f"ok      {dest}", flush=True)
            continue
        print(f"bajando {dest}", flush=True)
        try:
            download(url, dest)
            print(f"listo   {dest}", flush=True)
        except Exception as exc:
            print(f"ERROR   {dest}: {exc}", flush=True)
            if required:
                missing_required.append(dest.name)

    os.environ.setdefault("HF_HOME", str(WEIGHTS / "hf"))
    from huggingface_hub import snapshot_download
    for repo, required in HF_SNAPSHOTS:
        try:
            snapshot_download(repo, local_files_only=True, ignore_patterns=["*.h5", "*.msgpack", "*.ot", "tf_*", "*.bin.index*"])
            print(f"ok      {repo}", flush=True)
            continue
        except Exception:
            pass
        print(f"bajando {repo}", flush=True)
        try:
            snapshot_download(repo, ignore_patterns=["*.h5", "*.msgpack", "*.ot", "tf_*", "*.bin.index*"])
            print(f"listo   {repo}", flush=True)
        except Exception as exc:
            print(f"ERROR   {repo}: {exc}", flush=True)
            if required:
                missing_required.append(repo)

    if missing_required:
        print("Faltan archivos obligatorios y no se pudieron descargar: " + ", ".join(missing_required), flush=True)
        print("Comprueba la conexion a internet y vuelve a ejecutar: docker compose up -d", flush=True)
        return 1
    print("Modelos listos.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
