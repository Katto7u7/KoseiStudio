"""Entrenamiento real de LoRA con kohya sd-scripts como subproceso.

El dataset del LoRA sale SOLO del split de entrenamiento y se registran sus hashes:
asi se puede demostrar que val y test nunca influyeron en el modelo generativo.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

from .. import config, db
from ..jobs import Cancelled, JobContext, register
from . import arch as arch_mod
from . import fidelity, forge, gpu, storage

log = logging.getLogger("kosei.training")

SCRIPTS = {"sd15": "train_network.py", "sdxl": "sdxl_train_network.py", "flux": "flux_train_network.py"}
REQUIRED_VRAM_MIB = {"sd15": 4000, "sdxl": 8500, "flux": 9500}

STEP_RE = re.compile(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)")
LOSS_RE = re.compile(r"avr_loss=([0-9.eE+-]+)")

# Componentes de Flux que kohya necesita por separado.
FLUX_DIR = config.WEIGHTS_DIR / "flux"
FLUX_FILES = {"clip_l": "clip_l.safetensors", "t5xxl": "t5xxl_fp16.safetensors", "ae": "ae.safetensors"}
FLUX_UNET = config.CHECKPOINT_DIR / "flux1-dev-fp8.safetensors"


def parse_progress(line: str) -> Optional[tuple[int, int, Optional[float]]]:
    """Solo la barra `steps:` es progreso real; otras barras de tqdm son de cache."""
    match = STEP_RE.search(line)
    if not match:
        return None
    loss = LOSS_RE.search(line)
    return int(match.group(1)), int(match.group(2)), float(loss.group(1)) if loss else None


def flux_ready() -> tuple[bool, str]:
    missing = [f for f in FLUX_FILES.values() if not (FLUX_DIR / f).exists()]
    if not FLUX_UNET.exists():
        missing.append(FLUX_UNET.name)
    if missing:
        return False, "Faltan componentes de Flux: " + ", ".join(missing)
    return True, ""


def build_command(arch: str, checkpoint: Path, dataset: Path, output: Path, logs: Path, cfg: dict[str, Any]) -> list[str]:
    lr = str(cfg["learning_rate"])
    cmd = [
        config.KOHYA_PYTHON, "-m", "accelerate.commands.launch",
        "--num_processes", "1", "--num_machines", "1", "--dynamo_backend", "no",
        "--mixed_precision", "bf16" if arch == "flux" else "fp16",
        "--num_cpu_threads_per_process", "4",
        str(config.KOHYA_DIR / SCRIPTS[arch]),
        "--pretrained_model_name_or_path", str(checkpoint),
        "--train_data_dir", str(dataset),
        "--output_dir", str(output),
        "--output_name", cfg["name"],
        "--logging_dir", str(logs),
        "--save_model_as", "safetensors",
        "--max_train_steps", str(cfg["steps"]),
        "--save_every_n_steps", str(cfg["save_every"]),
        "--train_batch_size", str(cfg["batch_size"]),
        "--network_dim", str(cfg["network_dim"]),
        "--network_alpha", str(cfg["network_alpha"]),
        "--learning_rate", lr,
        "--unet_lr", lr,
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", str(int(cfg["steps"]) // 20),
        "--optimizer_type", "AdamW8bit",
        "--max_grad_norm", "1",
        "--seed", str(cfg["seed"]),
        "--resolution", f"{cfg['resolution']},{cfg['resolution']}",
        "--enable_bucket", "--bucket_no_upscale",
        "--min_bucket_reso", "256", "--max_bucket_reso", str(max(1024, int(cfg["resolution"]))),
        "--bucket_reso_steps", "64",
        "--cache_latents", "--cache_latents_to_disk",
        "--gradient_checkpointing", "--sdpa",
        "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
        "--caption_extension", ".txt",
        "--mixed_precision", "bf16" if arch == "flux" else "fp16",
        "--save_precision", "bf16" if arch == "flux" else "fp16",
    ]
    if arch == "sd15":
        cmd += ["--network_module", "networks.lora", "--text_encoder_lr", str(float(cfg["learning_rate"]) / 2), "--clip_skip", "1"]
    elif arch == "sdxl":
        # Con 12 GB se entrena solo el UNet y se cachean los codificadores de texto.
        cmd += [
            "--network_module", "networks.lora", "--network_train_unet_only",
            "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
            "--no_half_vae",
        ]
    elif arch == "flux":
        cmd += [
            "--network_module", "networks.lora_flux", "--network_train_unet_only",
            "--clip_l", str(FLUX_DIR / FLUX_FILES["clip_l"]),
            "--t5xxl", str(FLUX_DIR / FLUX_FILES["t5xxl"]),
            "--ae", str(FLUX_DIR / FLUX_FILES["ae"]),
            "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
            "--fp8_base", "--blocks_to_swap", "16",
            "--guidance_scale", "1.0", "--timestep_sampling", "flux_shift",
            "--model_prediction_type", "raw", "--discrete_flow_shift", "3.1582",
        ]
    return cmd


def checkpoint_path(title: str) -> Path:
    """Ruta real del checkpoint a partir del titulo de Forge, sin permitir salir del directorio."""
    name = title.split(" [")[0]
    path = (config.CHECKPOINT_DIR / name).resolve()
    if config.CHECKPOINT_DIR not in path.parents or not path.exists():
        raise FileNotFoundError(f"No se encuentra el checkpoint {name}")
    return path


def prepare_dataset(ctx: JobContext, images: list[dict[str, Any]], root: Path, token: str, repeats: int) -> list[str]:
    folder = root / f"{repeats}_{token}"
    if root.exists():
        shutil.rmtree(root)
    folder.mkdir(parents=True)
    hashes: list[str] = []
    for image in images:
        src = storage.resolve(image["normalized_path"])
        shutil.copy2(src, folder / f"{image['id']}.png")
        caption = (image["caption"] or "").strip() or token
        if token.lower() not in caption.lower():
            caption = f"{token} {caption}"
        (folder / f"{image['id']}.txt").write_text(caption, encoding="utf-8")
        hashes.append(image["sha256"])
    ctx.log(f"Dataset: {len(images)} imagenes del split train en {folder}")
    return hashes


def read_losses(lora_id: str, project_id: str, limit: int = 400) -> list[dict[str, float]]:
    path = storage.project_dir(project_id) / "loras" / lora_id / "loss.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(rows) > limit:
        stride = len(rows) / limit
        rows = [rows[int(i * stride)] for i in range(limit)] + [rows[-1]]
    return rows


def _humanize_failure(tail: list[str], code: int) -> str:
    text = "\n".join(tail)
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "La GPU se quedo sin memoria. Baja la resolucion, el lote o la dimension de red."
    if "No data found" in text or "No images" in text:
        return "kohya no encontro imagenes en el dataset."
    return f"kohya termino con codigo {code}. Ultimas lineas:\n" + "\n".join(tail[-12:])


@register("train_lora", "gpu")
def run_training(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    lora_id, project_id, arch = p["lora_id"], ctx.project_id, p["arch"]
    name = p["name"]
    base = storage.project_dir(project_id) / "loras" / lora_id
    output, logs = base / "output", base / "log"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    loss_path = base / "loss.jsonl"

    images = db.query(
        "SELECT * FROM images WHERE project_id=? AND origin='real' AND split='train' AND status!='rejected'",
        [project_id],
    )
    if not images:
        raise ValueError("No hay imagenes reales en el split de entrenamiento")

    hashes = prepare_dataset(ctx, images, base / "dataset", p["token"], int(p["repeats"]))
    db.update("loras", lora_id, {"train_hashes": hashes, "status": "training"})

    if arch == "flux":
        ready, why = flux_ready()
        if not ready:
            raise FileNotFoundError(why)
        checkpoint = FLUX_UNET
    else:
        checkpoint = checkpoint_path(p["checkpoint"])

    ctx.progress(0, int(p["steps"]), "Liberando VRAM del motor")
    state = gpu.free_for_training(REQUIRED_VRAM_MIB[arch], ctx.log)
    ctx.log(f"VRAM libre al iniciar: {state['memory_free_mib']} MiB")

    cmd = build_command(arch, checkpoint, base / "dataset", output, logs, {**p, "name": name})
    ctx.log("Comando: " + " ".join(cmd))
    ctx.progress(0, int(p["steps"]), "Preparando entrenamiento")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(config.KOHYA_DIR), "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen(
        cmd, cwd=str(config.KOHYA_DIR), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def terminate() -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            threading.Timer(15, _kill_group, args=(proc,)).start()

    ctx.on_cancel(terminate)
    tail: list[str] = []
    last_step = -1
    buffer = b""
    try:
        with loss_path.open("a", encoding="utf-8") as losses:
            while True:
                chunk = proc.stdout.read1(4096)  # type: ignore[union-attr]
                if not chunk:
                    break
                buffer += chunk
                parts = re.split(rb"[\r\n]", buffer)
                buffer = parts.pop()
                for raw in parts:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    parsed = parse_progress(line)
                    if parsed:
                        step, total, loss = parsed
                        if step != last_step:
                            last_step = step
                            if loss is not None:
                                losses.write(json.dumps({"step": step, "loss": loss}) + "\n")
                                losses.flush()
                            detail = f"Paso {step} de {total}" + (f", pérdida {loss:.4f}" if loss is not None else "")
                            ctx.progress(step, total, detail)
                    else:
                        tail = (tail + [line])[-120:]
                        ctx.log(line)
        code = proc.wait()
    finally:
        ctx.on_cancel(None)

    if ctx.cancelled():
        db.update("loras", lora_id, {"status": "cancelled"})
        raise Cancelled()
    if code != 0:
        db.update("loras", lora_id, {"status": "failed"})
        raise RuntimeError(_humanize_failure(tail, code))

    final = output / f"{name}.safetensors"
    if not final.exists():
        db.update("loras", lora_id, {"status": "failed"})
        raise RuntimeError("kohya termino sin producir el archivo del LoRA")
    info = arch_mod.lora_info(final)
    if info["arch"] not in (arch, "unknown"):
        db.update("loras", lora_id, {"status": "failed"})
        raise RuntimeError(f"El LoRA resultante es {info['arch']}, se esperaba {arch}")

    dest = config.LORA_DIR / final.name
    if dest.exists():
        raise FileExistsError(f"Ya existe {dest.name}; no se sobrescribe")
    shutil.copy2(final, dest)
    try:
        forge.refresh()
    except forge.ForgeError as exc:
        ctx.log(f"Forge no pudo refrescar su lista de LoRAs: {exc}")
    checkpoints = sorted(p_.name for p_ in output.glob(f"{name}-*.safetensors"))
    db.update("loras", lora_id, {"status": "ready"})
    return {"lora_id": lora_id, "file": dest.name, "size_bytes": dest.stat().st_size, "checkpoints": checkpoints, "images": len(hashes)}


EVAL_PROMPTS = [
    "{token} {subject}, portrait photograph, soft daylight, looking at camera",
    "{token} {subject}, medium shot, sitting at a cafe table, natural window light",
    "{token} {subject}, full body photograph, walking on a city street at dusk",
    "{token} {subject}, candid photograph in an office, indoor lighting",
]


def lora_files(lora: dict[str, Any]) -> list[Path]:
    output = storage.project_dir(lora["project_id"]) / "loras" / lora["id"] / "output"
    final_step = f"step{int((lora['config'] or {}).get('steps') or 0):08d}"
    # El punto de control del ultimo paso es el mismo modelo que el archivo final.
    files = [p for p in output.glob("*.safetensors") if not p.stem.endswith(final_step)]
    return sorted(files, key=lambda p: (p.stem == lora["name"], p.stem))


@register("lora_eval", "gpu")
def run_lora_eval(ctx: JobContext) -> dict[str, Any]:
    """Genera las mismas escenas con cada punto de control y mide el parecido al sujeto."""
    p = ctx.params
    lora = db.one("SELECT * FROM loras WHERE id=?", [p["lora_id"]])
    files = lora_files(lora)
    if not files:
        raise ValueError("Este LoRA no tiene archivos locales que evaluar")
    arch = lora["arch"]
    defaults = forge.ARCH_DEFAULTS[arch]
    base = storage.project_dir(lora["project_id"]) / "loras" / lora["id"] / "eval"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    # Forge lee LoRAs recursivamente: una subcarpeta temporal no contamina el registro.
    staging = config.LORA_DIR / f"_eval_{lora['id']}"
    staging.mkdir(exist_ok=True)
    prompts = [t.format(token=lora["token"] or "", subject=p.get("subject") or "person").strip() for t in EVAL_PROMPTS]
    total = len(files) * len(prompts) * len(p.get("seeds", [11, 22]))
    done = 0
    results = []
    ctx.on_cancel(forge.interrupt)
    try:
        forge.ensure_checkpoint(p["checkpoint"], arch)
        for file in files:
            alias = f"_eval_{file.stem}"
            shutil.copy2(file, staging / f"{alias}.safetensors")
        forge.refresh()
        for file in files:
            alias = f"_eval_{file.stem}"
            folder = base / file.stem
            folder.mkdir()
            scores, shots = [], []
            for prompt_index, prompt in enumerate(prompts):
                for seed in p.get("seeds", [11, 22]):
                    ctx.check_cancel()
                    ctx.progress(done, total, f"{file.stem}: escena {prompt_index + 1} de {len(prompts)}")
                    payload = {
                        "prompt": f"{prompt} <lora:{alias}:{float(p.get('weight', 0.9)):g}>",
                        "negative_prompt": "cartoon, painting, illustration, deformed, blurry, low quality, watermark, text",
                        "seed": seed, "steps": defaults["steps"], "cfg_scale": defaults["cfg_scale"],
                        "width": defaults["width"], "height": defaults["height"],
                        "sampler_name": defaults["sampler"], "scheduler": defaults["scheduler"],
                        "batch_size": 1, "n_iter": 1,
                    }
                    if arch == "flux":
                        payload["distilled_cfg_scale"] = defaults["distilled_cfg_scale"]
                    images, _ = forge.txt2img(payload)
                    ctx.check_cancel()
                    name = f"{prompt_index}_{seed}.png"
                    images[0].convert("RGB").save(folder / name, "PNG")
                    scored = fidelity.score(lora["project_id"], folder / name) or {}
                    scores.append(scored)
                    shots.append({"file": name, "prompt_index": prompt_index, "seed": seed, **scored})
                    done += 1
            step = re.search(r"step0*(\d+)", file.stem)
            results.append({
                "file": file.name, "step": int(step.group(1)) if step else lora["config"].get("steps"),
                "is_final": file.stem == lora["name"], "shots": shots, **fidelity.summarize(scores),
            })
            (base / "result.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    finally:
        ctx.on_cancel(None)
        shutil.rmtree(staging, ignore_errors=True)
        try:
            forge.refresh()
        except forge.ForgeError:
            pass
    ref = fidelity.reference(lora["project_id"]) or {}
    summary = {"prompts": prompts, "checkpoints": results, "real_baseline": ref.get("baseline_face"), "threshold": fidelity.SAME_PERSON_THRESHOLD}
    (base / "result.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return {"checkpoints": len(results)}


def promote_checkpoint(lora: dict[str, Any], filename: str) -> dict[str, Any]:
    """Publica un punto de control como LoRA propio, con el mismo linaje."""
    source = next((f for f in lora_files(lora) if f.name == filename), None)
    if source is None:
        raise FileNotFoundError("Ese punto de control no existe")
    if source.stem == lora["name"]:
        raise ValueError("Ese ya es el archivo final del LoRA")
    step = re.search(r"step0*(\d+)", source.stem)
    name = f"{lora['name']}_s{step.group(1)}" if step else source.stem
    dest = config.LORA_DIR / f"{name}.safetensors"
    if dest.exists() or db.one("SELECT 1 FROM loras WHERE name=?", [name]):
        raise FileExistsError(f"Ya existe {name}")
    shutil.copy2(source, dest)
    new_id = db.new_id()
    db.insert("loras", {
        "id": new_id, "project_id": lora["project_id"], "name": name, "arch": lora["arch"],
        "base_model": lora["base_model"], "token": lora["token"], "origin": "trained", "status": "ready",
        "config": {**(lora["config"] or {}), "promoted_from": lora["id"], "steps": int(step.group(1)) if step else None},
        "train_hashes": lora["train_hashes"], "created_at": db.now(),
    })
    forge.refresh()
    return db.one("SELECT id, name FROM loras WHERE id=?", [new_id])


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def sync_external_loras() -> None:
    """Registra los .safetensors de la carpeta que el sistema no entreno. Su linaje no es verificable."""
    if not config.LORA_DIR.exists():
        return
    known = {r["name"] for r in db.query("SELECT name FROM loras")}
    for path in sorted(config.LORA_DIR.glob("*.safetensors")):
        if path.stem in known or path.stem.startswith("_"):
            continue
        try:
            info = arch_mod.lora_info(path)
        except (OSError, ValueError) as exc:
            log.warning("LoRA ilegible %s: %s", path.name, exc)
            continue
        db.insert("loras", {
            "id": db.new_id(), "project_id": None, "name": path.stem, "arch": info["arch"],
            "base_model": info["base_model"], "token": None, "origin": "external", "status": "ready",
            "config": {"network_dim": info["network_dim"], "steps": info["steps"]},
            "created_at": db.now(),
        })
