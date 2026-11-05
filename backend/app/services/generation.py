"""Generacion por lotes: reproducible, cancelable y guardada imagen por imagen."""
from __future__ import annotations

import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .. import config, db
from ..jobs import Cancelled, JobContext, register
from . import fidelity, forge, imaging, ingest, labeling, provenance, storage

log = logging.getLogger("kosei.generation")

NEAR_DUPLICATE_BITS = 4
REFINE_MARGIN = 1.6


def scenario_plan(scenarios: dict[str, list[str]], count: int, seed: int) -> list[list[str]]:
    """Reparte las combinaciones sin repetir hasta agotarlas. Asi se fuerza la diversidad."""
    axes = [[v.strip() for v in values if v and v.strip()] for values in scenarios.values()]
    axes = [a for a in axes if a]
    if not axes:
        return [[] for _ in range(count)]
    combos = [list(c) for c in itertools.product(*axes)]
    rng = random.Random(seed)
    plan: list[list[str]] = []
    while len(plan) < count:
        round_ = combos[:]
        rng.shuffle(round_)
        plan.extend(round_)
    return plan[:count]


def compose_prompt(base: str, scenario: list[str], token: Optional[str], lora_name: Optional[str], weight: float) -> str:
    text = base.strip()
    if token and token.lower() not in text.lower():
        text = f"{token} {text}"
    extra = ", ".join(scenario)
    if "{scenario}" in text:
        text = text.replace("{scenario}", extra)
    elif extra:
        text = f"{text}, {extra}"
    if lora_name:
        text = f"{text} <lora:{lora_name}:{weight:g}>"
    return text


def _batches(pending: list[int], plan: list[list[str]], batch_size: int) -> list[list[int]]:
    """Un lote de Forge comparte prompt: solo agrupa indices consecutivos con igual escenario."""
    groups: list[list[int]] = []
    for index in pending:
        last = groups[-1] if groups else None
        if last and len(last) < batch_size and last[-1] == index - 1 and plan[last[0]] == plan[index]:
            last.append(index)
        else:
            groups.append([index])
    return groups


def _chain_root(ctx: JobContext) -> str:
    current, root = ctx.retry_of, ctx.id
    while current:
        root = current
        row = db.one("SELECT retry_of FROM jobs WHERE id=?", [current])
        current = row["retry_of"] if row else None
    return root


def _subject_region(image: Image.Image, cls: Optional[dict[str, Any]], tmp: Path) -> Optional[tuple[int, int, int, int]]:
    """Region a refinar: el rostro si lo hay; si no, la caja del objeto de la clase."""
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))
    box = None
    if fidelity.face_models_available():
        face = fidelity.largest_face(rgb[..., ::-1].copy())
        if face is not None:
            x, y, w, h = [float(v) for v in face[:4]]
            box = (x, y, w, h)
    if box is None and cls is not None:
        probe = tmp / "region.png"
        image.convert("RGB").save(probe, "PNG")
        found, _ = labeling.detect(probe, cls)
        probe.unlink(missing_ok=True)
        if found:
            b = found[0]
            box = ((b["cx"] - b["w"] / 2) * width, (b["cy"] - b["h"] / 2) * height, b["w"] * width, b["h"] * height)
    if box is None:
        return None
    x, y, w, h = box
    side = max(w, h) * REFINE_MARGIN
    # Si la region ya ocupa casi toda la imagen, refinarla no aporta resolucion.
    if side >= min(width, height) * 0.8:
        return None
    side = int(side)
    cx, cy = x + w / 2, y + h / 2
    left = int(min(max(cx - side / 2, 0), width - side))
    top = int(min(max(cy - side / 2, 0), height - side))
    return left, top, left + side, top + side


def refine_subject(image: Image.Image, region: tuple[int, int, int, int], payload: dict[str, Any], denoise: float) -> Image.Image:
    """Regenera la region a resolucion completa conservando la composicion, y la vuelve a pegar."""
    crop = image.crop(region)
    target = (int(payload["width"]), int(payload["height"]))
    enlarged = crop.resize(target, Image.Resampling.LANCZOS)
    refined = forge.img2img({
        "prompt": payload["prompt"], "negative_prompt": payload["negative_prompt"], "seed": payload["seed"],
        "steps": payload["steps"], "cfg_scale": payload["cfg_scale"], "width": target[0], "height": target[1],
        "sampler_name": payload["sampler_name"], "scheduler": payload["scheduler"],
        "denoising_strength": denoise, "batch_size": 1, "n_iter": 1,
        **({"distilled_cfg_scale": payload["distilled_cfg_scale"]} if "distilled_cfg_scale" in payload else {}),
    }, enlarged)
    patch = refined.convert("RGB").resize(crop.size, Image.Resampling.LANCZOS)
    size = crop.size[0]
    feather = max(4, size // 12)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[feather:-feather, feather:-feather] = 1.0
    blurred = cv2.GaussianBlur(mask, (0, 0), feather / 2)
    out = image.convert("RGB").copy()
    out.paste(patch, region[:2], Image.fromarray((blurred * 255).astype(np.uint8)))
    return out


def _save_png(image: Image.Image, path: Path, record: dict[str, Any]) -> None:
    info = PngInfo()
    info.add_text("kosei_provenance", json.dumps(record, ensure_ascii=False))
    info.add_text("Software", "KoseiStudio")
    image.convert("RGB").save(path, "PNG", pnginfo=info)


@register("generate", "gpu")
def run_generate(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    project_id = ctx.project_id
    count, seed = int(p["count"]), int(p["seed"])
    batch_size = max(1, int(p.get("batch_size") or 1))
    plan = scenario_plan(p.get("scenarios") or {}, count, seed)

    lora = db.one("SELECT * FROM loras WHERE id=?", [p["lora_id"]]) if p.get("lora_id") else None
    token = lora["token"] if lora else None
    lora_name = lora["name"] if lora else None
    weight = float(p.get("lora_weight", 0.9))
    refine = bool(p.get("refine"))
    denoise = float(p.get("refine_denoise", 0.4))
    cls = db.one("SELECT * FROM classes WHERE id=?", [p.get("class_id")]) if p.get("class_id") else None

    # Reanudacion: una imagen ya guardada conserva su indice y no se vuelve a generar.
    chain = _chain_root(ctx)
    done = {
        r["i"] for r in db.query(
            "SELECT json_extract(generation,'$.index') AS i FROM images "
            "WHERE project_id=? AND json_extract(generation,'$.chain')=?",
            [project_id, chain],
        )
    }
    pending = [i for i in range(count) if i not in done]

    ctx.on_cancel(forge.interrupt)
    tmp = config.TMP_DIR / f"gen_{ctx.id}"
    tmp.mkdir(parents=True, exist_ok=True)
    known = db.query("SELECT id, phash FROM images WHERE project_id=?", [project_id])
    produced = len(done)
    stats = {"generated": 0, "near_duplicates": 0, "errors": 0, "resumed_from": len(done)}

    try:
        ctx.progress(produced, count, "Cargando modelo en el motor")
        forge.ensure_checkpoint(p["checkpoint"], p["arch"])
        for group in _batches(pending, plan, 1 if refine else batch_size):
            ctx.check_cancel()
            prompt = compose_prompt(p["prompt"], plan[group[0]], token, lora_name, weight)
            payload = {
                "prompt": prompt,
                "negative_prompt": p.get("negative_prompt") or "",
                "seed": seed + group[0],
                "steps": int(p["steps"]),
                "cfg_scale": float(p["cfg_scale"]),
                "width": int(p["width"]),
                "height": int(p["height"]),
                "sampler_name": p["sampler"],
                "scheduler": p["scheduler"],
                "batch_size": len(group),
                "n_iter": 1,
            }
            if p["arch"] == "flux":
                payload["distilled_cfg_scale"] = float(p.get("distilled_cfg_scale", 3.5))
            ctx.progress(produced, count, f"Generando {produced + 1} de {count}")
            images, info = forge.txt2img(payload)
            ctx.check_cancel()
            if not images:
                raise forge.ForgeError("El motor no devolvio imagenes")
            seeds = info.get("all_seeds") or [seed + j for j in group]
            for offset, (index, image) in enumerate(zip(group, images)):
                actual_seed = int(seeds[offset]) if offset < len(seeds) else seed + index
                refined_region = None
                if refine:
                    ctx.check_cancel()
                    refined_region = _subject_region(image, cls, tmp)
                    if refined_region:
                        ctx.progress(produced, count, f"Refinando el sujeto de la imagen {produced + 1}")
                        image = refine_subject(image, refined_region, {**payload, "seed": actual_seed}, denoise)
                generation = {
                    "chain": chain, "index": index, "prompt": prompt, "base_prompt": p["prompt"],
                    "scenario": plan[index], "negative_prompt": payload["negative_prompt"],
                    "seed": actual_seed, "steps": payload["steps"], "cfg_scale": payload["cfg_scale"],
                    "distilled_cfg_scale": payload.get("distilled_cfg_scale"),
                    "width": image.width, "height": image.height, "sampler": payload["sampler_name"],
                    "scheduler": payload["scheduler"], "checkpoint": p["checkpoint"], "arch": p["arch"],
                    "lora": lora_name, "lora_weight": weight if lora_name else None,
                    "model_hash": info.get("sd_model_hash"), "created_at": db.now(),
                    "refined_region": list(refined_region) if refined_region else None,
                    "refine_denoise": denoise if refined_region else None,
                }
                path = tmp / f"{ctx.id}_{index:05d}.png"
                marked = provenance.embed_watermark(image, ctx.project_id or "")
                _save_png(marked, path, provenance.record(generation))
                image_id, new, extra = ingest.register_image(
                    project_id, path, f"gen_{actual_seed}.png", "synthetic",
                    extra={
                        "status": "pending", "class_id": p.get("class_id"),
                        "class_source": "condicionamiento", "generation": generation,
                        "job_id": ctx.id, "lora_id": p.get("lora_id"),
                    },
                    move=True,
                )
                produced += 1
                if not new:
                    continue
                stats["generated"] += 1
                row = db.one("SELECT phash, qc FROM images WHERE id=?", [image_id])
                near = next((k for k in known if imaging.hamming(k["phash"], row["phash"]) <= NEAR_DUPLICATE_BITS), None)
                if near:
                    stats["near_duplicates"] += 1
                    db.update("images", image_id, {"qc": {**(row["qc"] or {}), "near_duplicate_of": near["id"]}})
                known.append({"id": image_id, "phash": row["phash"]})
            ctx.progress(produced, count, f"{produced} de {count} generadas")
            labeling.ensure_label_job(project_id)
    except Cancelled:
        labeling.ensure_label_job(project_id)
        raise
    finally:
        ctx.on_cancel(None)
        for leftover in tmp.glob("*"):
            leftover.unlink(missing_ok=True)
        try:
            tmp.rmdir()
        except OSError:
            pass
    return stats


def seed_for(params: dict[str, Any]) -> int:
    value = params.get("seed")
    if value is None or int(value) < 0:
        return random.randrange(1, 2**31 - params.get("count", 1) - 1)
    return int(value)
