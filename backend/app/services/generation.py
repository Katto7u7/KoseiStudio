"""Generacion por lotes: reproducible, cancelable y guardada imagen por imagen."""
from __future__ import annotations

import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .. import config, db
from ..jobs import Cancelled, JobContext, register
from . import forge, imaging, ingest, labeling, provenance, storage

log = logging.getLogger("kosei.generation")

NEAR_DUPLICATE_BITS = 4


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
        if True:
            for group in _batches(pending, plan, batch_size):
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
                    generation = {
                        "chain": chain, "index": index, "prompt": prompt, "base_prompt": p["prompt"],
                        "scenario": plan[index], "negative_prompt": payload["negative_prompt"],
                        "seed": actual_seed, "steps": payload["steps"], "cfg_scale": payload["cfg_scale"],
                        "distilled_cfg_scale": payload.get("distilled_cfg_scale"),
                        "width": image.width, "height": image.height, "sampler": payload["sampler_name"],
                        "scheduler": payload["scheduler"], "checkpoint": p["checkpoint"], "arch": p["arch"],
                        "lora": lora_name, "lora_weight": weight if lora_name else None,
                        "model_hash": info.get("sd_model_hash"), "created_at": db.now(),
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
