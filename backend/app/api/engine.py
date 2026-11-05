"""Motor de generación: modelos disponibles, generación y vista previa en vivo."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db, jobs
from ..services import forge, generation, training
from .projects import project_or_404

router = APIRouter(tags=["motor"])


class GenerateIn(BaseModel):
    checkpoint: str
    lora_id: Optional[str] = None
    lora_weight: float = Field(0.9, ge=0, le=2)
    class_id: int
    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str = Field("", max_length=2000)
    scenarios: dict[str, list[str]] = Field(default_factory=dict)
    count: int = Field(ge=1, le=5000)
    seed: int = -1
    steps: int = Field(ge=1, le=150)
    cfg_scale: float = Field(ge=1, le=30)
    distilled_cfg_scale: float = Field(3.5, ge=0, le=30)
    width: int = Field(ge=256, le=2048)
    height: int = Field(ge=256, le=2048)
    sampler: str
    scheduler: str
    batch_size: int = Field(1, ge=1, le=8)
    refine: bool = False
    refine_denoise: float = Field(0.4, ge=0.05, le=0.9)


@router.get("/engine/models")
def engine_models():
    return forge.models()


@router.get("/engine/options")
def engine_options():
    return {"samplers": forge.samplers(), "schedulers": forge.schedulers()}


@router.get("/engine/progress")
def engine_progress():
    return forge.progress()


@router.post("/projects/{project_id}/generate", status_code=202)
def start_generation(project_id: str, body: GenerateIn):
    project_or_404(project_id)
    if not db.one("SELECT 1 FROM classes WHERE id=? AND project_id=?", [body.class_id, project_id]):
        raise HTTPException(422, "La clase no pertenece al proyecto")
    if body.width % 8 or body.height % 8:
        raise HTTPException(422, "Ancho y alto deben ser multiplos de 8")

    model = next((m for m in forge.models() if m["title"] == body.checkpoint), None)
    if model is None:
        raise HTTPException(422, "El checkpoint no existe en el motor")

    if body.lora_id:
        lora = db.one("SELECT * FROM loras WHERE id=?", [body.lora_id])
        if not lora or lora["status"] != "ready":
            raise HTTPException(422, "El LoRA no está disponible")
        if lora["arch"] != "unknown" and lora["arch"] != model["arch"]:
            raise HTTPException(
                422,
                f"Incompatible: el LoRA {lora['name']} es {lora['arch']} y el checkpoint es {model['arch']}",
            )
    params = body.model_dump()
    params["arch"] = model["arch"]
    params["seed"] = generation.seed_for(params)
    return {"job_id": jobs.enqueue("generate", project_id, params), "seed": params["seed"]}
