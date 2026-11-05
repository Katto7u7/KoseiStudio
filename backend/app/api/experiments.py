"""Experimento comparativo real contra real + sintético."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db, jobs
from ..services import experiment as service
from .projects import project_or_404

router = APIRouter(tags=["experimentos"])


class ConditionIn(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_-]{1,24}$")
    n_synthetic: int = Field(0, ge=0)
    use_real: bool = True


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    conditions: List[ConditionIn] = Field(min_length=2, max_length=8)
    baseline: str
    seeds: List[int] = Field(min_length=1, max_length=10)
    epochs: int = Field(100, ge=1, le=1000)
    imgsz: int = Field(640, ge=160, le=1280)
    batch: int = Field(16, ge=1, le=128)
    patience: int = Field(0, ge=0, le=500)
    model: str = Field("yolo11n.yaml", pattern=r"^yolo11[nsm]\.yaml$")
    selection_seed: int = Field(0, ge=0)


@router.post("/projects/{project_id}/experiments/preflight")
def preflight(project_id: str, body: ExperimentIn):
    project_or_404(project_id)
    return service.preflight(project_id, [c.model_dump() for c in body.conditions])


@router.post("/projects/{project_id}/experiments", status_code=202)
def start_experiment(project_id: str, body: ExperimentIn):
    project_or_404(project_id)
    names = [c.name for c in body.conditions]
    if len(set(names)) != len(names):
        raise HTTPException(422, "Los nombres de condición deben ser únicos")
    if body.baseline not in names:
        raise HTTPException(422, "La condición base no existe")
    config = body.model_dump()
    check = service.preflight(project_id, config["conditions"])
    if not check["ok"]:
        raise HTTPException(409, "; ".join(check["problems"]))
    experiment_id = db.new_id()
    job_id = jobs.enqueue("experiment", project_id, {"experiment_id": experiment_id})
    db.insert("experiments", {
        "id": experiment_id, "project_id": project_id, "name": body.name.strip(), "config": config,
        "status": "queued", "job_id": job_id, "created_at": db.now(),
    })
    return {"job_id": job_id, "experiment_id": experiment_id, "preflight": check}


@router.get("/projects/{project_id}/experiments")
def list_experiments(project_id: str):
    project_or_404(project_id)
    rows = db.query(
        "SELECT e.*, j.status AS job_status, j.error AS job_error, j.current AS job_current, "
        "j.total AS job_total, j.message AS job_message FROM experiments e LEFT JOIN jobs j ON j.id=e.job_id "
        "WHERE e.project_id=? ORDER BY e.created_at DESC",
        [project_id],
    )
    for row in rows:
        row["runs"] = db.query(
            "SELECT id, condition, seed, status, counts, metrics, error FROM experiment_runs "
            "WHERE experiment_id=? ORDER BY condition, seed",
            [row["id"]],
        )
    return rows
