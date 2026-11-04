"""Proyectos, clases y resumen del estado del flujo."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .. import db
from ..services import storage

router = APIRouter(tags=["proyectos"])


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field("", max_length=1000)


class ProjectPatch(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=80)
    description: Optional[str] = Field(None, max_length=1000)


class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    detector: str = Field("coco", pattern="^(coco|world)$")
    detector_prompt: str = Field("", max_length=200)


class ClassPatch(BaseModel):
    detector: Optional[str] = Field(None, pattern="^(coco|world)$")
    detector_prompt: Optional[str] = Field(None, max_length=200)


def project_or_404(project_id: str) -> dict[str, Any]:
    project = db.one("SELECT * FROM projects WHERE id=?", [project_id])
    if not project:
        raise HTTPException(404, "Proyecto no encontrado")
    return project


def pipeline(project_id: str) -> dict[str, Any]:
    """Estado real de cada etapa, calculado de los datos y nunca guardado aparte."""
    rows = db.query(
        "SELECT origin, status, split, COUNT(*) AS n FROM images WHERE project_id=? GROUP BY origin, status, split",
        [project_id],
    )
    real = sum(r["n"] for r in rows if r["origin"] == "real")
    synth = sum(r["n"] for r in rows if r["origin"] == "synthetic")
    by_status: dict[str, dict[str, int]] = {"real": {}, "synthetic": {}}
    by_split = {"train": 0, "val": 0, "test": 0, "none": 0}
    for r in rows:
        by_status[r["origin"]][r["status"]] = by_status[r["origin"]].get(r["status"], 0) + r["n"]
        if r["origin"] == "real":
            by_split[r["split"] or "none"] += r["n"]
    project = db.one("SELECT split_locked_at FROM projects WHERE id=?", [project_id])
    captions = db.scalar(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND origin='real' AND split='train' AND caption IS NOT NULL AND caption!=''",
        [project_id],
    )
    return {
        "real_images": real, "synthetic_images": synth, "by_status": by_status, "by_split": by_split,
        "split_locked": bool(project["split_locked_at"]), "train_captions": captions,
        "classes": db.scalar("SELECT COUNT(*) FROM classes WHERE project_id=?", [project_id]),
        "loras_ready": db.scalar("SELECT COUNT(*) FROM loras WHERE project_id=? AND status='ready'", [project_id]),
        "quality_reports": db.scalar("SELECT COUNT(*) FROM quality_reports WHERE project_id=? AND status='ready'", [project_id]),
        "builds_ready": db.scalar("SELECT COUNT(*) FROM builds WHERE project_id=? AND status='ready'", [project_id]),
        "experiments_ready": db.scalar("SELECT COUNT(*) FROM experiments WHERE project_id=? AND status='ready'", [project_id]),
    }


@router.get("/projects")
def list_projects():
    projects = db.query("SELECT * FROM projects ORDER BY updated_at DESC")
    for project in projects:
        project["pipeline"] = pipeline(project["id"])
    return projects


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn):
    project_id = db.new_id()
    now = db.now()
    db.insert("projects", {
        "id": project_id, "name": body.name.strip(), "description": body.description.strip(),
        "created_at": now, "updated_at": now,
    })
    storage.project_dir(project_id).mkdir(parents=True, exist_ok=True)
    return get_project(project_id)


@router.get("/projects/{project_id}")
def get_project(project_id: str):
    project = project_or_404(project_id)
    project["pipeline"] = pipeline(project_id)
    project["classes"] = db.query("SELECT * FROM classes WHERE project_id=? ORDER BY idx", [project_id])
    return project


@router.patch("/projects/{project_id}")
def patch_project(project_id: str, body: ProjectPatch):
    project_or_404(project_id)
    values = {k: v.strip() for k, v in body.model_dump(exclude_none=True).items()}
    if values:
        db.update("projects", project_id, {**values, "updated_at": db.now()})
    return get_project(project_id)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str):
    project_or_404(project_id)
    if db.scalar("SELECT COUNT(*) FROM jobs WHERE project_id=? AND status IN ('queued','running')", [project_id]):
        raise HTTPException(409, "El proyecto tiene trabajos activos")
    db.execute("DELETE FROM projects WHERE id=?", [project_id])
    storage.remove_tree(storage.project_dir(project_id))


@router.post("/projects/{project_id}/classes", status_code=201)
def add_class(project_id: str, body: ClassIn):
    project_or_404(project_id)
    name = body.name.strip()
    if db.one("SELECT 1 FROM classes WHERE project_id=? AND name=?", [project_id, name]):
        raise HTTPException(409, f"La clase {name} ya existe")
    idx = int(db.scalar("SELECT COALESCE(MAX(idx)+1,0) FROM classes WHERE project_id=?", [project_id]))
    db.insert("classes", {
        "project_id": project_id, "name": name, "idx": idx,
        "detector": body.detector, "detector_prompt": body.detector_prompt.strip(),
    })
    return db.one("SELECT * FROM classes WHERE project_id=? AND name=?", [project_id, name])


@router.patch("/projects/{project_id}/classes/{class_id}")
def patch_class(project_id: str, class_id: int, body: ClassPatch):
    if not db.one("SELECT 1 FROM classes WHERE id=? AND project_id=?", [class_id, project_id]):
        raise HTTPException(404, "Clase no encontrada")
    values = body.model_dump(exclude_none=True)
    if values:
        db.update("classes", class_id, values)
    return db.one("SELECT * FROM classes WHERE id=?", [class_id])


@router.delete("/projects/{project_id}/classes/{class_id}", status_code=204)
def delete_class(project_id: str, class_id: int):
    if not db.one("SELECT 1 FROM classes WHERE id=? AND project_id=?", [class_id, project_id]):
        raise HTTPException(404, "Clase no encontrada")
    if db.scalar("SELECT COUNT(*) FROM annotations WHERE class_id=?", [class_id]):
        raise HTTPException(409, "La clase tiene anotaciones")
    db.execute("DELETE FROM classes WHERE id=?", [class_id])
