"""Calidad y construccion del dataset extendido."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import db, jobs
from ..services import export, storage
from .projects import project_or_404

router = APIRouter(tags=["resultados"])


class BuildIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    n_synthetic: Optional[int] = Field(None, ge=0)
    seed: int = Field(0, ge=0)
    tasks: List[str] = Field(default_factory=lambda: ["detect", "segment"])


@router.post("/projects/{project_id}/quality", status_code=202)
def start_quality(project_id: str):
    project_or_404(project_id)
    report_id = db.new_id()
    job_id = jobs.enqueue("quality", project_id, {"report_id": report_id})
    db.insert("quality_reports", {
        "id": report_id, "project_id": project_id, "status": "running", "job_id": job_id, "created_at": db.now(),
    })
    return {"job_id": job_id, "report_id": report_id}


@router.get("/projects/{project_id}/quality")
def list_quality(project_id: str):
    project_or_404(project_id)
    rows = db.query(
        "SELECT q.*, j.status AS job_status, j.error AS job_error FROM quality_reports q "
        "LEFT JOIN jobs j ON j.id=q.job_id WHERE q.project_id=? ORDER BY q.created_at DESC LIMIT 20",
        [project_id],
    )
    return rows


@router.post("/projects/{project_id}/builds", status_code=202)
def start_build(project_id: str, body: BuildIn):
    project_or_404(project_id)
    bad = [t for t in body.tasks if t not in export.TASKS]
    if bad or not body.tasks:
        raise HTTPException(422, "Tarea de exportacion no valida")
    build_id = db.new_id()
    params = {**body.model_dump(), "build_id": build_id}
    job_id = jobs.enqueue("build", project_id, params)
    db.insert("builds", {
        "id": build_id, "project_id": project_id, "name": body.name.strip(), "config": body.model_dump(),
        "status": "running", "job_id": job_id, "created_at": db.now(),
    })
    return {"job_id": job_id, "build_id": build_id}


@router.get("/projects/{project_id}/builds")
def list_builds(project_id: str):
    project_or_404(project_id)
    return db.query(
        "SELECT b.*, j.status AS job_status, j.error AS job_error, j.message AS job_message, "
        "j.current AS job_current, j.total AS job_total FROM builds b LEFT JOIN jobs j ON j.id=b.job_id "
        "WHERE b.project_id=? ORDER BY b.created_at DESC",
        [project_id],
    )


@router.get("/builds/{build_id}/download")
def download_build(build_id: str):
    build = db.one("SELECT * FROM builds WHERE id=?", [build_id])
    if not build or build["status"] != "ready":
        raise HTTPException(404, "Exportacion no disponible")
    name = build["summary"]["zip"]
    path = storage.project_dir(build["project_id"]) / "builds" / build_id / storage.safe_filename(name)
    if not path.exists():
        raise HTTPException(404, "El archivo ya no esta en disco")
    return FileResponse(path, media_type="application/zip", filename=path.name)
