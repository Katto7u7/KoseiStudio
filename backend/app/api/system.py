"""Estado del sistema y gestion de trabajos."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from .. import config, db, jobs
from ..services import forge, gpu

router = APIRouter(tags=["sistema"])


@router.get("/health")
def health():
    db.scalar("SELECT 1")
    return {"status": "ok"}


@router.get("/system")
def system_status():
    engine_ready = forge.is_ready()
    disk = shutil.disk_usage(config.DATA_DIR)
    weights = {
        "yolo11m.pt": (config.YOLO_DIR / "yolo11m.pt").exists(),
        "mobile_sam.pt": (config.YOLO_DIR / "mobile_sam.pt").exists(),
        "yolov8s-worldv2.pt": (config.YOLO_DIR / "yolov8s-worldv2.pt").exists(),
        "BLIP": (config.WEIGHTS_DIR / "hf/hub/models--Salesforce--blip-image-captioning-large").exists(),
        "CLIP": (config.WEIGHTS_DIR / "hf/hub/models--openai--clip-vit-base-patch32").exists(),
        "Inception (FID)": (config.WEIGHTS_DIR / "torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth").exists(),
    }
    return {
        "engine": {"ready": engine_ready, "checkpoint": forge.current_checkpoint() if engine_ready else None},
        "gpu": gpu.status(),
        "disk": {"free_bytes": disk.free, "total_bytes": disk.total},
        "weights": weights,
        "kohya": (Path(config.KOHYA_PYTHON).exists() and config.KOHYA_DIR.exists()),
        "active_jobs": db.query("SELECT id, kind, status, project_id, current, total, message FROM jobs WHERE status IN ('queued','running') ORDER BY created_at"),
    }


@router.get("/jobs")
def list_jobs(project_id: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    if project_id:
        return db.query("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?", [project_id, limit])
    return db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", [limit])


def _job_or_404(job_id: str) -> dict:
    job = db.one("SELECT * FROM jobs WHERE id=?", [job_id])
    if not job:
        raise HTTPException(404, "Trabajo no encontrado")
    return job


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    return _job_or_404(job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] not in jobs.ACTIVE:
        raise HTTPException(409, "El trabajo ya termino")
    return jobs.cancel(job_id)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] in jobs.ACTIVE:
        raise HTTPException(409, "El trabajo sigue activo")
    return {"job_id": jobs.retry(job_id)}


@router.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str, tail: int = Query(200, ge=1, le=5000)):
    _job_or_404(job_id)
    path = config.DATA_DIR / "logs" / f"{job_id}.log"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:])
