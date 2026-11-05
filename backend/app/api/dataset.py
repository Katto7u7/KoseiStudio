"""Importacion, estadisticas, partición, captions y caracteristicas."""
from __future__ import annotations

import shutil
from collections import Counter
from typing import Any, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import config, db, jobs
from ..services import splits, storage
from .projects import project_or_404

router = APIRouter(tags=["dataset"])


class SplitIn(BaseModel):
    train: int = Field(ge=0)
    val: int = Field(ge=0)
    test: int = Field(ge=0)
    seed: int = Field(0, ge=0)


class CaptionJobIn(BaseModel):
    token: str = Field("", max_length=64)
    overwrite: bool = False


@router.post("/projects/{project_id}/import", status_code=202)
async def import_files(project_id: str, files: List[UploadFile] = File(...)):
    project_or_404(project_id)
    if db.one("SELECT 1 FROM projects WHERE id=? AND split_locked_at IS NOT NULL", [project_id]):
        raise HTTPException(409, "La partición está bloqueada: importar ahora invalidaria el experimento")
    upload_dir = config.TMP_DIR / f"upload_{db.new_id()}"
    upload_dir.mkdir(parents=True)
    total = 0
    try:
        for index, upload in enumerate(files):
            name = storage.safe_filename(upload.filename or f"archivo_{index}")
            dest = upload_dir / f"{index:05d}_{name}"
            with dest.open("wb") as out:
                while chunk := await upload.read(1 << 20):
                    total += len(chunk)
                    if total > config.MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "La carga supera el límite permitido")
                    out.write(chunk)
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    return {"job_id": jobs.enqueue("ingest", project_id, {"upload_dir": str(upload_dir)}), "received": len(files), "bytes": total}


@router.get("/projects/{project_id}/stats")
def dataset_stats(project_id: str):
    project_or_404(project_id)
    rows = db.query(
        "SELECT i.width, i.height, i.format, i.size_bytes, i.split, i.status, i.qc, c.name AS class_name "
        "FROM images i LEFT JOIN classes c ON c.id=i.class_id WHERE i.project_id=? AND i.origin='real'",
        [project_id],
    )
    if not rows:
        return {"count": 0}
    resolutions = Counter(f"{r['width']}x{r['height']}" for r in rows)
    aspect = Counter(_aspect_bucket(r["width"], r["height"]) for r in rows)
    formats = Counter(r["format"] for r in rows)
    classes = Counter(r["class_name"] or "sin clase" for r in rows)
    flags = Counter(f for r in rows for f in (r["qc"] or {}).get("flags", []))
    named = [n for c, n in classes.items() if c != "sin clase"]
    imbalance = round(max(named) / min(named), 2) if len(named) >= 2 else None
    instances = db.query(
        "SELECT c.name, COUNT(*) AS n FROM annotations a JOIN classes c ON c.id=a.class_id "
        "JOIN images i ON i.id=a.image_id WHERE i.project_id=? AND i.origin='real' GROUP BY c.name",
        [project_id],
    )
    return {
        "count": len(rows),
        "bytes": sum(r["size_bytes"] for r in rows),
        "resolutions": resolutions.most_common(12),
        "aspect_ratios": aspect.most_common(),
        "formats": formats.most_common(),
        "classes": classes.most_common(),
        "instances": {r["name"]: r["n"] for r in instances},
        "imbalance_ratio": imbalance,
        "flags": flags.most_common(),
        "sharpness": _dist([(r["qc"] or {}).get("sharpness") for r in rows]),
        "brightness": _dist([(r["qc"] or {}).get("brightness") for r in rows]),
    }


def _aspect_bucket(w: int, h: int) -> str:
    ratio = w / h
    if abs(ratio - 1) < 0.05:
        return "1:1"
    return "horizontal" if ratio > 1 else "vertical"


def _dist(values: list[Any]) -> Optional[dict[str, float]]:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    pick = lambda q: round(clean[min(len(clean) - 1, int(q * len(clean)))], 2)
    return {"min": round(clean[0], 2), "p50": pick(0.5), "max": round(clean[-1], 2)}


@router.post("/projects/{project_id}/split")
def assign_split(project_id: str, body: SplitIn):
    project_or_404(project_id)
    try:
        return splits.assign(project_id, body.train, body.val, body.test, body.seed)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/projects/{project_id}/split")
def split_status(project_id: str):
    project = project_or_404(project_id)
    return {
        **splits.leakage_report(project_id),
        "locked": bool(project["split_locked_at"]), "split_hash": project["split_hash"],
        "config": project["split_config"],
    }


@router.post("/projects/{project_id}/split/lock")
def lock_split(project_id: str):
    project_or_404(project_id)
    try:
        return splits.lock(project_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/projects/{project_id}/captions", status_code=202)
def caption_job(project_id: str, body: CaptionJobIn):
    project_or_404(project_id)
    return {"job_id": jobs.enqueue("caption", project_id, body.model_dump())}


@router.post("/projects/{project_id}/features", status_code=202)
def features_job(project_id: str):
    project_or_404(project_id)
    return {"job_id": jobs.enqueue("features", project_id, {})}


@router.get("/projects/{project_id}/features")
def latest_features(project_id: str):
    project_or_404(project_id)
    row = db.one(
        "SELECT result, finished_at FROM jobs WHERE project_id=? AND kind='features' AND status='succeeded' "
        "ORDER BY finished_at DESC LIMIT 1",
        [project_id],
    )
    return row or {"result": None}
