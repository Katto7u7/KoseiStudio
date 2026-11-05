"""Galeria, archivos de imagen, anotaciones y revisión humana."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from .. import db, jobs
from ..services import labeling, provenance, storage
from .projects import project_or_404

router = APIRouter(tags=["revisión"])


class BoxIn(BaseModel):
    class_id: int
    cx: float = Field(ge=0, le=1)
    cy: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class BoxesIn(BaseModel):
    boxes: List[BoxIn]


class ImagePatch(BaseModel):
    caption: Optional[str] = Field(None, max_length=2000)


class ReviewIn(BaseModel):
    status: str = Field(pattern="^(approved|rejected|needs_review)$")
    negative: bool = False


class BulkReviewIn(BaseModel):
    ids: Optional[List[str]] = None
    from_status: Optional[str] = Field(None, pattern="^(candidate|needs_review|no_detection)$")
    origin: str = Field("synthetic", pattern="^(real|synthetic)$")
    status: str = Field(pattern="^(approved|rejected)$")


def _image_or_404(image_id: str) -> dict:
    image = db.one("SELECT * FROM images WHERE id=?", [image_id])
    if not image:
        raise HTTPException(404, "Imagen no encontrada")
    return image


def _public(image: dict) -> dict:
    image = {k: v for k, v in image.items() if k not in ("path", "normalized_path", "thumb_path")}
    image["thumb_url"] = f"/api/images/{image['id']}/file?variant=thumb"
    image["file_url"] = f"/api/images/{image['id']}/file"
    return image


@router.get("/projects/{project_id}/images")
def list_images(
    project_id: str,
    origin: Optional[str] = Query(None, pattern="^(real|synthetic)$"),
    status: Optional[str] = None,
    split: Optional[str] = Query(None, pattern="^(train|val|test|none)$"),
    sort: str = Query("recent", pattern="^(recent|fidelity)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    project_or_404(project_id)
    where, args = ["i.project_id=?"], [project_id]
    if origin:
        where.append("i.origin=?")
        args.append(origin)
    if status:
        where.append("i.status=?")
        args.append(status)
    if split == "none":
        where.append("i.split IS NULL")
    elif split:
        where.append("i.split=?")
        args.append(split)
    clause = " AND ".join(where)
    total = db.scalar(f"SELECT COUNT(*) FROM images i WHERE {clause}", args)
    rows = db.query(
        f"SELECT i.*, c.name AS class_name, "
        f"(SELECT COUNT(*) FROM annotations a WHERE a.image_id=i.id) AS annotation_count, "
        f"(SELECT MAX(a.confidence) FROM annotations a WHERE a.image_id=i.id) AS top_confidence "
        f"FROM images i LEFT JOIN classes c ON c.id=i.class_id WHERE {clause} "
        + ("ORDER BY COALESCE(json_extract(i.qc,'$.fidelity.face'), json_extract(i.qc,'$.fidelity.clip')) DESC, i.id "
           if sort == "fidelity" else "ORDER BY i.created_at DESC, i.id ")
        + "LIMIT ? OFFSET ?",
        [*args, limit, offset],
    )
    return {"total": total, "items": [_public(r) for r in rows]}


@router.get("/images/{image_id}")
def get_image(image_id: str):
    image = _image_or_404(image_id)
    out = _public(image)
    out["annotations"] = labeling.annotations_for(image_id)
    return out


@router.get("/images/{image_id}/file")
def image_file(image_id: str, variant: str = Query("full", pattern="^(full|thumb|original)$")):
    image = _image_or_404(image_id)
    key = {"full": "normalized_path", "thumb": "thumb_path", "original": "path"}[variant]
    path = storage.resolve(image[key])
    if not path.exists():
        raise HTTPException(404, "El archivo no está en disco")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=604800, immutable"})


@router.patch("/images/{image_id}")
def patch_image(image_id: str, body: ImagePatch):
    _image_or_404(image_id)
    if body.caption is not None:
        db.update("images", image_id, {"caption": body.caption.strip()})
    return _public(_image_or_404(image_id))


@router.put("/images/{image_id}/annotations")
def replace_annotations(image_id: str, body: BoxesIn):
    image = _image_or_404(image_id)
    allowed = {c["id"] for c in db.query("SELECT id FROM classes WHERE project_id=?", [image["project_id"]])}
    if any(b.class_id not in allowed for b in body.boxes):
        raise HTTPException(422, "Clase ajena al proyecto")
    return labeling.set_boxes(image, [b.model_dump() for b in body.boxes])


@router.post("/images/{image_id}/review")
def review_image(image_id: str, body: ReviewIn):
    image = _image_or_404(image_id)
    count = db.scalar("SELECT COUNT(*) FROM annotations WHERE image_id=?", [image_id])
    if body.status == "approved" and not count and not body.negative:
        raise HTTPException(409, "Sin cajas: confirma que la imagen es un negativo real o dibuja la caja")
    qc = image["qc"] or {}
    with db.transaction():
        if body.status == "approved":
            db.execute("UPDATE annotations SET verified=1 WHERE image_id=?", [image_id])
        if body.negative:
            qc["confirmed_negative"] = True
        db.update("images", image_id, {"status": body.status, "qc": qc})
    return _public(_image_or_404(image_id))


@router.post("/projects/{project_id}/review/bulk")
def review_bulk(project_id: str, body: BulkReviewIn):
    """Aprobar en lote solo actua sobre imágenes con cajas: un negativo exige revisión individual."""
    project_or_404(project_id)
    where, args = ["project_id=?", "origin=?"], [project_id, body.origin]
    if body.ids:
        where.append(f"id IN ({','.join('?' * len(body.ids))})")
        args.extend(body.ids)
    elif body.from_status:
        where.append("status=?")
        args.append(body.from_status)
    else:
        raise HTTPException(422, "Indica ids o from_status")
    rows = db.query(f"SELECT id FROM images WHERE {' AND '.join(where)}", args)
    changed = 0
    with db.transaction():
        for row in rows:
            count = db.scalar("SELECT COUNT(*) FROM annotations WHERE image_id=?", [row["id"]])
            if body.status == "approved":
                if not count:
                    continue
                db.execute("UPDATE annotations SET verified=1 WHERE image_id=?", [row["id"]])
            db.update("images", row["id"], {"status": body.status})
            changed += 1
    return {"changed": changed, "skipped": len(rows) - changed}


@router.post("/projects/{project_id}/label", status_code=202)
def start_labeling(project_id: str, relabel: bool = False):
    """Pone en cola lo pendiente. `relabel` reprocesa lo no verificado por una persona."""
    project_or_404(project_id)
    if relabel:
        db.execute(
            "UPDATE images SET status='pending' WHERE project_id=? AND status IN ('candidate','needs_review','no_detection')",
            [project_id],
        )
    return {"job_id": labeling.ensure_label_job(project_id)}


@router.get("/images/{image_id}/provenance")
def image_provenance(image_id: str):
    """Comprueba la procedencia en el archivo real: registro PNG y marca de agua."""
    image = _image_or_404(image_id)
    path = storage.resolve(image["path"])
    with Image.open(path) as handle:
        handle.load()
        record = provenance.read_record(handle)
        watermark = provenance.read_watermark(handle)
    return {
        "sha256": image["sha256"], "sha256_matches": storage.sha256_file(path) == image["sha256"],
        "origin": image["origin"], "png_record": record, "watermark_detected": watermark,
    }
