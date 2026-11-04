"""Registro y entrenamiento de LoRAs."""
from __future__ import annotations

import shutil
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import config, db, jobs
from ..services import arch as arch_mod
from ..services import forge, storage, training
from .projects import project_or_404

router = APIRouter(tags=["lora"])


class TrainIn(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    checkpoint: str
    token: str = Field(pattern=r"^[A-Za-z0-9_]{2,40}$")
    steps: int = Field(ge=10, le=20000)
    repeats: int = Field(10, ge=1, le=500)
    network_dim: int = Field(16, ge=1, le=256)
    network_alpha: float = Field(8, gt=0, le=256)
    learning_rate: float = Field(1e-4, gt=0, le=1)
    batch_size: int = Field(1, ge=1, le=8)
    resolution: int = Field(1024, ge=256, le=1536)
    save_every: int = Field(250, ge=10)
    seed: int = Field(42, ge=0)


@router.get("/loras")
def list_loras(project_id: Optional[str] = None):
    training.sync_external_loras()
    rows = db.query(
        "SELECT l.*, j.status AS job_status FROM loras l LEFT JOIN jobs j ON j.id=l.job_id ORDER BY l.created_at DESC"
    )
    for row in rows:
        row["train_image_count"] = len(row.get("train_hashes") or [])
        row.pop("train_hashes", None)
        row["arch_label"] = arch_mod.ARCH_LABEL.get(row["arch"], row["arch"])
    if project_id:
        rows = [r for r in rows if r["project_id"] in (project_id, None)]
    return rows


@router.get("/loras/{lora_id}/losses")
def lora_losses(lora_id: str):
    lora = db.one("SELECT project_id FROM loras WHERE id=?", [lora_id])
    if not lora:
        raise HTTPException(404, "LoRA no encontrado")
    return training.read_losses(lora_id, lora["project_id"]) if lora["project_id"] else []


@router.post("/projects/{project_id}/loras", status_code=202)
def train_lora(project_id: str, body: TrainIn):
    project = project_or_404(project_id)
    if not project["split_locked_at"]:
        raise HTTPException(409, "Bloquea la particion primero: garantiza que el LoRA no vea val ni test")
    if db.one("SELECT 1 FROM loras WHERE name=?", [body.name]) or (config.LORA_DIR / f"{body.name}.safetensors").exists():
        raise HTTPException(409, f"Ya existe un LoRA llamado {body.name}")
    train_images = db.scalar(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND origin='real' AND split='train' AND status!='rejected'",
        [project_id],
    )
    if not train_images:
        raise HTTPException(409, "No hay imagenes en el split de entrenamiento")

    name = body.checkpoint.split(" [")[0]
    path = (config.CHECKPOINT_DIR / name).resolve()
    if config.CHECKPOINT_DIR not in path.parents or not path.exists():
        raise HTTPException(422, "El checkpoint no existe")
    arch = arch_mod.checkpoint_arch(path)
    if arch not in training.SCRIPTS:
        raise HTTPException(422, f"Arquitectura no soportada para entrenar: {arch}")
    if arch == "flux":
        ready, why = training.flux_ready()
        if not ready:
            raise HTTPException(409, why)

    lora_id = db.new_id()
    config_values = body.model_dump()
    db.insert("loras", {
        "id": lora_id, "project_id": project_id, "name": body.name, "arch": arch, "base_model": name,
        "token": body.token, "origin": "trained", "status": "queued", "config": config_values,
        "created_at": db.now(),
    })
    job_id = jobs.enqueue("train_lora", project_id, {**config_values, "lora_id": lora_id, "arch": arch})
    db.update("loras", lora_id, {"job_id": job_id})
    return {"job_id": job_id, "lora_id": lora_id}


@router.post("/projects/{project_id}/loras/import", status_code=201)
async def import_lora(project_id: str, file: UploadFile = File(...), token: str = ""):
    """LoRA entrenado fuera del sistema (por ejemplo en Colab). Su linaje no se puede verificar."""
    project_or_404(project_id)
    name = storage.safe_filename(file.filename or "lora.safetensors")
    if not name.endswith(".safetensors"):
        raise HTTPException(422, "Solo se admite .safetensors: un .ckpt puede ejecutar codigo al cargarse")
    stem = name[: -len(".safetensors")]
    if not storage.LORA_NAME.match(stem):
        raise HTTPException(422, "Nombre de archivo no valido")
    dest = config.LORA_DIR / name
    if dest.exists() or db.one("SELECT 1 FROM loras WHERE name=?", [stem]):
        raise HTTPException(409, f"Ya existe {stem}")
    tmp = config.TMP_DIR / f"lora_{db.new_id()}.safetensors"
    try:
        with tmp.open("wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        try:
            info = arch_mod.lora_info(tmp)
        except Exception as exc:
            raise HTTPException(422, f"El archivo no es un safetensors valido: {exc}") from exc
        shutil.move(str(tmp), dest)
    finally:
        tmp.unlink(missing_ok=True)
    lora_id = db.new_id()
    db.insert("loras", {
        "id": lora_id, "project_id": project_id, "name": stem, "arch": info["arch"],
        "base_model": info["base_model"], "token": token or None, "origin": "external", "status": "ready",
        "config": {"network_dim": info["network_dim"], "steps": info["steps"]}, "created_at": db.now(),
    })
    try:
        forge.refresh()
    except forge.ForgeError:
        pass
    return db.one("SELECT * FROM loras WHERE id=?", [lora_id])
