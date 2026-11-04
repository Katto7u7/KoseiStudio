from __future__ import annotations

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import urlopen

import webuiapi
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from funciones import AutoEtiquetado, extraerPrompts
from funciones.dataset import build_yolo_zip, read_labels
from funciones import training as training_mod

BASE_DIR = Path(__file__).resolve().parent

try:
    _uiv0_1 = BASE_DIR.parents[1] / "UIv0_1"
except IndexError:
    _uiv0_1 = None

_default_lora = str(_uiv0_1 / "modelos" / "LORA") if _uiv0_1 and _uiv0_1.exists() else str(BASE_DIR / "lora")
_default_yolo = str(_uiv0_1 / "modelosAutoEtiquetas") if _uiv0_1 and _uiv0_1.exists() else str(BASE_DIR / "yolo")

LORA_DIR        = Path(os.getenv("LORA_DIR",        _default_lora)).resolve()
AUTO_MODELS_DIR = Path(os.getenv("AUTO_MODELS_DIR", _default_yolo)).resolve()
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR",      BASE_DIR / "out")).resolve()
OUTPUT_TXT_DIR  = Path(os.getenv("OUTPUT_TXT_DIR",  BASE_DIR / "outTxt")).resolve()
KOHYA_DIR       = Path(os.getenv("KOHYA_DIR",       BASE_DIR / "kohya_scripts")).resolve()
TRAIN_DATA_DIR  = Path(os.getenv("TRAIN_DATA_DIR",  BASE_DIR / "training")).resolve()

for d in (OUTPUT_DIR, OUTPUT_TXT_DIR, TRAIN_DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

NEGATIVE_PROMPT = os.getenv(
    "NEGATIVE_PROMPT",
    "ugly, out of frame, deformations, weird face, extra hands, extra fingers, mutilations",
)

ENABLE_ADETAILER = os.getenv("ENABLE_ADETAILER", "1").strip().lower() not in {"0", "false", "no"}
ADETAILER_AVAILABLE: Optional[bool] = None

SAMPLERS = [
    "Euler a", "Euler", "DPM++ 2M Karras", "DDIM",
    "DPM++ 2M Turbo", "DPM++ SDE Karras",
]

_yolo_cls = None
_yolo_det = None
_yolo_seg = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _yolo_cls, _yolo_det, _yolo_seg
    cls_path = AUTO_MODELS_DIR / "yolo11n-cls.pt"
    det_path = AUTO_MODELS_DIR / "yolo11n.pt"
    seg_path = AUTO_MODELS_DIR / "yolo11n-seg.pt"
    if cls_path.exists() and det_path.exists() and seg_path.exists():
        print("[startup] Loading YOLO models...")
        _yolo_cls, _yolo_det, _yolo_seg = AutoEtiquetado.load_models(
            str(cls_path), str(det_path), str(seg_path)
        )
        print("[startup] YOLO models loaded.")
    else:
        print(f"[startup] YOLO models not found at {AUTO_MODELS_DIR} — auto-labeling disabled.")
    yield


def build_api(sampler: str) -> webuiapi.WebUIApi:
    host = os.getenv("SD_HOST", "127.0.0.1")
    port = int(os.getenv("SD_PORT", "7860"))
    return webuiapi.WebUIApi(host=host, port=port, sampler=sampler, steps=20)


def is_adetailer_available() -> bool:
    global ADETAILER_AVAILABLE
    if not ENABLE_ADETAILER:
        return False
    if ADETAILER_AVAILABLE is not None:
        return ADETAILER_AVAILABLE
    host = os.getenv("SD_HOST", "127.0.0.1")
    port = int(os.getenv("SD_PORT", "7860"))
    try:
        with urlopen(f"http://{host}:{port}/sdapi/v1/scripts", timeout=3) as response:
            payload = json.load(response)
    except (URLError, ValueError) as exc:
        print(f"ADetailer check failed: {exc}")
        ADETAILER_AVAILABLE = False
        return False
    scripts: List[str] = []
    if isinstance(payload, dict):
        for key in ("txt2img", "img2img"):
            items = payload.get(key, [])
            if isinstance(items, list):
                scripts.extend([str(i) for i in items])
    elif isinstance(payload, list):
        scripts.extend([str(i) for i in payload])
    ADETAILER_AVAILABLE = any("adetailer" in i.lower() for i in scripts)
    return ADETAILER_AVAILABLE


# ─── Pydantic models ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    steps: int = Field(20, ge=1, le=99)
    width: int = Field(512, ge=224, le=1920)
    height: int = Field(512, ge=224, le=1920)
    batch: int = Field(1, ge=1, le=99)
    sampler: str = "Euler a"
    sd_model: str
    lora: str = "NINGUNO"
    negative_prompt: Optional[str] = None
    cfg_scale: float = Field(7.0, ge=1.0, le=30.0)
    target_class: str = "object"
    class_id: int = Field(0, ge=0, le=999)


class GenerateResult(BaseModel):
    id: str
    image_url: str


class GenerateResponse(BaseModel):
    results: List[GenerateResult]
    prompt: str
    negative_prompt: str
    sampler: str
    sd_model: str
    width: int
    height: int
    steps: int


class OutputEntry(BaseModel):
    id: str
    image_url: str
    metadata_url: Optional[str] = None
    timestamp: str


class DatasetEntry(BaseModel):
    id: str
    image_url: str
    class_label: Optional[str] = None
    has_detection: bool = False
    has_segmentation: bool = False
    lora_used: Optional[str] = None
    timestamp: str


class DatasetExportRequest(BaseModel):
    ids: Optional[List[str]] = None


class TrainingUploadResponse(BaseModel):
    job_id: str
    image_count: int
    filenames: List[str]


class CaptionEntry(BaseModel):
    filename: str
    caption: str


class TrainingCaptionRequest(BaseModel):
    job_id: str
    prefix: str


class TrainingCaptionResponse(BaseModel):
    captions: List[CaptionEntry]


class TrainingStartRequest(BaseModel):
    job_id: str
    output_name: str
    base_model: str
    token_prefix: str
    n_repeats: int = 100
    network_dim: int = 8
    learning_rate: float = 0.0001
    max_steps: int = 1000
    batch_size: int = 1
    captions: List[CaptionEntry] = []


# ─── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="KoseiStudio API", lifespan=lifespan)

allowed_origins = os.getenv(
    "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=str(OUTPUT_DIR)), name="assets")


def parse_sd_models(raw) -> List[str]:
    if isinstance(raw, str):
        import ast
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(raw, list):
        return []
    return [item["title"] for item in raw if isinstance(item, dict) and item.get("title")]


# ─── Core endpoints ───────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "sd_host": os.getenv("SD_HOST", "127.0.0.1"), "sd_port": int(os.getenv("SD_PORT", "7860"))}


@app.get("/api/models")
def list_models():
    api = build_api(SAMPLERS[0])
    raw = api.get_sd_models()
    return {"models": parse_sd_models(raw)}


@app.get("/api/loras")
def list_loras():
    if not LORA_DIR.exists():
        return {"loras": []}
    loras = sorted([item.stem for item in LORA_DIR.iterdir() if item.is_file()])
    return {"loras": loras}


@app.get("/api/samplers")
def list_samplers():
    return {"samplers": SAMPLERS}


@app.get("/api/outputs", response_model=List[OutputEntry])
def list_outputs():
    if not OUTPUT_DIR.exists():
        return []
    entries = []
    for png in sorted(OUTPUT_DIR.glob("*.png"), reverse=True):
        stem = png.stem
        if "_" in stem:
            continue
        txt_path = OUTPUT_TXT_DIR / f"{stem}.txt"
        entries.append(OutputEntry(
            id=stem,
            image_url=f"/assets/{stem}.png",
            metadata_url=f"/api/outputs/{stem}/meta" if txt_path.exists() else None,
            timestamp=stem,
        ))
    return entries


@app.get("/api/outputs/{output_id}/meta", response_class=PlainTextResponse)
def get_output_meta(output_id: str):
    txt = OUTPUT_TXT_DIR / f"{output_id}.txt"
    if not txt.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")
    return txt.read_text()


@app.post("/api/generate", response_model=GenerateResponse)
def generate_images(request: GenerateRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt required")

    api = build_api(request.sampler)
    try:
        api.set_options({"sd_model_checkpoint": request.sd_model})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to set model: {exc}")

    prompt = request.prompt.strip()
    if request.lora and request.lora != "NINGUNO":
        prompt = f"<lora:{request.lora}:1> {prompt}"

    negative_prompt = request.negative_prompt.strip() if request.negative_prompt else NEGATIVE_PROMPT
    results: List[GenerateResult] = []
    adetailer_payload = (
        [webuiapi.ADetailer(ad_model="face_yolov8n.pt")] if is_adetailer_available() else None
    )

    for _ in range(request.batch):
        def request_txt2img(use_adetailer: bool):
            payload = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=-1,
                styles=["string"],
                cfg_scale=request.cfg_scale,
                sampler_index=request.sampler,
                steps=request.steps,
                width=request.width,
                height=request.height,
                batch_size=1,
            )
            if use_adetailer and adetailer_payload:
                payload["adetailer"] = adetailer_payload
            return api.txt2img(**payload)

        try:
            response = request_txt2img(use_adetailer=True)
        except Exception as exc:
            if adetailer_payload and "adetailer" in str(exc).lower():
                global ADETAILER_AVAILABLE
                ADETAILER_AVAILABLE = False
                try:
                    response = request_txt2img(use_adetailer=False)
                except Exception as exc2:
                    raise HTTPException(status_code=500, detail=f"Generation failed: {exc2}")
            else:
                raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:17]
        image_path = OUTPUT_DIR / f"{timestamp}.png"
        response.image.save(image_path)

        try:
            data_list = extraerPrompts.extract_data_from_api_response(str(response))
            if data_list:
                extraerPrompts.write_data_to_txt(data_list, str(OUTPUT_TXT_DIR / f"{timestamp}.txt"))
        except Exception as exc:
            print(f"[extraerPrompts] {exc}")

        if _yolo_cls and _yolo_det and _yolo_seg:
            try:
                AutoEtiquetado.procesar_imagen(
                    str(image_path), _yolo_cls, _yolo_det, _yolo_seg,
                    target_class=request.target_class,
                    class_id=request.class_id,
                )
            except Exception as exc:
                print(f"[AutoEtiquetado] {exc}")

        results.append(GenerateResult(id=timestamp, image_url=f"/assets/{timestamp}.png"))

    return GenerateResponse(
        results=results,
        prompt=prompt,
        negative_prompt=negative_prompt,
        sampler=request.sampler,
        sd_model=request.sd_model,
        width=request.width,
        height=request.height,
        steps=request.steps,
    )


# ─── Dataset endpoints ────────────────────────────────────────────────

@app.get("/api/dataset", response_model=List[DatasetEntry])
def list_dataset():
    if not OUTPUT_DIR.exists():
        return []
    entries = []
    for png in sorted(OUTPUT_DIR.glob("*.png"), reverse=True):
        stem = png.stem
        if "_" in stem:
            continue
        labels = read_labels(OUTPUT_DIR, stem)
        entries.append(DatasetEntry(
            id=stem,
            image_url=f"/assets/{stem}.png",
            timestamp=stem,
            **labels,
        ))
    return entries


@app.post("/api/dataset/export")
def export_dataset(request: DatasetExportRequest):
    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=404, detail="No outputs found")

    if request.ids:
        pngs = [OUTPUT_DIR / f"{i}.png" for i in request.ids if (OUTPUT_DIR / f"{i}.png").exists()]
    else:
        pngs = sorted(OUTPUT_DIR.glob("*.png"))
        pngs = [p for p in pngs if "_" not in p.stem]

    entries = []
    for png in pngs:
        stem = png.stem
        labels = read_labels(OUTPUT_DIR, stem)
        entries.append({"id": stem, **labels})

    zip_bytes = build_yolo_zip(entries, OUTPUT_DIR)

    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=kosei-dataset.zip"},
    )


# ─── Training endpoints ───────────────────────────────────────────────

@app.post("/api/train/upload", response_model=TrainingUploadResponse)
async def train_upload(files: List[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())[:8]
    upload_dir = TRAIN_DATA_DIR / job_id / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    filenames = []
    for f in files:
        if f.content_type and not f.content_type.startswith("image/"):
            continue
        dest = upload_dir / (f.filename or f"img_{len(filenames)}.jpg")
        dest.write_bytes(await f.read())
        filenames.append(dest.name)

    if not filenames:
        raise HTTPException(status_code=400, detail="No valid image files provided")

    return TrainingUploadResponse(job_id=job_id, image_count=len(filenames), filenames=filenames)


@app.post("/api/train/caption", response_model=TrainingCaptionResponse)
def train_caption(request: TrainingCaptionRequest):
    upload_dir = TRAIN_DATA_DIR / request.job_id / "uploads"
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    if not KOHYA_DIR.exists():
        captions = []
        for img in sorted(upload_dir.glob("*.jpg")) + sorted(upload_dir.glob("*.png")):
            captions.append(CaptionEntry(
                filename=img.name,
                caption=f"{request.prefix} {img.stem}",
            ))
        return TrainingCaptionResponse(captions=captions)

    try:
        caption_dict = training_mod.run_blip_caption(upload_dir, request.prefix, KOHYA_DIR)
        captions = [CaptionEntry(filename=fn, caption=cap) for fn, cap in caption_dict.items()]
        return TrainingCaptionResponse(captions=captions)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/train/start")
def train_start(request: TrainingStartRequest):
    upload_dir = TRAIN_DATA_DIR / request.job_id / "uploads"
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    caption_map = {c.filename: c.caption for c in request.captions}
    image_paths = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))

    training_mod.prepare_training_dir(
        TRAIN_DATA_DIR, request.job_id,
        request.token_prefix, request.n_repeats,
        image_paths, caption_map,
    )

    config = {
        "output_name": request.output_name,
        "base_model": request.base_model,
        "network_dim": request.network_dim,
        "learning_rate": request.learning_rate,
        "max_steps": request.max_steps,
        "batch_size": request.batch_size,
    }

    try:
        job = training_mod.start_training_job(
            job_id=request.job_id,
            config=config,
            train_job_dir=TRAIN_DATA_DIR / request.job_id,
            kohya_dir=KOHYA_DIR,
            output_lora_dir=LORA_DIR,
        )
        return {"job_id": job.job_id}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str):
    job = training_mod.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/train/cancel/{job_id}")
def train_cancel(job_id: str):
    training_mod.cancel_job(job_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
