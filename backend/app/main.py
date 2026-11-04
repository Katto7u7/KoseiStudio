"""API de KoseiStudio."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config, db, jobs
from .api import dataset, engine, experiments, loras, outputs, projects, review, system
from .services import export, features, forge, generation, ingest, labeling, quality, training  # noqa: F401
from .services import experiment as experiment_service  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("kosei")


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    db.init()
    interrupted = jobs.recover()
    if interrupted:
        log.warning("%s trabajos quedaron interrumpidos por el reinicio", interrupted)
    training.sync_external_loras()
    jobs.start_workers()
    yield
    jobs.stop_workers()


app = FastAPI(title="KoseiStudio API", version="1.0", lifespan=lifespan)


@app.exception_handler(forge.ForgeUnavailable)
async def _forge_unavailable(_: Request, exc: forge.ForgeUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc), "code": "engine_unavailable"})


@app.exception_handler(forge.ForgeError)
async def _forge_error(_: Request, exc: forge.ForgeError):
    return JSONResponse(status_code=502, content={"detail": str(exc), "code": "engine_error"})


for module in (system, projects, dataset, engine, loras, review, outputs, experiments):
    app.include_router(module.router, prefix="/api")
