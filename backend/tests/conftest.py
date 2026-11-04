"""Pruebas de API con SQLite y disco temporales. Forge se sustituye por un doble: son pruebas SIMULADAS.

Las pruebas con modelos reales viven en tests/test_real_*.py y se marcan con `real`.
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pytest_configure(config):
    config.addinivalue_line("markers", "real: ejecuta modelos reales, requiere pesos y tarda")


def _reset_module_state(tmp: Path, monkeypatch):
    from app import config, db

    monkeypatch.setattr(config, "DATA_DIR", tmp)
    monkeypatch.setattr(config, "DB_PATH", tmp / "kosei.db")
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp / "projects")
    monkeypatch.setattr(config, "TMP_DIR", tmp / "tmp")
    monkeypatch.setattr(config, "LORA_DIR", tmp / "lora")
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp / "ckpt")
    (tmp / "lora").mkdir()
    (tmp / "ckpt").mkdir()
    db.close_thread_connection()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    _reset_module_state(tmp_path, monkeypatch)
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    from app import db
    db.close_thread_connection()


@pytest.fixture()
def project(client):
    return client.post("/api/projects", json={"name": "prueba"}).json()


def png_bytes(seed: int = 0, size: int = 128, fmt: str = "PNG") -> bytes:
    """Imagen con estructura propia por seed: pHash distinto, no solo ruido."""
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    image = Image.fromarray(cells).resize((size, size), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    image.save(buf, fmt)
    return buf.getvalue()


def wait_job(client, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"El trabajo {job_id} no termino: {job}")


def import_images(client, project_id: str, n: int, start: int = 0) -> dict:
    files = [("files", (f"img_{i}.png", png_bytes(start + i), "image/png")) for i in range(start, start + n)]
    res = client.post(f"/api/projects/{project_id}/import", files=files)
    assert res.status_code == 202, res.text
    return wait_job(client, res.json()["job_id"])


class FakeForge:
    """Doble de Forge: responde como la API real pero sin GPU."""

    def __init__(self):
        self.calls: list[dict] = []
        self.fail_at: int | None = None

    def txt2img(self, payload):
        if self.fail_at is not None and len(self.calls) >= self.fail_at:
            from app.services.forge import ForgeError
            raise ForgeError("fallo simulado")
        self.calls.append(payload)
        n = payload["batch_size"]
        images = [
            Image.open(io.BytesIO(png_bytes(payload["seed"] + i + 1000, payload["width"]))).convert("RGB")
            for i in range(n)
        ]
        return images, {"all_seeds": [payload["seed"] + i for i in range(n)], "sd_model_hash": "abc"}
