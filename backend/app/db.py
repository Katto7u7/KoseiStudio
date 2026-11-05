"""Persistencia en SQLite. Una conexión por hilo, modo WAL."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable, Optional

from . import config

_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    split_config TEXT,
    split_locked_at REAL,
    split_hash TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    idx INTEGER NOT NULL,
    detector TEXT NOT NULL DEFAULT 'coco',
    detector_prompt TEXT NOT NULL DEFAULT '',
    UNIQUE(project_id, name),
    UNIQUE(project_id, idx)
);
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    origin TEXT NOT NULL CHECK (origin IN ('real', 'synthetic')),
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    normalized_path TEXT,
    thumb_path TEXT,
    sha256 TEXT NOT NULL,
    phash TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    format TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    split TEXT CHECK (split IN ('train', 'val', 'test')),
    status TEXT NOT NULL DEFAULT 'pending',
    class_id INTEGER REFERENCES classes(id) ON DELETE SET NULL,
    class_source TEXT,
    caption TEXT,
    generation TEXT,
    job_id TEXT,
    lora_id TEXT,
    qc TEXT,
    created_at REAL NOT NULL,
    UNIQUE(project_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_images_project ON images(project_id, origin, status);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    cx REAL NOT NULL, cy REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
    polygon TEXT,
    confidence REAL,
    source TEXT NOT NULL,
    mask_source TEXT,
    verified INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_id);
CREATE TABLE IF NOT EXISTS loras (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id) ON DELETE SET NULL,
    name TEXT NOT NULL UNIQUE,
    arch TEXT NOT NULL,
    base_model TEXT,
    token TEXT,
    origin TEXT NOT NULL,
    status TEXT NOT NULL,
    config TEXT,
    train_hashes TEXT,
    job_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    kind TEXT NOT NULL,
    queue TEXT NOT NULL,
    status TEXT NOT NULL,
    params TEXT NOT NULL DEFAULT '{}',
    current INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    result TEXT,
    error TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    retry_of TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, queue, created_at);
CREATE TABLE IF NOT EXISTS builds (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    config TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL,
    job_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS quality_reports (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    report TEXT,
    status TEXT NOT NULL,
    job_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    config TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL,
    job_id TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS experiment_runs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    counts TEXT,
    metrics TEXT,
    error TEXT,
    run_dir TEXT,
    UNIQUE(experiment_id, condition, seed)
);
"""

JSON_FIELDS = {
    "params", "result", "generation", "qc", "config", "summary", "report",
    "metrics", "counts", "polygon", "train_hashes", "split_config",
}


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.ensure_dirs()
        conn = sqlite3.connect(config.DB_PATH, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


def close_thread_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def init() -> None:
    connect().executescript(SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def now() -> float:
    return time.time()


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in JSON_FIELDS & out.keys():
        if isinstance(out[key], str):
            try:
                out[key] = json.loads(out[key])
            except ValueError:
                pass
    return out


def encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def query(sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
    return [_decode(r) for r in connect().execute(sql, tuple(args)).fetchall()]


def one(sql: str, args: Iterable[Any] = ()) -> Optional[dict[str, Any]]:
    row = connect().execute(sql, tuple(args)).fetchone()
    return _decode(row) if row else None


def scalar(sql: str, args: Iterable[Any] = ()) -> Any:
    row = connect().execute(sql, tuple(args)).fetchone()
    return row[0] if row else None


def execute(sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
    return connect().execute(sql, tuple(args))


def insert(table: str, values: dict[str, Any]) -> None:
    cols = ",".join(values)
    marks = ",".join("?" * len(values))
    data = [encode(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()]
    connect().execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", data)


def update(table: str, row_id: Any, values: dict[str, Any]) -> None:
    sets = ",".join(f"{k}=?" for k in values)
    data = [encode(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()]
    connect().execute(f"UPDATE {table} SET {sets} WHERE id=?", [*data, row_id])


class transaction:
    def __enter__(self):
        connect().execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type, exc, tb):
        connect().execute("ROLLBACK" if exc_type else "COMMIT")
