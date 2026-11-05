"""Cola de trabajos persistente: una cola GPU y una CPU, cada una con un hilo.

La GPU es una sola, asi que serializar es correcto y evita agotar VRAM. Al arrancar,
lo que quedo corriendo pasa a `interrupted` porque su proceso ya no existe.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Callable, Optional

from . import config, db

log = logging.getLogger("kosei.jobs")

ACTIVE = ("queued", "running")
TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")


class Cancelled(Exception):
    """La lanza un handler al detectar que el usuario cancelo."""


class JobContext:
    def __init__(self, job: dict[str, Any]):
        self.id: str = job["id"]
        self.project_id: Optional[str] = job["project_id"]
        self.params: dict[str, Any] = job["params"] or {}
        self.retry_of: Optional[str] = job["retry_of"]
        self._on_cancel: Optional[Callable[[], None]] = None
        self.log_path = config.DATA_DIR / "logs" / f"{self.id}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_cancel(self, fn: Optional[Callable[[], None]]) -> None:
        self._on_cancel = fn

    def cancelled(self) -> bool:
        return bool(db.scalar("SELECT cancel_requested FROM jobs WHERE id=?", [self.id]))

    def check_cancel(self) -> None:
        if self.cancelled():
            raise Cancelled()

    def progress(self, current: int, total: int, message: str = "") -> None:
        db.update("jobs", self.id, {"current": current, "total": total, "message": message[:500]})

    def set_result(self, result: dict[str, Any]) -> None:
        db.update("jobs", self.id, {"result": result})

    def log(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(line.rstrip("\n") + "\n")


_handlers: dict[str, tuple[str, Callable[[JobContext], Optional[dict]]]] = {}
_running: dict[str, JobContext] = {}
_stop = threading.Event()
_threads: list[threading.Thread] = []


def register(kind: str, queue: str):
    def deco(fn):
        _handlers[kind] = (queue, fn)
        return fn
    return deco


def enqueue(kind: str, project_id: Optional[str], params: dict[str, Any], retry_of: Optional[str] = None) -> str:
    queue = _handlers[kind][0]
    job_id = db.new_id()
    db.insert("jobs", {
        "id": job_id, "project_id": project_id, "kind": kind, "queue": queue,
        "status": "queued", "params": params, "retry_of": retry_of, "created_at": db.now(),
    })
    return job_id


def active_job(project_id: Optional[str], kind: str) -> Optional[dict[str, Any]]:
    return db.one(
        "SELECT * FROM jobs WHERE kind=? AND status IN ('queued','running') AND project_id IS ? "
        "ORDER BY created_at LIMIT 1",
        [kind, project_id],
    )


def cancel(job_id: str) -> Optional[dict[str, Any]]:
    job = db.one("SELECT * FROM jobs WHERE id=?", [job_id])
    if not job:
        return None
    if job["status"] == "queued":
        db.update("jobs", job_id, {
            "status": "cancelled", "cancel_requested": 1,
            "finished_at": db.now(), "message": "Cancelado antes de iniciar",
        })
    elif job["status"] == "running":
        db.update("jobs", job_id, {"cancel_requested": 1, "message": "Cancelando..."})
        ctx = _running.get(job_id)
        if ctx and ctx._on_cancel:
            try:
                ctx._on_cancel()
            except Exception:
                log.exception("on_cancel fallo en %s", job_id)
    return db.one("SELECT * FROM jobs WHERE id=?", [job_id])


def retry(job_id: str) -> Optional[str]:
    job = db.one("SELECT * FROM jobs WHERE id=?", [job_id])
    if not job or job["status"] in ACTIVE:
        return None
    return enqueue(job["kind"], job["project_id"], job["params"] or {}, retry_of=job_id)


def _run_one(queue: str) -> bool:
    with db.transaction():
        job = db.one(
            "SELECT * FROM jobs WHERE status='queued' AND queue=? ORDER BY created_at LIMIT 1",
            [queue],
        )
        if not job:
            return False
        db.update("jobs", job["id"], {"status": "running", "started_at": db.now(), "message": "Iniciando"})
    ctx = JobContext(job)
    _running[job["id"]] = ctx
    handler = _handlers.get(job["kind"])
    try:
        if handler is None:
            raise RuntimeError(f"Tipo de trabajo desconocido: {job['kind']}")
        result = handler[1](ctx)
        ctx.check_cancel()
        values: dict[str, Any] = {"status": "succeeded", "finished_at": db.now(), "message": "Completado"}
        if result is not None:
            values["result"] = result
        db.update("jobs", job["id"], values)
    except Cancelled:
        db.update("jobs", job["id"], {
            "status": "cancelled", "finished_at": db.now(), "message": "Cancelado por el usuario",
        })
    except Exception as exc:
        log.exception("Trabajo %s (%s) fallo", job["id"], job["kind"])
        ctx.log(traceback.format_exc())
        db.update("jobs", job["id"], {
            "status": "failed", "finished_at": db.now(),
            "error": str(exc)[:4000], "message": "Falló",
        })
    finally:
        _running.pop(job["id"], None)
    return True


def _worker(queue: str) -> None:
    while not _stop.is_set():
        try:
            if not _run_one(queue):
                _stop.wait(0.5)
        except Exception:
            log.exception("Error en el worker %s", queue)
            _stop.wait(2)
    db.close_thread_connection()


def recover() -> int:
    """Marca como interrumpido lo que corria cuando murio el proceso anterior."""
    cur = db.execute(
        "UPDATE jobs SET status='interrupted', finished_at=?, "
        "message='Interrumpido por un reinicio. Se puede reintentar.' WHERE status='running'",
        [db.now()],
    )
    return cur.rowcount


def start_workers() -> None:
    _stop.clear()
    for queue in ("gpu", "cpu"):
        t = threading.Thread(target=_worker, args=(queue,), name=f"worker-{queue}", daemon=True)
        t.start()
        _threads.append(t)


def stop_workers(timeout: float = 5.0) -> None:
    _stop.set()
    for ctx in list(_running.values()):
        if ctx._on_cancel:
            try:
                ctx._on_cancel()
            except Exception:
                log.exception("Cierre: on_cancel fallo")
    deadline = time.time() + timeout
    for t in _threads:
        t.join(max(0, deadline - time.time()))
    _threads.clear()
