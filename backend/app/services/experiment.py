"""Experimento comparativo: la misma arquitectura y metodologia con distintos datos de train.

Val y test son reales, verificados por una persona y bloqueados. Los resultados son los
que salgan: el veredicto se calcula con una regla fijada de antemano sobre la prueba real.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from .. import db
from ..jobs import Cancelled, JobContext, register
from . import export, forge, gpu, storage

log = logging.getLogger("kosei.experiment")

PRIMARY_METRIC = "map50_95"
METRICS = ("map50_95", "map50", "precision", "recall", "f1")
# t de Student de dos colas al 95 % para n-1 grados de libertad.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def preflight(project_id: str, conditions: list[dict[str, Any]]) -> dict[str, Any]:
    """Comprueba que el experimento sea valido antes de gastar GPU."""
    project = db.one("SELECT * FROM projects WHERE id=?", [project_id])
    problems: list[str] = []
    if not project["split_locked_at"]:
        problems.append("La partición debe estar bloqueada antes de experimentar")
    counts = {
        s: int(db.scalar(
            "SELECT COUNT(*) FROM images WHERE project_id=? AND origin='real' AND split=? AND status='approved'",
            [project_id, s],
        ))
        for s in ("train", "val", "test")
    }
    unverified = int(db.scalar(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND origin='real' AND split IN ('val','test') AND status!='approved'",
        [project_id],
    ))
    if unverified:
        problems.append(f"{unverified} imágenes de val o test sin aprobar: son la verdad de referencia")
    if not counts["val"] or not counts["test"]:
        problems.append("Faltan imágenes aprobadas en val o test")
    synth_available = int(db.scalar(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND origin='synthetic' AND status='approved'",
        [project_id],
    ))
    needed = max((int(c.get("n_synthetic") or 0) for c in conditions), default=0)
    if needed > synth_available:
        problems.append(f"Se piden {needed} sintéticas y hay {synth_available} aprobadas")

    protected = {
        r["sha256"] for r in db.query(
            "SELECT sha256 FROM images WHERE project_id=? AND origin='real' AND split IN ('val','test')",
            [project_id],
        )
    }
    train_real = {
        r["sha256"] for r in db.query(
            "SELECT sha256 FROM images WHERE project_id=? AND origin='real' AND split='train'",
            [project_id],
        )
    }
    if protected & train_real:
        problems.append("Hay imágenes compartidas entre train y val/test")

    # El LoRA de cada sintética aprobada no puede haber visto val ni test.
    lora_ids = [
        r["lora_id"] for r in db.query(
            "SELECT DISTINCT lora_id FROM images WHERE project_id=? AND origin='synthetic' AND status='approved'",
            [project_id],
        )
    ]
    unverifiable: list[str] = []
    for lora_id in lora_ids:
        if not lora_id:
            continue
        lora = db.one("SELECT name, train_hashes, origin FROM loras WHERE id=?", [lora_id])
        if not lora:
            continue
        if lora["origin"] == "external" or not lora["train_hashes"]:
            unverifiable.append(lora["name"])
        elif protected & set(lora["train_hashes"]):
            problems.append(f"El LoRA {lora['name']} se entreno con imágenes de val o test")
    warnings = []
    if unverifiable:
        warnings.append("Linaje no verificable en: " + ", ".join(unverifiable) + ". No se puede descartar fuga de datos.")
    if counts["test"] < 30:
        warnings.append(f"Solo {counts['test']} imágenes de prueba: la incertidumbre es alta.")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "counts": counts, "synthetic_approved": synth_available}


def summarize(runs: list[dict[str, Any]], baseline: str) -> dict[str, Any]:
    """Media, desviacion y comparacion pareada por seed contra la condición base."""
    done = [r for r in runs if r["status"] == "succeeded" and r["metrics"]]
    conditions: dict[str, Any] = {}
    for name in sorted({r["condition"] for r in done}):
        rows = [r for r in done if r["condition"] == name]
        entry: dict[str, Any] = {"runs": len(rows), "counts": rows[0]["counts"]}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in rows]
            entry[metric] = {
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
                "values": [round(v, 4) for v in values],
            }
        conditions[name] = entry

    comparisons: dict[str, Any] = {}
    base = {r["seed"]: r["metrics"][PRIMARY_METRIC] for r in done if r["condition"] == baseline}
    for name in conditions:
        if name == baseline:
            continue
        pairs = [
            r["metrics"][PRIMARY_METRIC] - base[r["seed"]]
            for r in done if r["condition"] == name and r["seed"] in base
        ]
        comparisons[name] = _verdict(pairs)
    return {"primary_metric": PRIMARY_METRIC, "baseline": baseline, "conditions": conditions, "comparisons": comparisons}


def _verdict(diffs: list[float]) -> dict[str, Any]:
    """Regla fijada de antemano: solo hay mejora o empeora si el IC 95 % excluye el cero."""
    if not diffs:
        return {"verdict": "sin_datos", "pairs": 0}
    mean = statistics.fmean(diffs)
    if len(diffs) < 2:
        return {"verdict": "no_concluyente", "pairs": 1, "mean_delta": round(mean, 4), "ci95": None}
    sd = statistics.stdev(diffs)
    half = T_95.get(len(diffs) - 1, 2.0) * sd / math.sqrt(len(diffs))
    low, high = mean - half, mean + half
    verdict = "mejora" if low > 0 else "empeora" if high < 0 else "sin_cambio_concluyente"
    return {
        "verdict": verdict, "pairs": len(diffs), "mean_delta": round(mean, 4),
        "ci95": [round(low, 4), round(high, 4)],
    }


@register("experiment", "gpu")
def run_experiment(ctx: JobContext) -> dict[str, Any]:
    p = ctx.params
    exp_id, project_id = p["experiment_id"], ctx.project_id
    experiment = db.one("SELECT * FROM experiments WHERE id=?", [exp_id])
    config_ = experiment["config"]
    conditions, seeds = config_["conditions"], config_["seeds"]

    check = preflight(project_id, conditions)
    if not check["ok"]:
        db.update("experiments", exp_id, {"status": "failed"})
        raise ValueError("; ".join(check["problems"]))

    root = storage.project_dir(project_id) / "experiments" / exp_id
    classes = db.query("SELECT * FROM classes WHERE project_id=? ORDER BY idx", [project_id])
    db.update("experiments", exp_id, {"status": "running"})
    ctx.progress(0, 1, "Liberando VRAM del motor")
    gpu.free_for_training(2500, ctx.log)

    plan = [(c, s) for c in conditions for s in seeds]
    epochs = int(config_["epochs"])
    total = len(plan) * epochs
    finished = 0
    try:
        for index, (condition, seed) in enumerate(plan):
            ctx.check_cancel()
            run = db.one("SELECT * FROM experiment_runs WHERE experiment_id=? AND condition=? AND seed=?", [exp_id, condition["name"], seed])
            if run and run["status"] == "succeeded":
                finished += epochs
                continue
            run_id = run["id"] if run else db.new_id()
            run_dir = root / f"{condition['name']}_seed{seed}"
            if run_dir.exists():
                shutil.rmtree(run_dir)

            # Las sintéticas se eligen con una seed propia del experimento: subconjuntos anidados.
            selection = export.select_images(project_id, int(condition.get("n_synthetic") or 0), int(config_.get("selection_seed", 0)))
            if not condition.get("use_real", True):
                selection["train"] = [i for i in selection["train"] if i["origin"] == "synthetic"]
            if not selection["train"]:
                raise ValueError(f"La condición {condition['name']} quedo sin imágenes de train")
            written = export.write_tree(run_dir / "data", selection, classes, ["detect"], ctx)
            counts = written["counts"]["detect"]
            values = {
                "experiment_id": exp_id, "condition": condition["name"], "seed": seed,
                "status": "running", "counts": counts, "run_dir": storage.rel(run_dir), "error": None,
            }
            if run:
                db.update("experiment_runs", run_id, values)
            else:
                db.insert("experiment_runs", {"id": run_id, **values})

            job_cfg = {
                "data": str(run_dir / "data" / "detect" / "data.yaml"),
                "model": config_.get("model", "yolo11n.yaml"),
                "epochs": epochs, "imgsz": int(config_["imgsz"]), "batch": int(config_["batch"]),
                "seed": seed, "project": str(run_dir), "patience": int(config_.get("patience", 0)),
            }
            cfg_path = run_dir / "run.json"
            cfg_path.write_text(json.dumps(job_cfg), encoding="utf-8")

            def on_epoch(epoch: int, base: int = finished, name: str = condition["name"], s: int = seed) -> None:
                ctx.progress(base + epoch, total, f"{name}, seed {s}: época {epoch} de {epochs}")

            metrics = _run_worker(ctx, cfg_path, run_dir / "metrics.json", on_epoch)
            db.update("experiment_runs", run_id, {"status": "succeeded", "metrics": metrics})
            finished += epochs
            summary = summarize(db.query("SELECT * FROM experiment_runs WHERE experiment_id=?", [exp_id]), config_["baseline"])
            db.update("experiments", exp_id, {"summary": {**summary, "preflight": check}})
    except Cancelled:
        db.execute("UPDATE experiment_runs SET status='cancelled' WHERE experiment_id=? AND status='running'", [exp_id])
        db.update("experiments", exp_id, {"status": "cancelled"})
        raise
    except Exception as exc:
        db.execute("UPDATE experiment_runs SET status='failed', error=? WHERE experiment_id=? AND status='running'", [str(exc)[:2000], exp_id])
        db.update("experiments", exp_id, {"status": "failed"})
        raise
    summary = summarize(db.query("SELECT * FROM experiment_runs WHERE experiment_id=?", [exp_id]), config_["baseline"])
    db.update("experiments", exp_id, {"status": "ready", "summary": {**summary, "preflight": check}})
    return {"experiment_id": exp_id, "runs": len(plan)}


def _run_worker(ctx: JobContext, cfg_path: Path, metrics_path: Path, on_epoch) -> dict[str, Any]:
    """Un subproceso por corrida: al terminar se libera toda la VRAM."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "app.workers.yolo_run", str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
        start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    def terminate() -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    ctx.on_cancel(terminate)
    tail: list[str] = []
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            match = re.match(r"KOSEI_EPOCH (\d+)", line)
            if match:
                on_epoch(int(match.group(1)))
                continue
            if line:
                tail = (tail + [line])[-40:]
                ctx.log(line)
        code = proc.wait()
    finally:
        ctx.on_cancel(None)
    if ctx.cancelled():
        raise Cancelled()
    if code != 0 or not metrics_path.exists():
        raise RuntimeError(f"La corrida fallo (código {code}):\n" + "\n".join(tail[-10:]))
    return json.loads(metrics_path.read_text(encoding="utf-8"))
