"""Prueba de sistema REAL por nginx: GPU, Forge, kohya, YOLO y SAM de verdad, sin dobles.

Crea un proyecto temporal de pruebas, recorre todo el flujo y lo verifica. Aprueba
etiquetas en lote para no pedir revision humana: el resultado de este experimento NO es
evidencia para la tesis, solo demuestra que la ruta funciona.

Uso: python scripts/e2e_system.py <carpeta_con_imagenes> [--keep]
"""
from __future__ import annotations

import glob
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from client import call, upload, wait  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'OK' if ok else 'FALLO'}] {name} {detail}", flush=True)


def expect_refused(method, path, body, needle):
    try:
        call(method, path, body)
        return False, "no se rechazo"
    except SystemExit as exc:
        return needle in str(exc), str(exc)[:140]


def main() -> int:
    folder, keep = sys.argv[1], "--keep" in sys.argv
    files = sorted(glob.glob(str(Path(folder) / "*.jpg")))[:14]
    t0 = time.time()

    project = call("POST", "/projects", {"name": f"E2E {int(t0)}", "description": "Prueba de sistema; se puede borrar"})
    pid = project["id"]
    cls = call("POST", f"/projects/{pid}/classes", {"name": "persona", "detector": "coco", "detector_prompt": "person"})

    # Dataset
    job = wait(upload(f"/projects/{pid}/import", files)["job_id"], 1, False)
    check("importar", job["status"] == "succeeded" and job["result"]["imported"] == len(files), str(job["result"]["imported"]))
    report = call("POST", f"/projects/{pid}/split", {"train": 6, "val": 3, "test": 5, "seed": 0})
    check("particion sin fuga", not report["cross_split_near_duplicates"], str(report["counts"]))
    call("POST", f"/projects/{pid}/split/lock")
    job = wait(call("POST", f"/projects/{pid}/label")["job_id"], 2, False)
    check("etiquetado YOLO y SAM reales", job["status"] == "succeeded", str(job["result"]))

    # Aprobacion en lote: real solo en esta prueba.
    for status in ("candidate", "needs_review"):
        call("POST", f"/projects/{pid}/review/bulk", {"from_status": status, "origin": "real", "status": "approved"})
    pending = call("GET", f"/projects/{pid}")["pipeline"]["by_status"]["real"]
    for image in call("GET", f"/projects/{pid}/images?origin=real&status=no_detection&limit=50")["items"]:
        call("POST", f"/images/{image['id']}/review", {"status": "approved", "negative": True})

    job = wait(call("POST", f"/projects/{pid}/captions", {"token": "e2etoken", "overwrite": True})["job_id"], 3, False)
    check("captions BLIP reales", job["status"] == "succeeded" and job["result"]["captioned"] == 6, str(job["result"]))
    caption = call("GET", f"/projects/{pid}/images?origin=real&split=train&limit=1")["items"][0]["caption"]
    check("caption con token", caption.startswith("e2etoken "), caption[:80])

    # LoRA
    models = call("GET", "/engine/models")
    sd = next(m for m in models if m["name"].startswith("v1-5"))
    xl = next(m for m in models if m["arch"] == "sdxl")
    name = f"e2e_sd15_{int(t0)}"
    tr = call("POST", f"/projects/{pid}/loras", {
        "name": name, "checkpoint": sd["title"], "token": "e2etoken", "steps": 60, "repeats": 5, "network_dim": 8,
        "network_alpha": 4, "resolution": 512, "save_every": 30,
    })
    job = wait(tr["job_id"], 5, False)
    check("entrenamiento kohya SD 1.5 real", job["status"] == "succeeded", str(job["result"] or job["error"])[:200])
    lora = next(l for l in call("GET", f"/loras?project_id={pid}") if l["id"] == tr["lora_id"])
    check("linaje del LoRA", lora["arch"] == "sd15" and lora["train_image_count"] == 6, f"{lora['arch']}, {lora['train_image_count']} imagenes")
    losses = call("GET", f"/loras/{lora['id']}/losses")
    check("perdida real registrada", len(losses) > 10, f"{len(losses)} puntos")

    # Cancelacion de un entrenamiento real
    cancel_name = f"e2e_cancel_{int(t0)}"
    tr2 = call("POST", f"/projects/{pid}/loras", {
        "name": cancel_name, "checkpoint": sd["title"], "token": "e2etoken", "steps": 3000, "repeats": 5, "network_dim": 8,
        "network_alpha": 4, "resolution": 512, "save_every": 1000,
    })
    while call("GET", f"/jobs/{tr2['job_id']}")["current"] < 3:
        time.sleep(3)
    call("POST", f"/jobs/{tr2['job_id']}/cancel")
    job = wait(tr2["job_id"], 2, False)
    gpu_after = call("GET", "/system")["gpu"]
    check("cancelar entrenamiento", job["status"] == "cancelled", f"VRAM en uso {gpu_after['memory_used_mib']} MiB")

    # Compatibilidad y generacion
    base = {
        "class_id": cls["id"], "prompt": "photograph of a man", "negative_prompt": "cartoon, deformed",
        "scenarios": {"fondo": ["street", "office", "park"], "luz": ["daylight", "night"]},
        "count": 12, "seed": 777, "steps": 12, "cfg_scale": 6, "width": 512, "height": 512,
        "sampler": "DPM++ 2M", "scheduler": "karras", "batch_size": 2, "lora_weight": 0.8,
    }
    ok, detail = expect_refused("POST", f"/projects/{pid}/generate", {**base, "checkpoint": xl["title"], "lora_id": lora["id"]}, "Incompatible")
    check("rechaza LoRA incompatible", ok, detail)
    gen = call("POST", f"/projects/{pid}/generate", {**base, "checkpoint": sd["title"], "lora_id": lora["id"]})
    job = wait(gen["job_id"], 3, False)
    check("generacion real con LoRA", job["status"] == "succeeded" and job["result"]["generated"] == 12, str(job["result"] or job["error"])[:200])
    jobs_now = call("GET", f"/jobs?project_id={pid}&limit=5")
    for j in jobs_now:
        if j["kind"] == "label" and j["status"] in ("queued", "running"):
            wait(j["id"], 2, False)
    images = call("GET", f"/projects/{pid}/images?origin=synthetic&limit=50")["items"]
    check("dimensiones y metadatos", all(i["width"] == 512 and i["generation"]["seed"] >= 777 and i["generation"]["lora"] == name for i in images), f"{len(images)} imagenes")
    check("seeds unicas", len({i["generation"]["seed"] for i in images}) == 12)
    prov = call("GET", f"/images/{images[0]['id']}/provenance")
    check("procedencia PNG y sha256", prov["sha256_matches"] and prov["png_record"]["synthetic"] is True, "")
    check("marca de agua detectada", prov["watermark_detected"] is True, str(prov["watermark_detected"]))
    real_img = call("GET", f"/projects/{pid}/images?origin=real&limit=1")["items"][0]
    prov_real = call("GET", f"/images/{real_img['id']}/provenance")
    check("una foto real no da positivo", prov_real["watermark_detected"] is False and prov_real["png_record"] is None)

    # Cancelacion de una generacion
    g2 = call("POST", f"/projects/{pid}/generate", {**base, "count": 100, "steps": 30, "seed": 5000, "checkpoint": sd["title"], "lora_id": lora["id"], "batch_size": 1})
    while call("GET", f"/jobs/{g2['job_id']}")["current"] < 2:
        time.sleep(1)
    call("POST", f"/jobs/{g2['job_id']}/cancel")
    job = wait(g2["job_id"], 1, False)
    check("cancelar generacion", job["status"] == "cancelled", f"guardadas {job['current']}")
    time.sleep(1)
    for j in call("GET", f"/jobs?project_id={pid}&limit=5"):
        if j["kind"] == "label" and j["status"] in ("queued", "running"):
            wait(j["id"], 2, False)

    # Revision sinteticas
    for status in ("candidate", "needs_review"):
        call("POST", f"/projects/{pid}/review/bulk", {"from_status": status, "origin": "synthetic", "status": "approved"})
    approved = call("GET", f"/projects/{pid}")["pipeline"]["by_status"]["synthetic"].get("approved", 0)
    check("sinteticas aprobadas", approved > 0, str(approved))

    # Calidad y caracteristicas
    q = call("POST", f"/projects/{pid}/quality")
    job = wait(q["job_id"], 3, False)
    report = next(r for r in call("GET", f"/projects/{pid}/quality") if r["id"] == q["report_id"])
    check("calidad con FID y KID reales", job["status"] == "succeeded" and report["report"]["inception"]["fid"] > 0, json.dumps(report.get("report", {}).get("inception"))[:120] if report.get("report") else str(job["error"])[:200])
    job = wait(call("POST", f"/projects/{pid}/features")["job_id"], 3, False)
    check("mapa CLIP", job["status"] == "succeeded" and len(job["result"]["points"]) > 14, str(len((job["result"] or {}).get("points", []))))

    # Construccion y descarga
    b = call("POST", f"/projects/{pid}/builds", {"name": "e2e", "seed": 0, "tasks": ["detect", "segment", "classify"]})
    job = wait(b["job_id"], 2, False)
    check("construccion validada", job["status"] == "succeeded" and job["result"]["validation"]["ok"], str(job["result"]["validation"] if job["result"] else job["error"])[:200])
    with urllib.request.urlopen(f"http://localhost/api/builds/{b['build_id']}/download") as res:
        blob = res.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        leaked = [i for i in manifest["items"] if i["origin"] == "synthetic" and i["split"] != "train"]
        check("ZIP: sinteticas solo en train", not leaked and any(i["origin"] == "synthetic" for i in manifest["items"]), f"{len(manifest['items'])} items")
    out = Path(__file__).parent / "_e2e_build.zip"
    out.write_bytes(blob)
    print("ZIP para probar con yolo:", out, b["build_id"], flush=True)

    # Experimento pequeno y real
    exp = call("POST", f"/projects/{pid}/experiments", {
        "name": "e2e", "conditions": [{"name": "A", "n_synthetic": 0, "use_real": True}, {"name": "B", "n_synthetic": 6, "use_real": True}],
        "baseline": "A", "seeds": [0, 1], "epochs": 3, "imgsz": 320, "batch": 4,
    })
    job = wait(exp["job_id"], 5, False)
    row = next(e for e in call("GET", f"/projects/{pid}/experiments") if e["id"] == exp["experiment_id"])
    check("experimento real corrido", job["status"] == "succeeded" and row["status"] == "ready", str(job["error"])[:300])
    if row.get("summary"):
        check("resumen con veredicto calculado", "B" in row["summary"]["comparisons"], json.dumps(row["summary"]["comparisons"])[:160])

    # Las sinteticas no pueden tocar test: comprobado en el experimento real.
    runs = {r["condition"]: r["counts"] for r in row["runs"]}
    check("A sin sinteticas y test solo real", runs["A"]["detect"]["train"]["synthetic"] == 0 if "detect" in runs["A"] else runs["A"]["train"]["synthetic"] == 0, "")

    print(f"\nTiempo total: {round(time.time() - t0)} s. Proyecto: {pid}")
    bad = [r for r in results if not r[1]]
    print(f"{len(results) - len(bad)} de {len(results)} comprobaciones correctas")
    if not keep and not bad:
        call("DELETE", f"/projects/{pid}")
        for lora_row in call("GET", "/loras"):
            if lora_row["name"] in (name, cancel_name):
                call("DELETE", f"/loras/{lora_row['id']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
