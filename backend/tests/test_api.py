"""Pruebas SIMULADAS de API, persistencia, seguridad y flujo. Forge es un doble."""
from __future__ import annotations

import io
import json
import zipfile

import pytest
import yaml
from PIL import Image

from app import db, jobs
from app.services import export, forge, generation, ingest, labeling, splits, storage
from conftest import FakeForge, import_images, png_bytes, wait_job


def fake_label(monkeypatch, boxes=None, polygon=True):
    """Sustituye al detector y a SAM; lo que se prueba es la logica que los rodea."""
    default = [{"cx": .5, "cy": .5, "w": .4, "h": .6, "confidence": .93}]
    monkeypatch.setattr(labeling, "detect", lambda path, cls: (default if boxes is None else boxes, "doble-detector"))
    poly = [.3, .2, .7, .2, .7, .8, .3, .8] if polygon else None
    monkeypatch.setattr(labeling, "segment", lambda path, b, w, h: [poly for _ in b])


def add_class(client, pid, name="persona"):
    res = client.post(f"/api/projects/{pid}/classes", json={"name": name, "detector_prompt": "person"})
    assert res.status_code == 201
    return res.json()


def ready_project(client, monkeypatch, pid, n=10):
    fake_label(monkeypatch)
    add_class(client, pid)
    import_images(client, pid, n)
    assert client.post(f"/api/projects/{pid}/split", json={"train": 4, "val": 2, "test": 4, "seed": 1}).status_code == 200
    assert client.post(f"/api/projects/{pid}/split/lock").status_code == 200
    job = client.post(f"/api/projects/{pid}/label").json()["job_id"]
    wait_job(client, job)
    res = client.post(f"/api/projects/{pid}/review/bulk", json={"from_status": "candidate", "origin": "real", "status": "approved"})
    assert res.json()["changed"] == n


# ---------- Proyectos y persistencia ----------

def test_project_crud_and_persistence(client):
    created = client.post("/api/projects", json={"name": "Uno", "description": "d"}).json()
    assert created["pipeline"]["real_images"] == 0
    assert client.get(f"/api/projects/{created['id']}").json()["name"] == "Uno"
    assert client.patch(f"/api/projects/{created['id']}", json={"name": "Dos"}).json()["name"] == "Dos"
    # Una nueva conexion ve lo mismo: esta en disco, no en memoria.
    db.close_thread_connection()
    assert db.one("SELECT name FROM projects WHERE id=?", [created["id"]])["name"] == "Dos"
    assert client.delete(f"/api/projects/{created['id']}").status_code == 204
    assert client.get(f"/api/projects/{created['id']}").status_code == 404


def test_validation_rejects_empty_name(client):
    assert client.post("/api/projects", json={"name": ""}).status_code == 422


# ---------- Importacion ----------

def test_import_reports_corrupt_and_duplicates(client, project):
    pid = project["id"]
    files = [
        ("files", ("buena.png", png_bytes(1), "image/png")),
        ("files", ("copia.png", png_bytes(1), "image/png")),
        ("files", ("rota.png", b"esto no es una imagen", "image/png")),
        ("files", ("disfrazada.jpg", b"MZ\x90\x00 ejecutable", "image/jpeg")),
    ]
    job = wait_job(client, client.post(f"/api/projects/{pid}/import", files=files).json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["imported"] == 1
    assert len(job["result"]["duplicates"]) == 1
    assert {c["file"][6:] for c in job["result"]["corrupt"]} == {"rota.png", "disfrazada.jpg"}


def test_original_is_preserved_and_read_only(client, project):
    pid = project["id"]
    data = png_bytes(5)
    job = wait_job(client, client.post(f"/api/projects/{pid}/import", files=[("files", ("a.png", data, "image/png"))]).json()["job_id"])
    assert job["status"] == "succeeded"
    row = db.one("SELECT * FROM images WHERE project_id=?", [pid])
    original = storage.resolve(row["path"])
    assert original.read_bytes() == data
    assert not original.stat().st_mode & 0o222


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_zip_slip_is_rejected(client, project, tmp_path):
    payload = _zip({"../../fuera.png": png_bytes(2), "ok.png": png_bytes(3)})
    res = client.post(f"/api/projects/{project['id']}/import", files=[("files", ("x.zip", payload, "application/zip"))])
    job = wait_job(client, res.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["result"]["imported"] == 0
    assert "insegura" in job["result"]["corrupt"][0]["reason"]
    assert not list(tmp_path.parent.glob("fuera.png"))


def test_zip_bomb_limit(tmp_path, monkeypatch, client):
    from app import config
    monkeypatch.setattr(config, "MAX_ZIP_UNCOMPRESSED", 1000)
    archive = tmp_path / "b.zip"
    archive.write_bytes(_zip({"grande.bin": b"0" * 5000}))
    with pytest.raises(ingest.UnsafeArchive):
        ingest.extract_zip(archive, tmp_path / "x")


def test_zip_with_yolo_labels(client, project):
    pid = project["id"]
    payload = _zip({
        "ds/data.yaml": yaml.safe_dump({"names": ["persona", "auto"]}).encode(),
        "ds/images/train/a.png": png_bytes(10),
        "ds/labels/train/a.txt": b"1 0.5 0.5 0.2 0.2\n",
        "ds/images/train/b.png": png_bytes(11),
        "ds/labels/train/b.txt": b"0 0.1 0.1 0.9 0.1 0.9 0.9\n9 mal formada\n",
    })
    job = wait_job(client, client.post(f"/api/projects/{pid}/import", files=[("files", ("d.zip", payload, "application/zip"))]).json()["job_id"])
    assert job["result"]["imported"] == 2
    assert job["result"]["labels_imported"] == 2
    names = {c["name"]: c["idx"] for c in client.get(f"/api/projects/{pid}").json()["classes"]}
    assert names == {"persona": 0, "auto": 1}
    rows = db.query("SELECT a.*, i.status FROM annotations a JOIN images i ON i.id=a.image_id")
    assert all(r["verified"] == 0 for r in rows), "Una etiqueta importada no es verdad verificada"
    assert all(r["status"] == "needs_review" for r in rows)


@pytest.mark.parametrize("line,ok", [
    ("0 0.5 0.5 0.2 0.2", True),
    ("0 0.1 0.1 0.9 0.1 0.9 0.9", True),
    ("0 0.5 0.5 0.2", False),
    ("0 1.5 0.5 0.2 0.2", False),
    ("-1 0.5 0.5 0.2 0.2", False),
    ("a b c d e", False),
    ("0 0.1 0.1 0.9 0.1 0.9", False),
])
def test_yolo_line_validation(line, ok):
    assert (ingest.parse_yolo_line(line) is not None) is ok


# ---------- Rutas y archivos ----------

def test_path_traversal_blocked():
    with pytest.raises(ValueError):
        storage.resolve("../../etc/passwd")
    with pytest.raises(ValueError):
        storage.project_dir("../../etc")
    assert storage.safe_filename("../../evil.png") == "evil.png"
    assert storage.safe_filename("..\\..\\win.png") == "win.png"


def test_image_endpoint_only_serves_registered_files(client, project):
    assert client.get("/api/images/..%2F..%2Fkosei.db/file").status_code in (404, 422)
    assert client.get("/api/images/noexiste/file").status_code == 404


def test_lora_import_rejects_non_safetensors(client, project):
    res = client.post(f"/api/projects/{project['id']}/loras/import", files=[("file", ("x.ckpt", b"pickle", "application/octet-stream"))])
    assert res.status_code == 422
    res = client.post(f"/api/projects/{project['id']}/loras/import", files=[("file", ("x.safetensors", b"\x00\x00", "application/octet-stream"))])
    assert res.status_code == 422


# ---------- Particion y fuga ----------

def test_split_is_reproducible_and_lock_is_enforced(client, project):
    pid = project["id"]
    add_class(client, pid)
    import_images(client, pid, 20)

    def assignment():
        return {r["sha256"]: r["split"] for r in db.query("SELECT sha256, split FROM images WHERE project_id=?", [pid])}

    client.post(f"/api/projects/{pid}/split", json={"train": 8, "val": 4, "test": 8, "seed": 7})
    first = assignment()
    client.post(f"/api/projects/{pid}/split", json={"train": 8, "val": 4, "test": 8, "seed": 7})
    assert assignment() == first
    assert sorted(first.values()).count("train") == 8

    locked = client.post(f"/api/projects/{pid}/split/lock").json()
    assert len(locked["split_hash"]) == 64
    assert client.post(f"/api/projects/{pid}/split", json={"train": 1, "val": 1, "test": 18, "seed": 7}).status_code == 409
    res = client.post(f"/api/projects/{pid}/import", files=[("files", ("x.png", png_bytes(99), "image/png"))])
    assert res.status_code == 409


def test_near_duplicates_stay_in_same_split(client, project):
    pid = project["id"]
    base = Image.open(io.BytesIO(png_bytes(40)))
    variant = io.BytesIO()
    base.convert("RGB").save(variant, "JPEG", quality=95)
    files = [("files", ("a.png", png_bytes(40), "image/png")), ("files", ("a.jpg", variant.getvalue(), "image/jpeg"))]
    files += [("files", (f"o{i}.png", png_bytes(50 + i), "image/png")) for i in range(10)]
    wait_job(client, client.post(f"/api/projects/{pid}/import", files=files).json()["job_id"])
    report = client.post(f"/api/projects/{pid}/split", json={"train": 6, "val": 2, "test": 4, "seed": 3}).json()
    assert report["cross_split_near_duplicates"] == []


def test_leakage_report_detects_lora_with_test_images(client, monkeypatch, project):
    pid = project["id"]
    ready_project(client, monkeypatch, pid)
    test_hash = db.one("SELECT sha256 FROM images WHERE project_id=? AND split='test'", [pid])["sha256"]
    db.insert("loras", {
        "id": db.new_id(), "project_id": pid, "name": "contaminado", "arch": "sdxl", "origin": "trained",
        "status": "ready", "train_hashes": [test_hash], "created_at": db.now(),
    })
    report = client.get(f"/api/projects/{pid}/split").json()
    assert report["lora_leaks"] == [{"lora": "contaminado", "images": 1}]


# ---------- Trabajos ----------

def test_job_lifecycle_cancel_retry_and_recovery(client):
    jobs.register("prueba_lenta", "cpu")(lambda ctx: _slow(ctx))
    job_id = jobs.enqueue("prueba_lenta", None, {})
    # Cola: el trabajo cancelado antes de empezar no se ejecuta.
    queued = jobs.enqueue("prueba_lenta", None, {})
    assert client.post(f"/api/jobs/{queued}/cancel").json()["status"] == "cancelled"

    running = _wait_status(client, job_id, "running")
    assert running["status"] == "running"
    assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409
    client.post(f"/api/jobs/{job_id}/cancel")
    final = wait_job(client, job_id)
    assert final["status"] == "cancelled"
    retried = client.post(f"/api/jobs/{job_id}/retry").json()["job_id"]
    client.post(f"/api/jobs/{retried}/cancel")
    wait_job(client, retried)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def _slow(ctx):
    import time
    for _ in range(200):
        ctx.check_cancel()
        time.sleep(0.02)


def _wait_status(client, job_id, status):
    import time
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(job)


def test_recover_marks_running_as_interrupted(client):
    job_id = db.new_id()
    db.insert("jobs", {"id": job_id, "kind": "generate", "queue": "none", "status": "running", "params": {}, "created_at": db.now()})
    assert jobs.recover() == 1
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "interrupted"


def test_failed_job_keeps_error(client):
    jobs.register("prueba_falla", "cpu")(lambda ctx: 1 / 0)
    job = wait_job(client, jobs.enqueue("prueba_falla", None, {}))
    assert job["status"] == "failed" and "division" in job["error"]


# ---------- Generacion ----------

def _gen_body(class_id, **over):
    return {
        "checkpoint": "m", "class_id": class_id, "prompt": "foto", "count": 6, "seed": 100, "steps": 2, "cfg_scale": 5,
        "width": 256, "height": 256, "sampler": "Euler", "scheduler": "simple", "batch_size": 1,
        "scenarios": {"fondo": ["a", "b"], "luz": ["c"]}, **over,
    }


@pytest.fixture()
def fake_forge(monkeypatch):
    fake = FakeForge()
    monkeypatch.setattr(forge, "models", lambda: [{"title": "m", "name": "m", "arch": "sdxl", "arch_label": "SDXL", "defaults": {}}])
    monkeypatch.setattr(forge, "ensure_checkpoint", lambda *a: None)
    monkeypatch.setattr(forge, "txt2img", fake.txt2img)
    monkeypatch.setattr(forge, "interrupt", lambda: None)
    return fake


def test_generation_saves_metadata_and_labels(client, monkeypatch, project, fake_forge):
    fake_label(monkeypatch)
    pid = project["id"]
    cls = add_class(client, pid)
    res = client.post(f"/api/projects/{pid}/generate", json=_gen_body(cls["id"]))
    assert res.status_code == 202, res.text
    job = wait_job(client, res.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["generated"] == 6
    rows = db.query("SELECT * FROM images WHERE project_id=? AND origin='synthetic' ORDER BY created_at", [pid])
    assert len(rows) == 6
    meta = rows[0]["generation"]
    assert meta["prompt"] and meta["seed"] >= 100 and meta["checkpoint"] == "m" and meta["sampler"] == "Euler"
    assert rows[0]["class_source"] == "condicionamiento"
    with Image.open(storage.resolve(rows[0]["path"])) as png:
        assert json.loads(png.text["kosei_provenance"])["synthetic"] is True
    # Sin repetir combinaciones hasta agotarlas: 2 fondos x 1 luz.
    assert sorted(c["prompt"] for c in fake_forge.calls[:2]) == ["foto, a, c", "foto, b, c"]
    label_job = jobs.active_job(pid, "label")
    if label_job:
        wait_job(client, label_job["id"])
    assert {r["status"] for r in db.query("SELECT status FROM images WHERE project_id=?", [pid])} == {"candidate"}


def test_incompatible_lora_is_refused(client, project, fake_forge):
    pid = project["id"]
    cls = add_class(client, pid)
    lora_id = db.new_id()
    db.insert("loras", {"id": lora_id, "name": "viejo", "arch": "sd15", "origin": "external", "status": "ready", "created_at": db.now()})
    res = client.post(f"/api/projects/{pid}/generate", json=_gen_body(cls["id"], lora_id=lora_id))
    assert res.status_code == 422
    assert "Incompatible" in res.json()["detail"]


def test_generation_resumes_without_repeating(client, monkeypatch, project, fake_forge):
    fake_label(monkeypatch)
    pid = project["id"]
    cls = add_class(client, pid)
    fake_forge.fail_at = 3
    job_id = client.post(f"/api/projects/{pid}/generate", json=_gen_body(cls["id"])).json()["job_id"]
    failed = wait_job(client, job_id)
    assert failed["status"] == "failed"
    assert db.scalar("SELECT COUNT(*) FROM images WHERE project_id=?", [pid]) == 3

    fake_forge.fail_at = None
    resumed = wait_job(client, client.post(f"/api/jobs/{job_id}/retry").json()["job_id"])
    assert resumed["status"] == "succeeded"
    assert resumed["result"]["resumed_from"] == 3
    assert db.scalar("SELECT COUNT(*) FROM images WHERE project_id=?", [pid]) == 6
    assert sorted(c["seed"] for c in fake_forge.calls) == [100, 101, 102, 103, 104, 105]


def test_scenario_plan_covers_all_before_repeating():
    plan = generation.scenario_plan({"a": ["1", "2"], "b": ["x", "y", "z"]}, 12, 5)
    assert len({tuple(p) for p in plan[:6]}) == 6
    assert generation.scenario_plan({"a": ["1", "2"]}, 4, 5) == generation.scenario_plan({"a": ["1", "2"]}, 4, 5)


def test_batches_only_group_same_scenario():
    plan = [["a"], ["a"], ["b"], ["a"]]
    assert generation._batches([0, 1, 2, 3], plan, 4) == [[0, 1], [2], [3]]
    assert generation._batches([0, 1, 2, 3], [["a"]] * 4, 3) == [[0, 1, 2], [3]]
    assert generation._batches([0, 3], [["a"]] * 4, 3) == [[0], [3]]


# ---------- Etiquetado y revision ----------

@pytest.mark.parametrize("boxes,polygons,flags,expected", [
    ([], [], [], "no_detection"),
    ([{"cx": .5, "cy": .5, "w": .4, "h": .4, "confidence": .9}], [[0, 0, 1, 0, 1, 1]], [], "candidate"),
    ([{"cx": .5, "cy": .5, "w": .4, "h": .4, "confidence": .5}], [[0, 0, 1, 0, 1, 1]], [], "needs_review"),
    ([{"cx": .5, "cy": .5, "w": .4, "h": .4, "confidence": .9}], [None], [], "needs_review"),
    ([{"cx": .5, "cy": .5, "w": .4, "h": .4, "confidence": .9}] * 2, [[0, 0, 1, 0, 1, 1]] * 2, [], "needs_review"),
    ([{"cx": .5, "cy": .5, "w": .01, "h": .01, "confidence": .9}], [[0, 0, 1, 0, 1, 1]], [], "needs_review"),
    ([{"cx": .5, "cy": .5, "w": .4, "h": .4, "confidence": .9}], [[0, 0, 1, 0, 1, 1]], ["borrosa"], "needs_review"),
])
def test_status_rules(boxes, polygons, flags, expected):
    assert labeling.decide_status(boxes, polygons, flags)[0] == expected


def test_no_detection_is_not_exportable_without_confirmation(client, monkeypatch, project):
    pid = project["id"]
    fake_label(monkeypatch, boxes=[])
    add_class(client, pid)
    import_images(client, pid, 1)
    wait_job(client, client.post(f"/api/projects/{pid}/label").json()["job_id"])
    row = db.one("SELECT id, status FROM images WHERE project_id=?", [pid])
    assert row["status"] == "no_detection"
    assert client.post(f"/api/images/{row['id']}/review", json={"status": "approved"}).status_code == 409
    assert client.post(f"/api/images/{row['id']}/review", json={"status": "approved", "negative": True}).status_code == 200


def test_human_correction_is_verified(client, monkeypatch, project):
    pid = project["id"]
    fake_label(monkeypatch)
    cls = add_class(client, pid)
    import_images(client, pid, 1)
    wait_job(client, client.post(f"/api/projects/{pid}/label").json()["job_id"])
    row = db.one("SELECT id FROM images WHERE project_id=?", [pid])
    res = client.put(f"/api/images/{row['id']}/annotations", json={"boxes": [{"class_id": cls["id"], "cx": .4, "cy": .4, "w": .2, "h": .2}]})
    saved = res.json()
    assert saved[0]["source"] == "humano" and saved[0]["verified"] == 1 and saved[0]["polygon"]
    assert client.put(f"/api/images/{row['id']}/annotations", json={"boxes": [{"class_id": 99999, "cx": .4, "cy": .4, "w": .2, "h": .2}]}).status_code == 422


def test_bulk_approve_skips_images_without_boxes(client, monkeypatch, project):
    pid = project["id"]
    fake_label(monkeypatch, boxes=[])
    add_class(client, pid)
    import_images(client, pid, 2)
    wait_job(client, client.post(f"/api/projects/{pid}/label").json()["job_id"])
    res = client.post(f"/api/projects/{pid}/review/bulk", json={"from_status": "no_detection", "origin": "real", "status": "approved"}).json()
    assert res == {"changed": 0, "skipped": 2}


# ---------- Construccion y exportacion ----------

def _synthetic(client, monkeypatch, pid, n=3):
    cls_id = db.query("SELECT id FROM classes WHERE project_id=?", [pid])[0]["id"]
    fake_label(monkeypatch)
    for i in range(n):
        path = storage.project_dir(pid) / f"s{i}.png"
        path.write_bytes(png_bytes(500 + i))
        image_id, _, _ = ingest.register_image(pid, path, path.name, "synthetic", extra={"status": "approved", "class_id": cls_id, "generation": {"seed": i}})
        db.insert("annotations", {"image_id": image_id, "class_id": cls_id, "cx": .5, "cy": .5, "w": .3, "h": .3, "polygon": [.4, .4, .6, .4, .6, .6], "source": "doble", "verified": 1, "created_at": db.now()})


def test_export_is_valid_and_synthetics_only_in_train(client, monkeypatch, project):
    pid = project["id"]
    ready_project(client, monkeypatch, pid)
    _synthetic(client, monkeypatch, pid)
    res = client.post(f"/api/projects/{pid}/builds", json={"name": "d", "seed": 1, "tasks": ["detect", "segment", "classify"]}).json()
    job = wait_job(client, res["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["validation"]["ok"], job["result"]["validation"]

    zip_res = client.get(f"/api/builds/{res['build_id']}/download")
    assert zip_res.status_code == 200
    with zipfile.ZipFile(io.BytesIO(zip_res.content)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json"))
        data = yaml.safe_load(zf.read("detect/data.yaml"))
        assert data["names"] == {0: "persona"} and data["train"] == "images/train"
        for item in manifest["items"]:
            if item["origin"] == "synthetic":
                assert item["split"] == "train"
            assert item["sha256_original"]
        images = [n for n in names if n.startswith("detect/images/") and n.endswith(".png")]
        for name in images:
            assert name.replace("/images/", "/labels/").replace(".png", ".txt") in names
        det = [n for n in names if n.startswith("detect/labels/")][0]
        seg = [n for n in names if n.startswith("segment/labels/")][0]
        assert len(zf.read(det).split()) == 5
        assert len(zf.read(seg).split()) > 5
        assert any(n.startswith("classify/train/persona/") for n in names)


def test_export_excludes_unapproved(client, monkeypatch, project):
    pid = project["id"]
    ready_project(client, monkeypatch, pid)
    _synthetic(client, monkeypatch, pid, 2)
    db.execute("UPDATE images SET status='needs_review' WHERE project_id=? AND origin='synthetic'", [pid])
    res = client.post(f"/api/projects/{pid}/builds", json={"name": "d", "tasks": ["detect"]}).json()
    job = wait_job(client, res["job_id"])
    assert job["result"]["counts"]["detect"]["train"]["synthetic"] == 0


def test_validate_zip_detects_broken_exports(tmp_path):
    path = tmp_path / "x.zip"
    path.write_bytes(_zip({
        "manifest.json": b"{}",
        "detect/data.yaml": yaml.safe_dump({"names": {0: "a"}}).encode(),
        "detect/images/train/a.png": png_bytes(1),
        "detect/images/train/b.png": png_bytes(2),
        "detect/labels/train/a.txt": b"5 0.5 0.5 0.1 0.1\n",
    }))
    result = export.validate_zip(path, ["detect"])
    assert not result["ok"]
    joined = " ".join(result["problems"])
    assert "Sin etiqueta" in joined and "Etiqueta invalida" in joined


# ---------- Experimento ----------

def test_verdict_rule():
    from app.services.experiment import _verdict
    assert _verdict([0.10, 0.11, 0.12])["verdict"] == "mejora"
    assert _verdict([-0.10, -0.11, -0.12])["verdict"] == "empeora"
    assert _verdict([0.10, -0.10, 0.01])["verdict"] == "sin_cambio_concluyente"
    assert _verdict([0.1])["verdict"] == "no_concluyente"
    assert _verdict([])["verdict"] == "sin_datos"


def test_preflight_blocks_invalid_experiments(client, monkeypatch, project):
    from app.services import experiment
    pid = project["id"]
    add_class(client, pid)
    import_images(client, pid, 6)
    check = experiment.preflight(pid, [{"name": "A", "n_synthetic": 0}, {"name": "B", "n_synthetic": 5}])
    assert not check["ok"]
    assert any("bloqueada" in p for p in check["problems"])
    assert any("sinteticas" in p for p in check["problems"])


def test_preflight_flags_contaminated_and_external_lora(client, monkeypatch, project):
    from app.services import experiment
    pid = project["id"]
    ready_project(client, monkeypatch, pid)
    _synthetic(client, monkeypatch, pid, 1)
    test_hash = db.one("SELECT sha256 FROM images WHERE project_id=? AND split='test'", [pid])["sha256"]
    lora_id = db.new_id()
    db.insert("loras", {"id": lora_id, "project_id": pid, "name": "x", "arch": "sdxl", "origin": "trained", "status": "ready", "train_hashes": [test_hash], "created_at": db.now()})
    db.execute("UPDATE images SET lora_id=? WHERE project_id=? AND origin='synthetic'", [lora_id, pid])
    check = experiment.preflight(pid, [{"name": "A", "n_synthetic": 0}, {"name": "B", "n_synthetic": 1}])
    assert not check["ok"] and any("val o test" in p for p in check["problems"])

    db.update("loras", lora_id, {"origin": "external", "train_hashes": None})
    check = experiment.preflight(pid, [{"name": "A", "n_synthetic": 0}, {"name": "B", "n_synthetic": 1}])
    assert check["ok"] and any("no verificable" in w for w in check["warnings"])


def test_summary_uses_real_numbers_only():
    from app.services.experiment import summarize
    def run(cond, seed, value):
        return {"condition": cond, "seed": seed, "status": "succeeded", "counts": {"train": {"real": 1, "synthetic": 0}},
                "metrics": {"map50_95": value, "map50": value, "precision": value, "recall": value, "f1": value}}
    runs = [run("A", 0, .5), run("A", 1, .52), run("B", 0, .6), run("B", 1, .63), {"condition": "B", "seed": 2, "status": "failed", "metrics": None, "counts": None}]
    summary = summarize(runs, "A")
    assert summary["conditions"]["B"]["runs"] == 2
    assert summary["conditions"]["A"]["map50_95"]["mean"] == pytest.approx(0.51)
    assert summary["comparisons"]["B"]["pairs"] == 2


# ---------- Entrenamiento (sin GPU: lo que rodea a kohya) ----------

def test_training_progress_parser_ignores_other_bars():
    from app.services.training import parse_progress
    assert parse_progress("steps:  12%|#2        | 120/1000 [01:00<07:00,  2.09it/s, avr_loss=0.1234]") == (120, 1000, 0.1234)
    assert parse_progress("100%|##########| 20/20 [00:04<00:00,  4.8it/s]") is None
    assert parse_progress("caching latents: 5/10") is None


def test_train_requires_locked_split_and_valid_name(client, project):
    pid = project["id"]
    body = {"name": "ok", "checkpoint": "x", "token": "tok", "steps": 100}
    assert client.post(f"/api/projects/{pid}/loras", json=body).status_code == 409
    assert client.post(f"/api/projects/{pid}/loras", json={**body, "name": "../../evil"}).status_code == 422


def test_kohya_command_per_architecture(tmp_path):
    from app.services import training
    cfg = {"name": "n", "steps": 100, "save_every": 50, "batch_size": 1, "network_dim": 8, "network_alpha": 4,
           "learning_rate": 1e-4, "seed": 1, "resolution": 1024}
    sdxl = training.build_command("sdxl", tmp_path / "m.safetensors", tmp_path, tmp_path, tmp_path, cfg)
    assert any("sdxl_train_network.py" in part for part in sdxl)
    assert "--network_train_unet_only" in sdxl and "--sdpa" in sdxl and "--xformers" not in sdxl
    sd15 = training.build_command("sd15", tmp_path / "m.safetensors", tmp_path, tmp_path, tmp_path, cfg)
    assert any(part.endswith("/train_network.py") for part in sd15) and "--network_train_unet_only" not in sd15
    flux = training.build_command("flux", tmp_path / "m.safetensors", tmp_path, tmp_path, tmp_path, cfg)
    assert "networks.lora_flux" in flux and "--blocks_to_swap" in flux


def test_arch_detection_from_tensors(tmp_path):
    from app.services import arch
    from safetensors.numpy import save_file
    import numpy as np
    def make(name, keys):
        path = tmp_path / name
        save_file({k: np.zeros(1) for k in keys}, path)
        return path
    assert arch.lora_info(make("a.safetensors", ["lora_te_x.alpha", "lora_unet_down_blocks_0.alpha"]))["arch"] == "sd15"
    assert arch.lora_info(make("b.safetensors", ["lora_unet_input_blocks_1.alpha", "lora_te1_x.alpha"]))["arch"] == "sdxl"
    assert arch.lora_info(make("c.safetensors", ["lora_unet_double_blocks_0.alpha"]))["arch"] == "flux"
    assert arch.checkpoint_arch(make("d.safetensors", ["model.diffusion_model.double_blocks.0.x"])) == "flux"
    assert arch.checkpoint_arch(make("e.safetensors", ["conditioner.embedders.1.model.x"])) == "sdxl"
    assert arch.checkpoint_arch(make("f.safetensors", ["cond_stage_model.transformer.x"])) == "sd15"


def test_mask_polygon_uses_largest_contour_without_bridges():
    import numpy as np
    mask = np.zeros((100, 100))
    mask[10:60, 10:60] = 1
    mask[80:85, 80:85] = 1
    polygon = labeling.mask_to_polygon(mask)
    xs, ys = polygon[0::2], polygon[1::2]
    assert max(xs) < 0.7 and max(ys) < 0.7, "No debe unir la region pequena con una linea"
    assert labeling.mask_to_polygon(np.zeros((10, 10))) is None
