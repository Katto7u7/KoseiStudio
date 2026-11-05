"""Cliente minimo de la API por nginx, para pruebas de sistema reales."""
import json, mimetypes, sys, time, uuid
import urllib.error, urllib.request
from pathlib import Path

BASE = "http://localhost/api"


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as res:
            raw = res.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as err:
        raise SystemExit(f"{method} {path}: {err.code} {err.read()[:800].decode(errors='replace')}")


def upload(path, files):
    boundary = uuid.uuid4().hex
    parts = []
    for f in files:
        f = Path(f)
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{f.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode() + f.read_bytes() + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return json.loads(res.read())


def wait(job_id, every=2, show=True):
    last = None
    while True:
        job = call("GET", f"/jobs/{job_id}")
        line = f"{job['status']} {job['current']}/{job['total']} {job['message']}"
        if show and line != last:
            print(line[:150], flush=True)
            last = line
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(every)
