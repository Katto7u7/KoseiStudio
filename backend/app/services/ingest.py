"""Importacion segura de datasets y registro de imagenes.

El original se conserva tal cual y nunca se sobrescribe. Se guarda aparte un derivado
sRGB, que es el que usan las etapas posteriores. Las etiquetas YOLO se importan si vienen.
"""
from __future__ import annotations

import logging
import re
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any, Optional

import yaml

from .. import config, db
from ..jobs import JobContext, register
from . import imaging, storage

log = logging.getLogger("kosei.ingest")


class UnsafeArchive(ValueError):
    pass


def _safe_member(info: zipfile.ZipInfo) -> Optional[str]:
    """Devuelve el nombre normalizado, o None si es carpeta. Rechaza lo peligroso."""
    name = info.filename.replace("\\", "/")
    if info.is_dir():
        return None
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise UnsafeArchive(f"El ZIP contiene un enlace simbolico: {name}")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in parts:
        raise UnsafeArchive(f"Ruta insegura en el ZIP: {name}")
    if parts and parts[0] == "__MACOSX":
        return None
    return "/".join(parts)


def extract_zip(archive: Path, dest: Path) -> list[Path]:
    """Extrae validando zip-slip, enlaces y bombas de descompresion antes de escribir."""
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > config.MAX_ZIP_ENTRIES:
            raise UnsafeArchive(f"El ZIP tiene demasiados archivos ({len(infos)})")
        if sum(i.file_size for i in infos) > config.MAX_ZIP_UNCOMPRESSED:
            raise UnsafeArchive("El contenido descomprimido del ZIP supera el limite")
        root = dest.resolve()
        for info in infos:
            name = _safe_member(info)
            if name is None:
                continue
            target = (root / name).resolve()
            if root not in target.parents:
                raise UnsafeArchive(f"Ruta insegura en el ZIP: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(info) as src, target.open("wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    written += len(chunk)
                    if written > config.MAX_ZIP_UNCOMPRESSED:
                        raise UnsafeArchive("El contenido real del ZIP supera el limite declarado")
                    out.write(chunk)
            extracted.append(target)
    return extracted


def register_image(
    project_id: str,
    source: Path,
    filename: str,
    origin: str,
    extra: Optional[dict[str, Any]] = None,
    move: bool = False,
    sha: Optional[str] = None,
) -> tuple[str, bool, dict[str, Any]]:
    """Valida y registra. Devuelve (id, es_nueva, info). Un duplicado exacto no se vuelve a crear."""
    sha = sha or storage.sha256_file(source)
    existing = db.one("SELECT id FROM images WHERE project_id=? AND sha256=?", [project_id, sha])
    if existing:
        return existing["id"], False, {}

    image = imaging.open_verified(source)
    normalized = imaging.to_srgb(image)
    image_id = db.new_id()
    pdir = storage.project_dir(project_id)
    safe = storage.safe_filename(filename, "imagen")
    original = pdir / "images" / origin / f"{image_id}_{safe}"
    original.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(source), original)
    else:
        shutil.copy2(source, original)
    original.chmod(0o444)

    norm_path = pdir / "normalized" / f"{image_id}.png"
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(norm_path, "PNG")
    thumb = pdir / "thumbs" / f"{image_id}.webp"
    imaging.save_thumb(normalized, thumb)

    qc = imaging.measures(normalized)
    qc["flags"] = imaging.artifact_flags(qc)
    db.insert("images", {
        "id": image_id, "project_id": project_id, "origin": origin, "filename": safe,
        "path": storage.rel(original), "normalized_path": storage.rel(norm_path),
        "thumb_path": storage.rel(thumb), "sha256": sha, "phash": imaging.phash(normalized),
        "width": normalized.width, "height": normalized.height, "format": image.format or "",
        "size_bytes": original.stat().st_size, "qc": qc, "created_at": db.now(),
        **(extra or {}),
    })
    return image_id, True, {"qc": qc}


def _class_names_from_yaml(root: Path) -> list[str]:
    for candidate in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        try:
            data = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            continue
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            return [str(names[k]) for k in sorted(names)]
        if isinstance(names, list):
            return [str(n) for n in names]
    classes = next(iter(sorted(root.rglob("classes.txt"))), None)
    if classes:
        return [l.strip() for l in classes.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def _label_for(image: Path, root: Path) -> Optional[Path]:
    """Busca el .txt YOLO: al lado de la imagen, o en labels/ con la misma estructura."""
    sibling = image.with_suffix(".txt")
    if sibling.exists():
        return sibling
    parts = list(image.relative_to(root).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        candidate = (root / Path(*parts)).with_suffix(".txt")
        if candidate.exists():
            return candidate
    return None


def parse_yolo_line(line: str) -> Optional[tuple[int, list[float]]]:
    """Valida una linea YOLO. Devuelve (clase, numeros) o None si es invalida."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls, numbers = int(parts[0]), [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if cls < 0 or any(n < 0 or n > 1 for n in numbers):
        return None
    if len(numbers) != 4 and (len(numbers) < 6 or len(numbers) % 2):
        return None
    return cls, numbers


def ensure_class(project_id: str, name: str, idx: Optional[int] = None) -> dict[str, Any]:
    found = db.one("SELECT * FROM classes WHERE project_id=? AND name=?", [project_id, name])
    if found:
        return found
    if idx is None or db.one("SELECT 1 FROM classes WHERE project_id=? AND idx=?", [project_id, idx]):
        idx = int(db.scalar("SELECT COALESCE(MAX(idx)+1, 0) FROM classes WHERE project_id=?", [project_id]))
    db.insert("classes", {"project_id": project_id, "name": name, "idx": idx})
    return db.one("SELECT * FROM classes WHERE project_id=? AND name=?", [project_id, name])


def _import_labels(project_id: str, image_id: str, label: Path, names: list[str]) -> int:
    """Las etiquetas importadas no se marcan verificadas: las aprueba una persona."""
    count = 0
    for raw in label.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_yolo_line(raw)
        if parsed is None:
            continue
        cls_idx, numbers = parsed
        name = names[cls_idx] if cls_idx < len(names) else f"clase_{cls_idx}"
        cls = ensure_class(project_id, name, cls_idx)
        if len(numbers) == 4:
            cx, cy, w, h = numbers
            polygon = None
        else:
            xs, ys = numbers[0::2], numbers[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            polygon = numbers
        db.insert("annotations", {
            "image_id": image_id, "class_id": cls["id"], "cx": cx, "cy": cy, "w": w, "h": h,
            "polygon": polygon, "source": "importada", "mask_source": "importada" if polygon else None,
            "verified": 0, "created_at": db.now(),
        })
        count += 1
    return count


def _caption_for(image: Path) -> Optional[str]:
    """Las captions de kohya viven como .txt junto a la imagen, sin formato YOLO."""
    sibling = image.with_suffix(".txt")
    if not sibling.exists():
        return None
    text = sibling.read_text(encoding="utf-8", errors="replace").strip()
    return text if text and parse_yolo_line(text.splitlines()[0]) is None else None


@register("ingest", "cpu")
def run_ingest(ctx: JobContext) -> dict[str, Any]:
    project_id = ctx.project_id
    upload_dir = Path(ctx.params["upload_dir"])
    work = upload_dir.resolve()
    if config.TMP_DIR.resolve() not in work.parents:
        raise ValueError("Directorio de carga invalido")

    report: dict[str, Any] = {
        "imported": 0, "duplicates": [], "corrupt": [], "skipped": [], "labels_imported": 0,
        "near_duplicates": [],
    }
    try:
        for archive in sorted(work.glob("*.zip")):
            ctx.progress(0, 0, f"Revisando {archive.name}")
            try:
                extract_zip(archive, work / f"{archive.stem}_x")
            except (UnsafeArchive, zipfile.BadZipFile) as exc:
                report["corrupt"].append({"file": archive.name, "reason": str(exc)})
            archive.unlink(missing_ok=True)

        names = _class_names_from_yaml(work)
        files = sorted(p for p in work.rglob("*") if p.is_file())
        images = [p for p in files if p.suffix.lower() in imaging.IMAGE_EXTENSIONS]
        report["skipped"] = [p.name for p in files if p.suffix.lower() not in imaging.IMAGE_EXTENSIONS | {".txt", ".yaml", ".yml"}][:100]
        known = db.query("SELECT id, phash, filename FROM images WHERE project_id=?", [project_id])

        for i, path in enumerate(images, 1):
            ctx.check_cancel()
            ctx.progress(i, len(images), path.name)
            try:
                image_id, new, _ = register_image(project_id, path, path.name, "real", extra={"status": "pending"})
            except imaging.InvalidImage as exc:
                report["corrupt"].append({"file": path.name, "reason": str(exc)})
                continue
            if not new:
                report["duplicates"].append(path.name)
                continue
            report["imported"] += 1
            row = db.one("SELECT phash FROM images WHERE id=?", [image_id])
            near = next((k for k in known if imaging.hamming(k["phash"], row["phash"]) <= 4), None)
            if near:
                report["near_duplicates"].append({"file": path.name, "similar_to": near["filename"]})
            known.append({"id": image_id, "phash": row["phash"], "filename": path.name})
            label = _label_for(path, work)
            caption = _caption_for(path)
            if caption:
                db.update("images", image_id, {"caption": caption})
            elif label:
                report["labels_imported"] += _import_labels(project_id, image_id, label, names)
                db.update("images", image_id, {"status": "needs_review"})
    finally:
        shutil.rmtree(work, ignore_errors=True)
    db.update("projects", project_id, {"updated_at": db.now()})
    return report
