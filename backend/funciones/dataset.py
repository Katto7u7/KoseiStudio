from __future__ import annotations

import io
import zipfile
from pathlib import Path


def read_labels(output_dir: Path, image_id: str) -> dict:
    """Parse the three label files for a given image id."""
    stem = output_dir / image_id

    classification = None
    cls_file = Path(f"{stem}_clasificacion.txt")
    if cls_file.exists():
        classification = cls_file.read_text().strip() or None

    has_detection = Path(f"{stem}_detecciones.txt").exists()
    has_segmentation = Path(f"{stem}_segmentaciones.txt").exists()

    return {
        "class_label": classification,
        "has_detection": has_detection,
        "has_segmentation": has_segmentation,
    }


def build_yolo_zip(entries: list[dict], output_dir: Path) -> bytes:
    """Build an in-memory ZIP with images/ and labels/ in YOLO format."""
    buf = io.BytesIO()
    classes: set[str] = set()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            image_id = entry["id"]
            png_path = output_dir / f"{image_id}.png"
            if not png_path.exists():
                continue

            zf.write(png_path, f"images/{image_id}.png")

            # Detection label
            det_file = output_dir / f"{image_id}_detecciones.txt"
            if det_file.exists():
                zf.write(det_file, f"labels/{image_id}.txt")

            # Segmentation label
            seg_file = output_dir / f"{image_id}_segmentaciones.txt"
            if seg_file.exists():
                zf.write(seg_file, f"labels/{image_id}_seg.txt")

            if entry.get("class_label"):
                classes.add(entry["class_label"])

        # classes.txt
        zf.writestr("classes.txt", "\n".join(sorted(classes)))

    return buf.getvalue()
