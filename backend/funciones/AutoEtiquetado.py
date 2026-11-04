from pathlib import Path
from ultralytics import YOLO


def _stem(image_path: str) -> str:
    return str(Path(image_path).with_suffix(""))


def clasificar_imagen(image_path: str, model, target_class: str = "object") -> str | None:
    results = model.predict(image_path, save=False, conf=0.5, show=False, save_txt=False, save_conf=False)
    if results:
        top1_idx = results[0].probs.top1
        label = target_class if target_class else results[0].names[top1_idx]
        Path(f"{_stem(image_path)}_clasificacion.txt").write_text(label)
        return label
    return None


def autodeteccion(image_path: str, model, class_id: int = 0) -> bool:
    results = model.predict(image_path, save=False, conf=0.5, show=False, save_txt=False, save_conf=False)
    if not results or not results[0].boxes:
        return False

    boxes = results[0].boxes
    if len(boxes) == 0:
        return False

    img_h, img_w = results[0].orig_shape
    lines = []
    for box in boxes.xywh.cpu().numpy():
        cx, cy, w, h = box
        lines.append(
            f"{class_id} {cx/img_w:.6f} {cy/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}"
        )

    Path(f"{_stem(image_path)}_detecciones.txt").write_text("\n".join(lines))
    return True


def segmentar_imagen(image_path: str, model, class_id: int = 0) -> bool:
    results = model.predict(image_path, save=False, conf=0.5, show=False, save_txt=False, save_conf=False)
    if not results or results[0].masks is None:
        return False

    lines = []
    for mask_xyn in results[0].masks.xyn:
        pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in mask_xyn)
        if pts:
            lines.append(f"{class_id} {pts}")

    if lines:
        Path(f"{_stem(image_path)}_segmentaciones.txt").write_text("\n".join(lines))
    return bool(lines)


def procesar_imagen(
    image_path: str,
    model_cls,
    model_det,
    model_seg,
    target_class: str = "object",
    class_id: int = 0,
) -> None:
    label = clasificar_imagen(image_path, model_cls, target_class)
    if label:
        if autodeteccion(image_path, model_det, class_id):
            segmentar_imagen(image_path, model_seg, class_id)


def load_models(cls_path: str, det_path: str, seg_path: str):
    return YOLO(cls_path), YOLO(det_path), YOLO(seg_path)
