"""Una corrida del experimento: entrena YOLO y evalua el mejor modelo en la prueba real."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(config_path: str) -> None:
    from ultralytics import YOLO

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_root = Path(cfg["project"])
    model = YOLO(cfg["model"])

    def on_epoch_end(trainer) -> None:
        print(f"KOSEI_EPOCH {trainer.epoch + 1}", flush=True)

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(
        data=cfg["data"], epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
        seed=cfg["seed"], deterministic=True, pretrained=False,
        patience=cfg["patience"] or 0, project=str(run_root), name="train", exist_ok=True,
        workers=2, plots=False, verbose=False, device=0, cache=False,
    )
    best = run_root / "train" / "weights" / "best.pt"
    # La prueba real solo se toca aqui, una vez, con el mejor modelo segun validacion.
    metrics = YOLO(str(best)).val(
        data=cfg["data"], split="test", imgsz=cfg["imgsz"], batch=cfg["batch"],
        project=str(run_root), name="test", exist_ok=True, plots=False, verbose=False, device=0, workers=2,
    )
    precision, recall = float(metrics.box.mp), float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    (run_root / "metrics.json").write_text(json.dumps({
        "map50_95": float(metrics.box.map), "map50": float(metrics.box.map50),
        "precision": precision, "recall": recall, "f1": f1,
    }), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
