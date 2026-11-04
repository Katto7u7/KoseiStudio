from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


class TrainingJob:
    def __init__(self, job_id: str, total_steps: int):
        self.job_id = job_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "running"
        self.log_lines: list[str] = []
        self.start_time = time.time()
        self.proc: Optional[subprocess.Popen] = None
        self.output_name: Optional[str] = None

    @property
    def elapsed(self) -> int:
        return int(time.time() - self.start_time)

    def add_log(self, line: str):
        self.log_lines.append(line.rstrip())
        if len(self.log_lines) > 200:
            self.log_lines.pop(0)
        step = parse_training_progress(line)
        if step is not None:
            self.current_step = step

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "log_tail": self.log_lines[-15:],
            "elapsed": self.elapsed,
            "output_name": self.output_name,
        }


# Global job registry
_jobs: dict[str, TrainingJob] = {}


def get_job(job_id: str) -> TrainingJob | None:
    return _jobs.get(job_id)


def prepare_training_dir(
    base_train_dir: Path,
    job_id: str,
    token_prefix: str,
    n_repeats: int,
    image_paths: list[Path],
    captions: dict[str, str],
) -> Path:
    """Create the kohya_ss-style image directory structure and copy files."""
    folder_name = f"{n_repeats}_{token_prefix}"
    image_dir = base_train_dir / job_id / "image" / folder_name
    image_dir.mkdir(parents=True, exist_ok=True)

    for src in image_paths:
        dest = image_dir / src.name
        shutil.copy2(src, dest)
        stem = src.stem
        caption = captions.get(src.name, f"{token_prefix} {stem}")
        (image_dir / f"{stem}.txt").write_text(caption)

    return base_train_dir / job_id


def run_blip_caption(
    image_dir: Path,
    token_prefix: str,
    kohya_dir: Path,
) -> dict[str, str]:
    """Run make_captions.py and return {filename: caption} dict."""
    script = kohya_dir / "finetune" / "make_captions.py"
    if not script.exists():
        raise RuntimeError(f"make_captions.py not found at {script}")

    cmd = [
        sys.executable,
        str(script),
        str(image_dir),
        "--batch_size", "1",
        "--num_beams", "1",
        "--top_p", "0.9",
        "--max_length", "75",
        "--min_length", "5",
        "--beam_search",
        "--caption_extension", ".txt",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] if result.stderr else "BLIP captioning failed")

    captions: dict[str, str] = {}
    for txt in image_dir.glob("*.txt"):
        raw = txt.read_text().strip()
        img_name = txt.with_suffix("").name + ".jpg"
        caption = f"{token_prefix} {raw}" if raw and not raw.startswith(token_prefix) else raw
        captions[img_name] = caption
        txt.write_text(caption)

    return captions


def build_accelerate_cmd(
    config: dict,
    kohya_dir: Path,
    train_data_dir: Path,
    output_dir: Path,
    log_dir: Path,
) -> list[str]:
    script = kohya_dir / "train_network.py"
    if not script.exists():
        raise RuntimeError(f"train_network.py not found at {script}")

    return [
        "accelerate", "launch",
        "--mixed_precision", "fp16",
        "--num_processes", "1",
        "--num_machines", "1",
        "--num_cpu_threads_per_process", "2",
        str(script),
        "--bucket_no_upscale",
        "--bucket_reso_steps", "64",
        "--cache_latents",
        "--enable_bucket",
        "--min_bucket_reso", "256",
        "--max_bucket_reso", "2048",
        "--learning_rate", str(config["learning_rate"]),
        "--logging_dir", str(log_dir),
        "--loss_type", "l2",
        "--lr_scheduler", "cosine",
        "--lr_warmup_steps", "0",
        "--max_data_loader_n_workers", "0",
        "--max_grad_norm", "1",
        "--resolution", "512,512",
        "--max_train_steps", str(config["max_steps"]),
        "--mixed_precision", "fp16",
        "--network_alpha", "1",
        "--network_dim", str(config["network_dim"]),
        "--network_module", "networks.lora",
        "--optimizer_type", "AdamW8bit",
        "--output_dir", str(output_dir),
        "--output_name", config["output_name"],
        "--pretrained_model_name_or_path", config["base_model"],
        "--save_every_n_epochs", "1",
        "--save_model_as", "safetensors",
        "--save_precision", "fp16",
        "--text_encoder_lr", str(config["learning_rate"]),
        "--train_batch_size", str(config["batch_size"]),
        "--train_data_dir", str(train_data_dir),
        "--unet_lr", str(config["learning_rate"]),
        "--xformers",
    ]


def parse_training_progress(line: str) -> int | None:
    """Extract current step from tqdm-style progress line."""
    m = re.search(r"(\d+)/\d+", line)
    if m:
        return int(m.group(1))
    return None


def start_training_job(
    job_id: str,
    config: dict,
    train_job_dir: Path,
    kohya_dir: Path,
    output_lora_dir: Path,
) -> TrainingJob:
    """Launch training subprocess and register the job."""
    train_data_dir = train_job_dir / "image"
    log_dir = train_job_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_lora_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_accelerate_cmd(config, kohya_dir, train_data_dir, output_lora_dir, log_dir)

    job = TrainingJob(job_id=job_id, total_steps=config["max_steps"])
    job.output_name = config["output_name"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    job.proc = proc
    _jobs[job_id] = job

    # Read output in a background thread
    import threading

    def reader():
        for line in proc.stdout:  # type: ignore[union-attr]
            job.add_log(line)
        proc.wait()
        if proc.returncode == 0:
            job.status = "done"
        elif job.status == "running":
            job.status = "error"

    threading.Thread(target=reader, daemon=True).start()
    return job


def cancel_job(job_id: str) -> None:
    job = _jobs.get(job_id)
    if job and job.proc and job.proc.poll() is None:
        job.proc.terminate()
        job.status = "cancelled"
