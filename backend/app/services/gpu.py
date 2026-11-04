"""Estado real de la GPU y liberacion de VRAM antes de un entrenamiento."""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Optional

from . import forge

log = logging.getLogger("kosei.gpu")


def status() -> Optional[dict[str, Any]]:
    """Consulta nvidia-smi. None si no hay GPU visible."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, total, used, free, util, temp = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    return {
        "name": name, "memory_total_mib": int(total), "memory_used_mib": int(used),
        "memory_free_mib": int(free), "utilization_pct": int(util), "temperature_c": int(temp),
    }


class NotEnoughVram(RuntimeError):
    pass


def free_for_training(required_mib: int, log_fn=None) -> dict[str, Any]:
    """Descarga el modelo de Forge y comprueba que de verdad quedo VRAM libre."""
    say = log_fn or log.info
    try:
        forge.unload()
        say("Modelo descargado del motor")
    except forge.ForgeError as exc:
        say(f"No se pudo descargar el modelo del motor: {exc}")
    deadline = time.time() + 30
    state = status()
    while state and state["memory_free_mib"] < required_mib and time.time() < deadline:
        time.sleep(2)
        state = status()
    if state is None:
        raise NotEnoughVram("No se detecta una GPU NVIDIA en el contenedor")
    if state["memory_free_mib"] < required_mib:
        raise NotEnoughVram(
            f"Se necesitan {required_mib} MiB libres de VRAM y hay {state['memory_free_mib']} MiB. "
            "Cierra otras aplicaciones que usen la GPU y reintenta."
        )
    return state
