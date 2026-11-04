# KoseiStudio backend: API, colas de trabajo, etiquetado y entrenamiento.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/weights/hf \
    TORCH_HOME=/weights/torch \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl libgl1 libglib2.0-0 procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch con CUDA, compartido por el backend y el venv de kohya.
RUN pip install --upgrade "pip==25.2" \
    && pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# kohya sd-scripts fijado a un commit (entrena SD 1.5, SDXL y Flux).
# Sus pines chocan con el backend, por eso vive en su propio venv.
ENV KOHYA_COMMIT=cb70b96858477e716a2096292ab8aab433277f38
RUN git clone https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts \
    && git -C /opt/sd-scripts checkout "$KOHYA_COMMIT" \
    && rm -rf /opt/sd-scripts/.git \
    && python -m venv --system-site-packages /opt/kohya \
    && cd /opt/sd-scripts \
    && /opt/kohya/bin/pip install -r requirements.txt

# Las rutas de cache apuntan a los volumenes: nada se descarga en tiempo de ejecucion.
RUN mkdir -p /root/.cache /tmp/ultralytics && ln -s /weights/clip /root/.cache/clip

COPY . .

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -sf http://localhost:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "5"]
