# Stable Diffusion WebUI Forge — API-only container
# Build time: ~20-40 min (downloads PyTorch + Forge deps)
# GPU required at runtime (NVIDIA, CUDA 11.8+)

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    git wget curl \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /webui

# Forge fijado al commit verificado en la imagen en uso
RUN git clone https://github.com/lllyasviel/stable-diffusion-webui-forge.git . \
    && git checkout dfdcbab685e57677014f05a3309b48cc87383167

# PyTorch with CUDA 11.8 — install before requirements to avoid version conflicts
RUN pip3 install --no-cache-dir \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Forge requirements (installs xformers, transformers, diffusers, etc.)
RUN pip3 install --no-cache-dir -r requirements_versions.txt

EXPOSE 7860

# Models are mounted as volumes at runtime:
#   /models/Stable-diffusion  — SD checkpoints
#   /models/Lora              — LoRA adapters
CMD ["python3", "launch.py", \
     "--api", \
     "--listen", \
     "--port", "7860", \
     "--no-download-sd-model", \
     "--ckpt-dir", "/models/Stable-diffusion", \
     "--lora-dir", "/models/Lora"]
