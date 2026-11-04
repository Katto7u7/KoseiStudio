# Volúmenes de Atlas Forge

Coloca los modelos en las carpetas correspondientes antes de ejecutar `docker compose up`.

## models/Stable-diffusion/
Checkpoints de Stable Diffusion (`.ckpt` o `.safetensors`):
- Copia aquí: `sd-v1-4.ckpt`, `realisticVisionV60B1_v51VAE.safetensors`, etc.

## models/Lora/
Adaptadores LoRA (`.safetensors`):
- Copia aquí todos los archivos de `UIv0_1/modelos/LORA/`

## yolo/
Modelos de auto-etiquetado YOLO (`.pt`):
- `yolo11n-cls.pt`
- `yolo11n.pt`
- `yolo11n-seg.pt`
- Fuente: `UIv0_1/modelosAutoEtiquetas/`

## output/
Generado automáticamente por el sistema:
- `output/images/` — imágenes generadas (PNG)
- `output/metadata/` — metadatos de generación + etiquetas YOLO (TXT)
