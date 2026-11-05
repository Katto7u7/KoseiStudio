# Volúmenes de KoseiStudio

Datos y modelos persistentes. `docker compose up -d` descarga lo que falte y nunca sobrescribe lo que ya existe. Nada de esto se versiona en git.

| Ruta | Contenido |
|---|---|
| `data/` | Base de datos, proyectos, imágenes, exportaciones y registros. **Es tu trabajo: respáldalo** |
| `models/Stable-diffusion/` | Checkpoints. Coloca aquí otros `.safetensors` y aparecerán en la app con su arquitectura detectada |
| `models/Lora/` | Adaptadores LoRA. Los que no entrenó la app aparecen como "Externo, no verificable" |
| `yolo/` | YOLO11, MobileSAM y YOLO-World para el etiquetado |
| `weights/` | BLIP, CLIP, tokenizadores e Inception |
| `output/` | Salidas de la versión anterior. La app actual no las usa |
