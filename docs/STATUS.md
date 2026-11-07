# Estado de KoseiStudio

Actualizado: 2026-11-05. Rama `entrega-final`. Contexto general y reglas: `../../context.md`.

## Arranque

`docker compose up -d` levanta `model-init` (descarga lo que falte y termina), `sd-webui` (Forge), `backend` y `frontend`. Interfaz en http://localhost.

## Verificado con ejecución real

| Área | Evidencia |
|---|---|
| Arranque | 3 servicios sanos; `model-init` descarga y verifica pesos |
| Comparación de modelos | RealVisXL 23 s/imagen, 9.7 GB de VRAM; Flux nf4 105 s/imagen, 11.2 GB; SD 1.5 3.7 s/imagen |
| Importación | 50 fotos por nginx; 1 casi duplicado detectado; partición 20/10/20 bloqueada (huella `7d71f3dd…`) |
| Etiquetado | YOLO11m COCO + MobileSAM en CPU, 1.6 s/imagen; máscara a partir del contorno mayor |
| LoRA SDXL | `katto_xl_v1`, 2000 pasos, 69 min, puntos de control a 500, 1000 y 1500 |
| Memoria y tiempos (6-nov, proyecto Demo) | Entrenamiento SDXL 1024 px: pico 7.6 GB de VRAM y 16.7 GB de RAM en contenedores. SDXL 1024, 30 pasos: 23 s/imagen, pico 9.2 GB de VRAM. SD 1.5 512, 30 pasos: 4.3 s/imagen. RAM en generación: pico 7.7 GB al cambiar de modelo. `katto_xl_v1` (2000 pasos): 4,036 s de ejecución. Almacenamiento: 26.5 GB de imágenes Docker + 37 GB de volúmenes |
| Con y sin LoRA (6-nov, proyecto Demo) | SDXL, semilla 20261106, 30 pasos, "photograph of ohwxkatto man wearing glasses": parecido 0.15 sin LoRA y 0.64 con `katto_xl_demo`; repetido dos veces con imágenes idénticas. El LoRA (rango 32, 170 MB) modifica 722 módulos, todos en bloques Transformer del U-Net; ‖ΔW‖ máximo en output_blocks 1 y 2 (10.5 y 10.7) |
| Fidelidad | Fotos reales no vistas 0.57-0.89; personas ajenas 0.06-0.36; LoRA final 0.54; con "ohwxkatto man wearing glasses", 0.51-0.66 |
| Refinado del sujeto | +0.03 de media en 6 pares; no concluyente. Apagado por defecto |
| Marca de agua | `dwtDctSvd`: 24 de 24 recuperadas, PSNR 41 dB; sin marca, máximo 59 % de bits |
| Interfaz | 10 pantallas en tema oscuro, claro, 1280×720 y 390 px: sin errores de consola ni desbordes |
| Interacciones | Dibujar caja y guardar (caja humana, verificada y con máscara SAM), tema, panel lateral con Escape, ayuda con teclado |

## Pruebas automáticas

- Backend: 51 simuladas (Forge y detector sustituidos) y 8 con pesos reales en CPU. Comando: `python -m pytest tests` dentro de la imagen.
- Frontend: 19 simuladas (`npm test`).

## Pendiente

- **Prueba de sistema `scripts/e2e_system.py` sin terminar.** La detuvo el sistema por falta de memoria en la PC. Falta: entrenar y cancelar, generar y cancelar, calidad, ZIP con `yolo train`, experimento, reinicio y prueba sin internet. Se relanza solo cuando lo indique el usuario.
- **Generación masiva y experimento sin correr.** Requieren que el usuario verifique las cajas de las 50 fotos reales.
- **Guardado de cajas.** En 1 de 4 intentos automatizados, el botón "Guardar y recalcular máscara" no envió la petición. No se reprodujo; las otras 3 respondieron 200. Aprobar también guarda las cajas.
- **Animaciones ligadas a trabajos** (pulso del riel, conectores, línea de escaneo): implementadas, pero no se observaron con un trabajo en curso.
- El LoRA aprendió de selfies y tiende al plano cercano: hay poca variedad de escalas para el experimento.
- LoRA Flux sin probar. Con 18 GB en WSL la memoria queda muy justa.
