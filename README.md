# KoseiStudio

Plataforma para ampliar datasets de visión por computadora con imágenes sintéticas: lleva un dataset real por entrenamiento de LoRA, generación, etiquetado, revisión y exportación, y compara después un detector entrenado solo con datos reales contra otro con datos reales y sintéticos.

## Arrancar

```powershell
docker compose up -d
```

Abre **http://localhost**. La interfaz responde en segundos; el motor de generación tarda unos minutos más y la interfaz muestra su estado real.

La primera vez, el servicio `model-init` descarga lo que falte (unos 25 GB) y las imágenes se construyen solas. Después todo funciona sin internet.

**Requisitos:** Docker Desktop con WSL2, GPU NVIDIA con 12 GB de VRAM, 16 GB de RAM asignados a WSL y 60 GB libres. Flux requiere más RAM.

```powershell
docker compose ps             # estado de los servicios
docker compose logs -f backend
docker compose down           # detiene todo; los datos quedan en ./volumes
```

## Flujo de trabajo

| Etapa | Qué haces | Qué garantiza el sistema |
|---|---|---|
| **Dataset** | Importas imágenes o un ZIP, defines clases y partición | El original no se modifica. Detecta corruptos y duplicados. Los casi duplicados quedan en el mismo split. Al bloquear, la prueba queda fijada con un hash |
| **LoRA** | Generas descripciones con BLIP, las corriges y entrenas | Usa solo train y registra sus hashes. Curva de pérdida real y puntos de control |
| **Generación** | Eliges modelo, LoRA y escenarios | Rechaza LoRAs incompatibles. Guarda cada imagen con seed y parámetros. Se puede cancelar y reanudar |
| **Etiquetas** | Revisas, corriges y apruebas | La clase viene del condicionamiento, la caja de YOLO y la máscara de SAM. Nada se exporta sin aprobación humana |
| **Calidad** | Evalúas lo generado | FID, KID, precisión y cobertura, diversidad, memorización y artefactos, con sus límites explicados |
| **Dataset extendido** | Eliges cuántas sintéticas incluir | Sintéticas solo en train. ZIP YOLO con manifiesto, abierto y validado al terminar |
| **Experimento** | Defines condiciones y seeds | Misma arquitectura, prueba real bloqueada, comprobación de fuga. El veredicto se calcula, no se ajusta |

## Arquitectura

```
nginx (127.0.0.1:80) -> FastAPI (interno) -> Forge (GPU)
                          |-- SQLite y archivos en ./volumes/data
                          |-- cola GPU y cola CPU en proceso
                          '-- subprocesos: kohya, entrenamiento YOLO
```

Es para un solo usuario en una máquina local, sin autenticación. Solo se publica en `127.0.0.1`: no lo expongas a una red sin agregar control de acceso.

| Ruta | Contenido |
|---|---|
| `backend/app/api` | Rutas HTTP por dominio |
| `backend/app/services` | Lógica: importación, etiquetado, generación, entrenamiento, calidad, exportación, experimento |
| `backend/app/jobs.py` | Cola persistente; al reiniciar, lo que corría queda `interrupted` y se puede reintentar |
| `src/pages`, `src/ui`, `src/lib` | Interfaz React |
| `scripts/e2e_system.py` | Prueba de sistema real |
| `docs/STATUS.md` | Estado verificado, evidencias y pendientes |

## Pruebas

```powershell
npm test                                                    # frontend (simuladas)
docker compose run --rm --no-deps backend python -m pytest  # backend (Forge y detector simulados)
python scripts\e2e_system.py <carpeta_con_jpg>              # sistema real, con GPU
```

## Desarrollo

Tras cambiar código, sube la etiqueta de imagen en `docker-compose.yml`, o reconstruye nombrando el servicio:

```powershell
docker compose build backend frontend
docker compose up -d
```

**No reconstruyas `sd-webui` sin necesidad:** son 15 GB y tarda más de 20 minutos.
