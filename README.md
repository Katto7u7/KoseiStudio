# UI_v3_REACT

React + TypeScript UI with a small FastAPI backend that reuses the existing UIv0_1 logic.

## Backend

```powershell
cd UI_v3_REACT\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python server.py
```

Environment overrides (optional):

- `SD_HOST` (default `127.0.0.1`)
- `SD_PORT` (default `7860`)
- `LEGACY_UI_DIR` (default `../UIv0_1`)
- `LORA_DIR` (default `../UIv0_1/modelos/LORA`)
- `AUTO_MODELS_DIR` (default `../UIv0_1/modelosAutoEtiquetas`)
- `FRONTEND_ORIGINS` (default `http://localhost:5173,http://127.0.0.1:5173`)
- `ENABLE_ADETAILER` (default `1`; set `0` to disable if the ADetailer script is not installed)

## Frontend

```powershell
cd UI_v3_REACT
npm install
npm run dev
```

The Vite dev server proxies `/api` and `/assets` to `http://127.0.0.1:8000`.
