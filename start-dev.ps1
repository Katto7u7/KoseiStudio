# Atlas Forge — Desarrollo local
# Uso: .\start-dev.ps1  (desde UI_v3_REACT/)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "Arrancando Atlas Forge en modo desarrollo..." -ForegroundColor Cyan
Write-Host ""
Write-Host "  Backend  -> http://localhost:8000"  -ForegroundColor Green
Write-Host "  Frontend -> http://localhost:5173"  -ForegroundColor Green
Write-Host ""
Write-Host "Asegurate de que Stable Diffusion Forge este corriendo en puerto 7860." -ForegroundColor Yellow
Write-Host ""

# Backend en nueva ventana
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
  Set-Location '$root\backend'
  Write-Host 'Backend iniciando...' -ForegroundColor Cyan
  .venv\Scripts\python.exe server.py
"@

# Espera un segundo y arranca frontend
Start-Sleep -Seconds 1

# Frontend en nueva ventana
Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
  Set-Location '$root'
  Write-Host 'Frontend iniciando...' -ForegroundColor Cyan
  npm run dev
"@

Write-Host "Abriendo en http://localhost:5173 ..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Start-Process "http://localhost:5173"
