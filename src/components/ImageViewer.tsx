import type { GenerateResult } from "../types";

interface Props {
  activeOutput: GenerateResult | null;
  isBusy: boolean;
  elapsedSeconds: number;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export default function ImageViewer({ activeOutput, isBusy, elapsedSeconds }: Props) {
  return (
    <div className="preview">
      {isBusy && !activeOutput && (
        <div className="preview-skeleton">
          <div className="skeleton-label">
            <span className="elapsed-badge">Generando... {formatTime(elapsedSeconds)}</span>
          </div>
        </div>
      )}

      {isBusy && activeOutput && (
        <div style={{ position: "relative" }}>
          <img
            src={activeOutput.image_url}
            alt={`Salida ${activeOutput.id}`}
            style={{ opacity: 0.5 }}
          />
          <div className="skeleton-overlay">
            <span className="elapsed-badge">Generando... {formatTime(elapsedSeconds)}</span>
          </div>
        </div>
      )}

      {!isBusy && activeOutput && (
        <img src={activeOutput.image_url} alt={`Salida ${activeOutput.id}`} />
      )}

      {!isBusy && !activeOutput && (
        <div className="placeholder">
          <div className="placeholder-title">Listo para generar</div>
          <div className="placeholder-text">
            Ejecuta un prompt para ver las imágenes aquí.
          </div>
        </div>
      )}

      {activeOutput && !isBusy && (
        <div className="preview-actions">
          <a
            className="ghost"
            href={activeOutput.image_url}
            download={`${activeOutput.id}.png`}
          >
            Descargar imagen
          </a>
          <a
            className="ghost"
            href={activeOutput.image_url}
            target="_blank"
            rel="noreferrer"
          >
            Abrir imagen
          </a>
        </div>
      )}
    </div>
  );
}
