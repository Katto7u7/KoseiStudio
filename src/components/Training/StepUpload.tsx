import { useRef, useState } from "react";
import { uploadTrainingImages } from "../../api";

interface Props {
  onComplete: (jobId: string, filenames: string[], previewUrls: string[]) => void;
}

export default function StepUpload({ onComplete }: Props) {
  const [files, setFiles]           = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState<string | null>(null);
  const [dragOver, setDragOver]     = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (incoming: FileList | File[]) => {
    const arr = Array.from(incoming).filter((f) =>
      f.type.startsWith("image/")
    );
    setFiles((prev) => {
      const all = [...prev, ...arr];
      setPreviewUrls(all.map((f) => URL.createObjectURL(f)));
      return all;
    });
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  };

  const handleContinue = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const result = await uploadTrainingImages(files);
      onComplete(result.job_id, result.filenames, previewUrls);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error al subir imágenes");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {files.length < 20 && files.length > 0 && (
        <div className="callout callout-warn" style={{ marginBottom: 14 }}>
          Se recomiendan al menos 20 imágenes para un buen entrenamiento. Tienes {files.length}.
        </div>
      )}

      {error && <div className="alert">{error}</div>}

      <div
        className={`drop-zone${dragOver ? " drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
      >
        <span className="drop-zone-icon">⬆</span>
        <div className="drop-zone-title">Arrastra imágenes aquí</div>
        <div className="drop-zone-sub">o haz clic para seleccionar · JPG, PNG · 512×512 recomendado</div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />
      </div>

      {previewUrls.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="gallery-header">
            <span className="gallery-count">{files.length} imágenes seleccionadas</span>
            <button
              className="ghost small"
              onClick={() => { setFiles([]); setPreviewUrls([]); }}
              type="button"
            >
              Limpiar
            </button>
          </div>
          <div className="thumb-grid">
            {previewUrls.slice(0, 30).map((url, i) => (
              <div key={i} className="thumb">
                <img src={url} alt={`img-${i}`} />
              </div>
            ))}
            {files.length > 30 && (
              <div
                className="thumb"
                style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 64, fontSize: 12, color: "var(--text-muted)" }}
              >
                +{files.length - 30}
              </div>
            )}
          </div>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 20 }}>
        <button
          className="primary"
          style={{ width: "auto", padding: "9px 20px" }}
          onClick={handleContinue}
          disabled={files.length === 0 || loading}
          type="button"
        >
          {loading ? "Subiendo..." : "Continuar →"}
        </button>
      </div>
    </div>
  );
}
