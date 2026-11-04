import { useState } from "react";
import { generateCaptions } from "../../api";
import type { CaptionEntry } from "../../types";

interface Props {
  jobId: string;
  filenames: string[];
  previewUrls: string[];
  tokenPrefix: string;
  setTokenPrefix: (v: string) => void;
  captions: CaptionEntry[];
  setCaptions: (v: CaptionEntry[]) => void;
  onBack: () => void;
  onNext: () => void;
}

export default function StepCaption({
  jobId, filenames, previewUrls,
  tokenPrefix, setTokenPrefix,
  captions, setCaptions,
  onBack, onNext,
}: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const hasCaptions = captions.length > 0;

  const handleAutoCaption = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await generateCaptions(jobId, tokenPrefix);
      setCaptions(result.captions);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error al generar captions");
    } finally {
      setLoading(false);
    }
  };

  const updateCaption = (filename: string, newCaption: string) => {
    setCaptions(captions.map((c) => c.filename === filename ? { ...c, caption: newCaption } : c));
  };

  return (
    <div>
      <div className="callout callout-info" style={{ marginBottom: 14 }}>
        El token/prefix es la palabra clave que identifica al sujeto en los prompts (ej: <strong>ohwxcat</strong>). Se antepone a cada caption automáticamente.
      </div>

      {error && <div className="alert">{error}</div>}

      <div className="form-grid" style={{ marginBottom: 16 }}>
        <div className="form-group">
          <label htmlFor="tokenPrefix">Token / Prefix</label>
          <input
            id="tokenPrefix"
            type="text"
            placeholder="ohwxsubject"
            value={tokenPrefix}
            onChange={(e) => setTokenPrefix(e.target.value)}
          />
        </div>
        <div style={{ display: "flex", alignItems: "flex-end", paddingBottom: 12 }}>
          <button
            className="primary"
            style={{ width: "100%", padding: "7px 12px" }}
            onClick={handleAutoCaption}
            disabled={loading || !tokenPrefix.trim()}
            type="button"
          >
            {loading ? "Generando..." : hasCaptions ? "↺ Re-generar BLIP" : "Auto-generar BLIP"}
          </button>
        </div>
      </div>

      {hasCaptions ? (
        <div>
          <span className="section-label">Captions — edita si es necesario</span>
          <div className="caption-list" style={{ marginTop: 10 }}>
            {captions.map((c, i) => (
              <div key={c.filename} className="caption-item">
                <img
                  className="caption-thumb"
                  src={previewUrls[i] ?? ""}
                  alt={c.filename}
                />
                <textarea
                  value={c.caption}
                  onChange={(e) => updateCaption(c.filename, e.target.value)}
                  placeholder="Caption..."
                />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div style={{ textAlign: "center", padding: "24px", color: "var(--text-muted)", fontSize: 13 }}>
          {loading
            ? "Generando captions con BLIP..."
            : `${filenames.length} imágenes cargadas. Haz clic en "Auto-generar BLIP" para crear los captions.`}
        </div>
      )}

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 20 }}>
        <button className="ghost" onClick={onBack} type="button">← Atrás</button>
        <button
          className="primary"
          style={{ width: "auto", padding: "9px 20px" }}
          onClick={onNext}
          disabled={!hasCaptions}
          type="button"
        >
          Continuar →
        </button>
      </div>
    </div>
  );
}
