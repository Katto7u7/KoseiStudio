import { useEffect, useState } from "react";
import { fetchModels, startTraining } from "../../api";
import type { CaptionEntry } from "../../types";

interface Props {
  jobId: string;
  tokenPrefix: string;
  captions: CaptionEntry[];
  imageCount: number;
  onBack: () => void;
  onStarted: (jobId: string) => void;
}

const DIM_OPTIONS = [4, 8, 16, 32];

export default function StepConfig({ jobId, tokenPrefix, captions, imageCount, onBack, onStarted }: Props) {
  const [models, setModels]           = useState<string[]>([]);
  const [baseModel, setBaseModel]     = useState("");
  const [outputName, setOutputName]   = useState("my_lora");
  const [nRepeats, setNRepeats]       = useState(100);
  const [networkDim, setNetworkDim]   = useState(8);
  const [lr, setLr]                   = useState("0.0001");
  const [batchSize, setBatchSize]     = useState(1);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState<string | null>(null);

  const autoSteps = Math.max(500, Math.round(imageCount * nRepeats * 0.12));
  const [maxSteps, setMaxSteps]       = useState(autoSteps);

  useEffect(() => {
    fetchModels().then((m) => { setModels(m); if (m.length > 0) setBaseModel(m[0]); }).catch(() => {});
  }, []);

  useEffect(() => {
    setMaxSteps(Math.max(500, Math.round(imageCount * nRepeats * 0.12)));
  }, [imageCount, nRepeats]);

  const handleStart = async () => {
    if (!baseModel) return;
    setLoading(true);
    setError(null);
    try {
      const result = await startTraining({
        job_id: jobId,
        output_name: outputName,
        base_model: baseModel,
        token_prefix: tokenPrefix,
        n_repeats: nRepeats,
        network_dim: networkDim,
        learning_rate: parseFloat(lr),
        max_steps: maxSteps,
        batch_size: batchSize,
        captions,
      });
      onStarted(result.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo iniciar el training");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {error && <div className="alert">{error}</div>}

      <div className="callout callout-warn" style={{ marginBottom: 14 }}>
        El training y la generación de imágenes comparten GPU. No generes imágenes mientras entrenas.
      </div>

      <span className="section-label">Modelo base</span>
      <div className="form-group" style={{ marginBottom: 16 }}>
        <label htmlFor="baseModel">Checkpoint SD</label>
        <select id="baseModel" value={baseModel} onChange={(e) => setBaseModel(e.target.value)}>
          {models.length === 0 && <option value="">Cargando...</option>}
          {models.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      <span className="section-label">Salida</span>
      <div className="form-grid" style={{ marginBottom: 16 }}>
        <div className="form-group">
          <label htmlFor="outputName">Nombre del LoRA</label>
          <input
            id="outputName"
            type="text"
            placeholder="my_lora"
            value={outputName}
            onChange={(e) => setOutputName(e.target.value.replace(/\s/g, "_"))}
          />
        </div>
        <div className="form-group">
          <label htmlFor="nRepeats">Repeticiones</label>
          <input id="nRepeats" type="number" min={10} max={500} value={nRepeats} onChange={(e) => setNRepeats(Number(e.target.value))} />
        </div>
      </div>

      <span className="section-label">Hiperparámetros</span>
      <div style={{ marginBottom: 10 }}>
        <label style={{ marginBottom: 8 }}>Network Dim (rango LoRA)</label>
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          {DIM_OPTIONS.map((d) => (
            <button
              key={d}
              type="button"
              className={`preset-btn${networkDim === d ? " active" : ""}`}
              style={{ flex: "none", width: 52 }}
              onClick={() => setNetworkDim(d)}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      <div className="form-grid" style={{ marginBottom: 16 }}>
        <div className="form-group">
          <label htmlFor="lr">Learning Rate</label>
          <input id="lr" type="text" value={lr} onChange={(e) => setLr(e.target.value)} />
        </div>
        <div className="form-group">
          <label htmlFor="maxSteps">Max Steps (auto)</label>
          <input id="maxSteps" type="number" min={100} value={maxSteps} onChange={(e) => setMaxSteps(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label htmlFor="batchSize">Batch Size</label>
          <select id="batchSize" value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))}>
            <option value={1}>1</option>
            <option value={2}>2</option>
            <option value={4}>4</option>
          </select>
        </div>
        <div className="form-group" style={{ justifyContent: "flex-end" }}>
          <label style={{ opacity: 0.5 }}>Estimado</label>
          <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-muted)", paddingTop: 8 }}>
            ~{imageCount} imgs × {nRepeats} reps
          </span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <button className="ghost" onClick={onBack} type="button">← Atrás</button>
        <button
          className="primary"
          style={{ width: "auto", padding: "9px 20px" }}
          onClick={handleStart}
          disabled={loading || !baseModel || !outputName.trim()}
          type="button"
        >
          {loading ? "Iniciando..." : "Iniciar entrenamiento"}
        </button>
      </div>
    </div>
  );
}
