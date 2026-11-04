const PRESETS = [
  { label: "Cuadrado", w: 512, h: 512 },
  { label: "Retrato",  w: 512, h: 768 },
  { label: "Paisaje",  w: 768, h: 512 },
];

interface Props {
  prompt: string; setPrompt: (v: string) => void;
  negativePrompt: string; setNegativePrompt: (v: string) => void;
  steps: number; setSteps: (v: number) => void;
  cfgScale: number; setCfgScale: (v: number) => void;
  width: number; setWidth: (v: number) => void;
  height: number; setHeight: (v: number) => void;
  batch: number; setBatch: (v: number) => void;
  sampler: string; setSampler: (v: string) => void;
  sdModel: string; setSdModel: (v: string) => void;
  lora: string; setLora: (v: string) => void;
  targetClass: string; setTargetClass: (v: string) => void;
  classId: number; setClassId: (v: number) => void;
  models: string[]; loras: string[]; samplers: string[];
  isBusy: boolean;
  loadState: "idle" | "loading" | "success" | "error";
  error: string | null;
  onGenerate: () => void;
  onCancel: () => void;
  onRetry: () => void;
}

export default function Controls({
  prompt, setPrompt,
  negativePrompt, setNegativePrompt,
  steps, setSteps, cfgScale, setCfgScale,
  width, setWidth, height, setHeight,
  batch, setBatch,
  sampler, setSampler,
  sdModel, setSdModel,
  lora, setLora,
  targetClass, setTargetClass,
  classId, setClassId,
  models, loras, samplers,
  isBusy, loadState, error,
  onGenerate, onCancel, onRetry,
}: Props) {
  const isSquare    = width === 512 && height === 512;
  const isPortrait  = width === 512 && height === 768;
  const isLandscape = width === 768 && height === 512;

  return (
    <section className="panel controls stagger-1">

      {error && (
        <div className="alert">
          {error}
          {loadState === "error" && (
            <button className="ghost small" onClick={onRetry} style={{ marginLeft: 12 }}>
              Reintentar
            </button>
          )}
        </div>
      )}

      <span className="section-label">Prompt</span>
      <div className="form-group">
        <textarea
          id="prompt"
          rows={3}
          placeholder="Describe la escena, estilo y detalles clave"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
      </div>

      <span className="section-label">Canvas</span>
      <div className="preset-row">
        {PRESETS.map((p) => {
          const active =
            (p.label === "Cuadrado" && isSquare) ||
            (p.label === "Retrato"  && isPortrait) ||
            (p.label === "Paisaje"  && isLandscape);
          return (
            <button
              key={p.label}
              type="button"
              className={`preset-btn${active ? " active" : ""}`}
              onClick={() => { setWidth(p.w); setHeight(p.h); }}
            >
              {p.label}
              <span className="preset-dim">{p.w}×{p.h}</span>
            </button>
          );
        })}
      </div>

      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="width">Ancho</label>
          <input id="width" type="number" min={224} max={1920} value={width} onChange={(e) => setWidth(Number(e.target.value))} />
          {width > 1024 && <span className="field-hint">SD 1.x: artefactos sobre 1024 px.</span>}
        </div>
        <div className="form-group">
          <label htmlFor="height">Alto</label>
          <input id="height" type="number" min={224} max={1920} value={height} onChange={(e) => setHeight(Number(e.target.value))} />
          {height > 1024 && <span className="field-hint">SD 1.x: artefactos sobre 1024 px.</span>}
        </div>
      </div>

      <span className="section-label">Sampling</span>
      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="steps">Pasos</label>
          <input id="steps" type="number" min={1} max={99} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          {steps > 50 && <span className="field-hint">Más de 50 puede ser lento.</span>}
        </div>
        <div className="form-group">
          <label htmlFor="batch">Cantidad</label>
          <input id="batch" type="number" min={1} max={99} value={batch} onChange={(e) => setBatch(Number(e.target.value))} />
        </div>
      </div>

      <div className="form-group">
        <label htmlFor="sampler">Sampler</label>
        <select id="sampler" value={sampler} onChange={(e) => setSampler(e.target.value)}>
          {samplers.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <span className="section-label">Modelo</span>
      <div className="form-group">
        <label htmlFor="sdModel">Checkpoint</label>
        <select id="sdModel" value={sdModel} onChange={(e) => setSdModel(e.target.value)}>
          {models.length === 0 && <option value="">Cargando modelos...</option>}
          {models.map((m) => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      <span className="section-label">LoRA</span>
      <div className="form-group">
        <label htmlFor="lora">Adaptador</label>
        <select id="lora" value={lora} onChange={(e) => setLora(e.target.value)}>
          {loras.length === 0 && <option value="NINGUNO">NINGUNO</option>}
          {loras.map((l) => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      <span className="section-label">Etiqueta (Dataset)</span>
      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="targetClass">Clase objetivo</label>
          <input
            id="targetClass"
            type="text"
            placeholder="ej: person, dog"
            value={targetClass}
            onChange={(e) => setTargetClass(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="classId">ID de clase</label>
          <input
            id="classId"
            type="number"
            min={0}
            max={999}
            value={classId}
            onChange={(e) => setClassId(Number(e.target.value))}
          />
        </div>
      </div>

      <details className="advanced">
        <summary>Avanzado</summary>
        <div className="form-group">
          <label htmlFor="negativePrompt">Negative prompt</label>
          <textarea id="negativePrompt" rows={2} value={negativePrompt} onChange={(e) => setNegativePrompt(e.target.value)} />
        </div>
        <div className="form-group">
          <label htmlFor="cfgScale">CFG Scale</label>
          <input id="cfgScale" type="number" min={1} max={30} step={0.5} value={cfgScale} onChange={(e) => setCfgScale(Number(e.target.value))} />
        </div>
      </details>

      {isBusy ? (
        <button className="danger" onClick={onCancel} type="button">Cancelar generación</button>
      ) : (
        <button className="primary" onClick={onGenerate} disabled={!prompt.trim() || !sdModel} type="button" title="Ctrl+Enter">
          Generar
        </button>
      )}
    </section>
  );
}
