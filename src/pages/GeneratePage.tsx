import { useEffect, useMemo, useState } from "react";
import { EngineModel, ImageRow, Job, Klass, Lora, Project, SystemStatus, get, post } from "../lib/api";
import { usePoll, useStored } from "../lib/hooks";
import { Card, Empty, Field, Notice, OriginPill, PageHead, Progress, StatusPill, Thumb, useAction } from "../ui/kit";

const DEFAULT_SCENARIOS = {
  fondo: "city street\noffice interior\npark with trees\nsupermarket aisle\nparking garage\nhome living room",
  luz: "soft daylight\novercast light\nwarm indoor lighting\nnight with street lights",
  encuadre: "full body shot\nmedium shot\nwide angle from a distance\nhigh angle view",
};
const AXIS_LABEL: Record<string, string> = { fondo: "Fondos", luz: "Iluminación", encuadre: "Encuadre" };
const DEFAULT_NEGATIVE = "cartoon, painting, illustration, 3d render, deformed, blurry, low quality, watermark, text";

export default function GeneratePage({ projectId }: { projectId: string }) {
  const { data: models, error: modelsError } = usePoll(() => get<EngineModel[]>("/engine/models"), 15000);
  const { data: options } = usePoll(() => get<{ samplers: string[]; schedulers: string[] }>("/engine/options"), 30000);
  const { data: loras } = usePoll(() => get<Lora[]>(`/loras?project_id=${projectId}`), 15000, [projectId]);
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), 10000, [projectId]);
  const { data: recent, refresh: reloadRecent } = usePoll(() => get<{ total: number; items: ImageRow[] }>(`/projects/${projectId}/images?origin=synthetic&limit=60`), 4000, [projectId]);
  const { data: system } = usePoll(() => get<SystemStatus>("/system"), 2000);
  const active = system?.active_jobs.find((j) => j.kind === "generate" && j.project_id === projectId);

  const [checkpoint, setCheckpoint] = useStored(`kosei_ckpt_${projectId}`, "");
  const [loraId, setLoraId] = useStored<string>(`kosei_lora_${projectId}`, "");
  const [weight, setWeight] = useStored(`kosei_lw_${projectId}`, 0.9);
  const [classId, setClassId] = useState<number | null>(null);
  const [prompt, setPrompt] = useStored(`kosei_prompt_${projectId}`, "photograph of a person");
  const [negative, setNegative] = useStored(`kosei_neg_${projectId}`, DEFAULT_NEGATIVE);
  const [scenarios, setScenarios] = useStored<Record<string, string>>(`kosei_scn_${projectId}`, DEFAULT_SCENARIOS);
  const [count, setCount] = useStored(`kosei_count_${projectId}`, 8);
  const [seed, setSeed] = useStored(`kosei_seed_${projectId}`, -1);
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(30);
  const [cfg, setCfg] = useState(5);
  const [distilled, setDistilled] = useState(3.5);
  const [sampler, setSampler] = useState("DPM++ 2M");
  const [scheduler, setScheduler] = useState("karras");
  const [batch, setBatch] = useState(1);
  const [refine, setRefine] = useStored(`kosei_refine_${projectId}`, false);
  const [refineDenoise, setRefineDenoise] = useStored(`kosei_refine_d_${projectId}`, 0.4);
  const [selected, setSelected] = useState<ImageRow | null>(null);
  const { busy, run } = useAction();

  const model = models?.find((m) => m.title === checkpoint);
  useEffect(() => {
    if (models?.length && !model) {
      setCheckpoint((models.find((m) => m.arch === "sdxl") ?? models[0]).title);
    }
  }, [models, model, setCheckpoint]);
  useEffect(() => {
    if (!model) return;
    const d = model.defaults;
    setWidth(d.width); setHeight(d.height); setSteps(d.steps); setCfg(d.cfg_scale);
    setSampler(d.sampler); setScheduler(d.scheduler); setBatch(d.batch_size);
    if (d.distilled_cfg_scale) setDistilled(d.distilled_cfg_scale);
  }, [model?.title]);

  const classes: Klass[] = project?.classes ?? [];
  useEffect(() => { if (classId === null && classes.length) setClassId(classes[0].id); }, [classes, classId]);

  const compatible = useMemo(() => (loras ?? []).filter((l) => l.status === "ready"), [loras]);
  const lora = compatible.find((l) => l.id === loraId);
  const mismatch = lora && model && lora.arch !== "unknown" && lora.arch !== model.arch;
  const axes = useMemo(() => Object.entries(scenarios).map(([k, v]) => [k, v.split("\n").map((s) => s.trim()).filter(Boolean)] as const), [scenarios]);
  const combos = axes.reduce((n, [, values]) => n * Math.max(1, values.length), 1);

  const start = () => run(async () => {
    await post(`/projects/${projectId}/generate`, {
      checkpoint, lora_id: lora?.id ?? null, lora_weight: weight, class_id: classId, prompt, negative_prompt: negative,
      scenarios: Object.fromEntries(axes), count, seed, steps, cfg_scale: cfg, distilled_cfg_scale: distilled,
      width, height, sampler, scheduler, batch_size: batch, refine, refine_denoise: refineDenoise,
    });
  }, "Generación en cola");

  return (
    <>
      <PageHead title="Generación" sub="Una imagen generada no es válida solo por existir: cada una pasa por el etiquetado y la revisión antes de entrar al dataset." />
      <div className="grid grid-side">
        <div className="stack">
          <Card title="Modelo">
            <div className="fields">
              {modelsError && <Notice kind="warn">{modelsError}</Notice>}
              <Field label="Checkpoint">
                <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
                  {models?.map((m) => <option key={m.title} value={m.title}>{m.name} ({m.arch_label})</option>)}
                </select>
              </Field>
              <Field label="LoRA">
                <select value={loraId} onChange={(e) => setLoraId(e.target.value)}>
                  <option value="">Sin LoRA</option>
                  {compatible.map((l) => {
                    const off = !!model && l.arch !== "unknown" && l.arch !== model.arch;
                    return <option key={l.id} value={l.id} disabled={off}>{l.name} ({l.arch_label}){off ? ", incompatible" : ""}</option>;
                  })}
                </select>
              </Field>
              {mismatch && <Notice kind="bad">Este LoRA es {lora!.arch_label} y el checkpoint es {model!.arch_label}. Elige una combinación compatible.</Notice>}
              {lora && (
                <Field label={`Peso del LoRA: ${weight.toFixed(2)}`}>
                  <input type="range" min={0} max={1.5} step={0.05} value={weight} onChange={(e) => setWeight(+e.target.value)} />
                </Field>
              )}
              <Field label="Clase que se etiquetará" hint="La clase viene del condicionamiento. El detector debe confirmar que el objeto está en la imagen.">
                <select value={classId ?? ""} onChange={(e) => setClassId(+e.target.value)}>
                  {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                </select>
              </Field>
              {!classes.length && <Notice kind="warn">Define una clase en Dataset para poder generar.</Notice>}
            </div>
          </Card>
          <Card title="Descripción">
            <div className="fields">
              <Field label="Prompt" hint={lora?.token ? `El token ${lora.token} se agrega si no aparece.` : undefined}>
                <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
              </Field>
              {model?.arch !== "flux" && (
                <Field label="Evitar"><textarea rows={2} value={negative} onChange={(e) => setNegative(e.target.value)} /></Field>
              )}
            </div>
          </Card>
          <Card title="Variación de escenarios" note={`${combos} combinaciones. No se repite ninguna hasta agotarlas.`}>
            <div className="fields">
              {Object.keys(DEFAULT_SCENARIOS).map((axis) => (
                <Field key={axis} label={`${AXIS_LABEL[axis]} (uno por línea, en inglés)`}>
                  <textarea rows={3} value={scenarios[axis] ?? ""} onChange={(e) => setScenarios({ ...scenarios, [axis]: e.target.value })} />
                </Field>
              ))}
            </div>
          </Card>
          <Card title="Lote">
            <div className="fields">
              <div className="fields-2">
                <Field label="Cantidad"><input type="number" min={1} max={5000} value={count} onChange={(e) => setCount(+e.target.value)} /></Field>
                <Field label="Seed base" hint="-1 elige una al azar. Se guarda por imagen."><input type="number" min={-1} value={seed} onChange={(e) => setSeed(+e.target.value)} /></Field>
              </div>
              <label className="row" style={{ fontSize: 13, alignItems: "flex-start", flexWrap: "nowrap" }}>
                <input type="checkbox" checked={refine} onChange={(e) => setRefine(e.target.checked)} style={{ marginTop: 3 }} />
                <span>Refinar el sujeto<br /><span className="field-hint">Regenera el rostro o el objeto a resolución completa cuando sale pequeño. Cuesta cerca de un tercio más de tiempo, y su mejora medida es leve y no concluyente. Revisa los resultados.</span></span>
              </label>
              {refine && (
                <Field label={`Fuerza del refinado: ${refineDenoise.toFixed(2)}`}>
                  <input type="range" min={0.15} max={0.7} step={0.05} value={refineDenoise} onChange={(e) => setRefineDenoise(+e.target.value)} />
                </Field>
              )}
              <details className="advanced">
                <summary>Ajustes avanzados</summary>
                <div className="fields">
                  <div className="fields-2">
                    <Field label="Ancho"><input type="number" step={8} min={256} max={2048} value={width} onChange={(e) => setWidth(+e.target.value)} /></Field>
                    <Field label="Alto"><input type="number" step={8} min={256} max={2048} value={height} onChange={(e) => setHeight(+e.target.value)} /></Field>
                    <Field label="Pasos"><input type="number" min={1} max={150} value={steps} onChange={(e) => setSteps(+e.target.value)} /></Field>
                    {model?.arch === "flux"
                      ? <Field label="Guía destilada"><input type="number" step={0.5} min={0} max={30} value={distilled} onChange={(e) => setDistilled(+e.target.value)} /></Field>
                      : <Field label="Escala CFG"><input type="number" step={0.5} min={1} max={30} value={cfg} onChange={(e) => setCfg(+e.target.value)} /></Field>}
                    <Field label="Muestreador">
                      <select value={sampler} onChange={(e) => setSampler(e.target.value)}>{options?.samplers.map((s) => <option key={s}>{s}</option>)}</select>
                    </Field>
                    <Field label="Programador">
                      <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>{options?.schedulers.map((s) => <option key={s}>{s}</option>)}</select>
                    </Field>
                    <Field label="Imágenes por tanda"><input type="number" min={1} max={8} value={batch} onChange={(e) => setBatch(+e.target.value)} /></Field>
                  </div>
                </div>
              </details>
            </div>
          </Card>
          <div className="sticky-action">
            <button className="btn btn-primary btn-block" disabled={busy || !system?.engine.ready || !checkpoint || !prompt.trim() || !classId || !!mismatch} onClick={start}>
              Generar {count} {count === 1 ? "imagen" : "imágenes"}
            </button>
            {!system?.engine.ready && <p className="field-hint">El motor aún está cargando.</p>}
          </div>
        </div>
        <div className="stack">
          <LivePreview job={active} selected={selected} />
          <Card title={`${recent?.total ?? 0} imágenes sintéticas`} actions={<button className="btn btn-small" onClick={reloadRecent}>Actualizar</button>}>
            {recent?.items.length
              ? <div className="gallery">{recent.items.map((i) => <Thumb key={i.id} image={i} selected={selected?.id === i.id} onClick={() => setSelected(i)} />)}</div>
              : <Empty title="Sin imágenes todavía">Configura el lote y genera las primeras.</Empty>}
          </Card>
        </div>
      </div>
    </>
  );
}

function LivePreview({ job, selected }: { job?: Job; selected: ImageRow | null }) {
  const { data: progress } = usePoll(
    () => (job ? get<{ current_image: string | null; progress: number; eta_relative: number }>("/engine/progress") : Promise.resolve(null)),
    job ? 1500 : null, [job?.id],
  );
  const [lastFrame, setLastFrame] = useState<string | null>(null);
  useEffect(() => { if (progress?.current_image) setLastFrame(progress.current_image); }, [progress?.current_image]);
  useEffect(() => { if (!job) setLastFrame(null); }, [job]);

  if (job) {
    return (
      <Card title={job.status === "queued" ? "En cola" : "Generando"} note={job.message}>
        <div className="stack">
          <Progress value={job.current} total={job.total} />
          <div className="preview">
            {lastFrame
              ? <img src={`data:image/png;base64,${lastFrame}`} alt="Vista previa de la imagen en generación" />
              : <span className="faint">La vista previa aparece en cuanto el motor empieza a muestrear.</span>}
          </div>
        </div>
      </Card>
    );
  }
  if (!selected) return null;
  const g = selected.generation ?? {};
  return (
    <Card title={selected.filename} actions={<div className="row"><OriginPill origin={selected.origin} /><StatusPill status={selected.status} /></div>}>
      <div className="grid grid-2">
        <div className="preview" style={{ minHeight: 0 }}><img src={selected.file_url} alt="" /></div>
        <dl className="meta-list">
          <dt>Prompt</dt><dd>{g.prompt}</dd>
          <dt>Seed</dt><dd className="mono">{g.seed}</dd>
          <dt>Modelo</dt><dd>{g.checkpoint}</dd>
          <dt>LoRA</dt><dd>{g.lora ?? "—"}</dd>
          <dt>Muestreo</dt><dd>{g.sampler} / {g.scheduler}, {g.steps} pasos</dd>
          <dt>Guía</dt><dd className="mono">{g.arch === "flux" ? g.distilled_cfg_scale : g.cfg_scale}</dd>
          <dt>Tamaño</dt><dd className="mono">{selected.width}×{selected.height}</dd>
          <dt>Refinado</dt><dd>{g.refined_region ? `Sí, fuerza ${g.refine_denoise}` : "No"}</dd>
          {selected.qc?.fidelity?.face != null && <><dt>Parecido</dt><dd className="mono">{selected.qc.fidelity.face.toFixed(2)}</dd></>}
          <dt>SHA-256</dt><dd className="mono">{selected.sha256.slice(0, 24)}…</dd>
        </dl>
      </div>
    </Card>
  );
}
