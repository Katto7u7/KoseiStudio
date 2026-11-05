import { useEffect, useMemo, useState } from "react";
import { EngineModel, ImageRow, Job, Klass, Lora, Project, SystemStatus, get, post } from "../lib/api";
import { usePoll, useStored, formatDuration } from "../lib/hooks";
import { Sparkles, Wand2 } from "lucide-react";
import { Card, Check, Empty, Field, Help, Notice, OriginPill, PageHead, Progress, StatusPill, Thumb, useAction } from "../ui/kit";

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
  const { data: history } = usePoll(() => get<Job[]>(`/jobs?project_id=${projectId}&limit=60`), 30000, [projectId]);
  const active = system?.active_jobs.find((j) => j.kind === "generate" && j.project_id === projectId);

  const [checkpoint, setCheckpoint] = useStored(`kosei_ckpt_${projectId}`, "");
  const [loraId, setLoraId] = useStored<string | null>(`kosei_lora2_${projectId}`, null);
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
  // Sin eleccion previa, se propone el LoRA mas reciente de este proyecto compatible con el modelo.
  useEffect(() => {
    if (loraId !== null || !model) return;
    const own = compatible.find((l) => l.project_id === projectId && l.origin === "trained" && l.arch === model.arch);
    if (own) setLoraId(own.id);
  }, [loraId, model, compatible, projectId, setLoraId]);
  const mismatch = lora && model && lora.arch !== "unknown" && lora.arch !== model.arch;
  const axes = useMemo(() => Object.entries(scenarios).map(([k, v]) => [k, v.split("\n").map((s) => s.trim()).filter(Boolean)] as const), [scenarios]);
  const combos = axes.reduce((n, [, values]) => n * Math.max(1, values.length), 1);

  // Tiempo por imagen medido en la ultima generacion terminada de este proyecto, no supuesto.
  const measured = useMemo(() => {
    const last = history?.find((j) => j.kind === "generate" && j.status === "succeeded" && j.started_at && j.finished_at && j.total > 0);
    return last ? (last.finished_at! - last.started_at!) / last.total : null;
  }, [history]);
  const recommended = lora?.token ? `photograph of ${lora.token} ${classes.find((c) => c.id === classId)?.detector_prompt || "person"}` : null;

  const start = () => run(async () => {
    await post(`/projects/${projectId}/generate`, {
      checkpoint, lora_id: lora?.id ?? null, lora_weight: weight, class_id: classId, prompt, negative_prompt: negative,
      scenarios: Object.fromEntries(axes), count, seed, steps, cfg_scale: cfg, distilled_cfg_scale: distilled,
      width, height, sampler, scheduler, batch_size: batch, refine, refine_denoise: refineDenoise,
    });
  }, "Generación en cola");

  return (
    <>
      <PageHead title="Generación" sub="Crea escenas nuevas de tu sujeto. Cada imagen pasa después por el etiquetado y tu revisión antes de entrar al dataset." />
      <div className="grid grid-studio">
        <div className="stack">
          <LivePreview job={active} selected={selected} />
          <Card title="Resultados" note={`${recent?.total ?? 0} imágenes sintéticas en el proyecto`} actions={<button className="btn btn-small" onClick={reloadRecent}>Actualizar</button>}>
            {recent?.items.length
              ? <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))" }}>{recent.items.map((i, n) => <Thumb key={i.id} index={n} image={i} selected={selected?.id === i.id} onClick={() => setSelected(i)} />)}</div>
              : <Empty title="Sin imágenes todavía">Configura el lote en el panel de ajustes y genera las primeras.</Empty>}
          </Card>
        </div>

        <div className="panel-sticky">
          <div className="panel-scroll">
            <Card title="Modelo">
              <div className="fields">
                {modelsError && <Notice kind="warn">{modelsError}</Notice>}
                <Field label="Modelo base" help="El modelo de difusión que pinta la imagen. Tu LoRA debe ser de la misma arquitectura.">
                  <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
                    {models?.map((m) => <option key={m.title} value={m.title}>{m.name} · {m.arch_label}</option>)}
                  </select>
                </Field>
                <Field label="Tu sujeto (LoRA)" help="El adaptador entrenado con tus fotos. Es lo que hace que aparezca tu persona u objeto.">
                  <select value={loraId ?? ""} onChange={(e) => setLoraId(e.target.value)}>
                    <option value="">Sin LoRA</option>
                    {compatible.map((l) => {
                      const off = !!model && l.arch !== "unknown" && l.arch !== model.arch;
                      return <option key={l.id} value={l.id} disabled={off}>{l.name} · {l.arch_label}{off ? " (incompatible)" : ""}</option>;
                    })}
                  </select>
                </Field>
                {mismatch && <Notice kind="bad">Este LoRA es {lora!.arch_label} y el modelo es {model!.arch_label}. Elige una combinación compatible.</Notice>}
                {lora && (
                  <Field label={`Intensidad del sujeto: ${weight.toFixed(2)}`} hint="Más alto se parece más; demasiado alto puede deformar la imagen.">
                    <input type="range" min={0} max={1.5} step={0.05} value={weight} onChange={(e) => setWeight(+e.target.value)} />
                  </Field>
                )}
                <Field label="Clase que se etiquetará" help="Viene del condicionamiento: el detector debe confirmar después que el objeto está en la imagen.">
                  <select value={classId ?? ""} onChange={(e) => setClassId(+e.target.value)}>
                    {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                  </select>
                </Field>
                {!classes.length && <Notice kind="warn">Define una clase en Dataset para poder generar.</Notice>}
              </div>
            </Card>

            <Card title="Descripción">
              <div className="fields">
                <Field label="Qué quieres ver (en inglés)" hint={lora?.token ? `Escribe el token y qué es, por ejemplo "${lora.token} man". El token se agrega si falta.` : undefined}>
                  <textarea rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
                </Field>
                {recommended && prompt !== recommended && (
                  <button type="button" className="btn btn-small" style={{ justifySelf: "start" }} onClick={() => setPrompt(recommended)}><Sparkles size={14} /> Usar "{recommended}"</button>
                )}
                {model?.arch !== "flux" && (
                  <details className="advanced">
                    <summary>Qué evitar</summary>
                    <textarea rows={2} aria-label="Qué evitar" value={negative} onChange={(e) => setNegative(e.target.value)} />
                  </details>
                )}
              </div>
            </Card>

            <Card title="Variación" note={`${combos} combinaciones, sin repetir hasta agotarlas`}>
              <div className="fields">
                {Object.keys(DEFAULT_SCENARIOS).map((axis) => (
                  <Field key={axis} label={`${AXIS_LABEL[axis]}, uno por línea`}>
                    <textarea rows={3} value={scenarios[axis] ?? ""} onChange={(e) => setScenarios({ ...scenarios, [axis]: e.target.value })} />
                  </Field>
                ))}
              </div>
            </Card>

            <Card title="Lote">
              <div className="fields">
                <div className="fields-2">
                  <Field label="Cantidad"><input type="number" min={1} max={5000} value={count} onChange={(e) => setCount(+e.target.value)} /></Field>
                  <Field label="Semilla (seed)" help="Número de partida del azar. La misma semilla repite la misma imagen. -1 elige una al azar; cada imagen guarda la suya.">
                    <input type="number" min={-1} value={seed} onChange={(e) => setSeed(+e.target.value)} />
                  </Field>
                </div>
                <Check checked={refine} onChange={setRefine}>
                  Refinar el sujeto
                  <span className="field-hint" style={{ display: "block" }}>Regenera el rostro o el objeto a resolución completa cuando sale pequeño. Cuesta cerca de un tercio más de tiempo y su mejora medida es leve.</span>
                </Check>
                {refine && (
                  <Field label={`Fuerza del refinado: ${refineDenoise.toFixed(2)}`}>
                    <input type="range" min={0.15} max={0.7} step={0.05} value={refineDenoise} onChange={(e) => setRefineDenoise(+e.target.value)} />
                  </Field>
                )}
                <details className="advanced">
                  <summary>Ajustes avanzados</summary>
                  <div className="fields-2">
                    <Field label="Ancho"><input type="number" step={8} min={256} max={2048} value={width} onChange={(e) => setWidth(+e.target.value)} /></Field>
                    <Field label="Alto"><input type="number" step={8} min={256} max={2048} value={height} onChange={(e) => setHeight(+e.target.value)} /></Field>
                    <Field label="Pasos" help="Iteraciones de limpieza del ruido. Más pasos, más detalle y más tiempo."><input type="number" min={1} max={150} value={steps} onChange={(e) => setSteps(+e.target.value)} /></Field>
                    {model?.arch === "flux"
                      ? <Field label="Guía destilada"><input type="number" step={0.5} min={0} max={30} value={distilled} onChange={(e) => setDistilled(+e.target.value)} /></Field>
                      : <Field label="Fidelidad al texto (CFG)" help="Cuánto obedece al texto. Muy alto satura la imagen."><input type="number" step={0.5} min={1} max={30} value={cfg} onChange={(e) => setCfg(+e.target.value)} /></Field>}
                    <Field label="Muestreador">
                      <select value={sampler} onChange={(e) => setSampler(e.target.value)}>{options?.samplers.map((s) => <option key={s}>{s}</option>)}</select>
                    </Field>
                    <Field label="Programador">
                      <select value={scheduler} onChange={(e) => setScheduler(e.target.value)}>{options?.schedulers.map((s) => <option key={s}>{s}</option>)}</select>
                    </Field>
                    <Field label="Imágenes por tanda"><input type="number" min={1} max={8} value={batch} onChange={(e) => setBatch(+e.target.value)} /></Field>
                  </div>
                </details>
              </div>
            </Card>

            <div className="sticky-action">
              <button className="btn btn-primary btn-block btn-large" disabled={busy || !system?.engine.ready || !checkpoint || !prompt.trim() || !classId || !!mismatch} onClick={start}>
                <Wand2 size={17} /> Generar {count} {count === 1 ? "imagen" : "imágenes"}
              </button>
              <p className="field-hint" style={{ marginTop: 6, textAlign: "center" }}>
                {!system?.engine.ready ? "El motor aún está cargando."
                  : measured ? <>Unos {formatDuration(measured * count * (refine ? 1.3 : 1))}, según tu última generación ({measured.toFixed(0)} s por imagen) <Help>Se calcula con el tiempo real de la última generación terminada en este proyecto. Cambia si usas otro modelo o tamaño.</Help></>
                  : "El tiempo estimado aparece tras tu primera generación."}
              </p>
            </div>
          </div>
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
      <div className="stack-sm">
        <div className="preview">
          {lastFrame
            ? <img src={`data:image/png;base64,${lastFrame}`} alt="Vista previa de la imagen en generación" />
            : <div className="preview-empty"><strong>{job.status === "queued" ? "En cola" : "Preparando el modelo"}</strong><span>La vista previa aparece en cuanto el motor empieza a pintar.</span></div>}
          {job.status === "running" && <span className="scanline" aria-hidden />}
          <span className="preview-meta">{job.message}</span>
        </div>
        <Progress value={job.current} total={job.total} live />
      </div>
    );
  }
  if (!selected) {
    return (
      <div className="preview">
        <div className="preview-empty">
          <strong>Aquí verás cada imagen mientras se genera</strong>
          <span>Configura tu sujeto y el lote en el panel de ajustes. Al elegir un resultado verás sus detalles.</span>
        </div>
      </div>
    );
  }
  const g = selected.generation ?? {};
  return (
    <div className="stack-sm">
      <div className="preview"><img key={selected.id} src={selected.file_url} alt="" /></div>
      <Card title={`Seed ${g.seed}`} actions={<div className="row"><OriginPill origin={selected.origin} /><StatusPill status={selected.status} /></div>}>
        <dl className="meta-list">
          <dt>Prompt</dt><dd>{g.prompt}</dd>
          <dt>Modelo</dt><dd>{g.checkpoint}</dd>
          <dt>LoRA</dt><dd>{g.lora ?? "—"}{g.lora_weight ? ` · intensidad ${g.lora_weight}` : ""}</dd>
          <dt>Muestreo</dt><dd>{g.sampler} / {g.scheduler}, {g.steps} pasos, guía {g.arch === "flux" ? g.distilled_cfg_scale : g.cfg_scale}</dd>
          <dt>Tamaño</dt><dd className="mono">{selected.width}×{selected.height}</dd>
          <dt>Refinado</dt><dd>{g.refined_region ? `Sí, fuerza ${g.refine_denoise}` : "No"}</dd>
          {selected.qc?.fidelity?.face != null && <><dt>Parecido</dt><dd className="mono">{selected.qc.fidelity.face.toFixed(2)}</dd></>}
          <dt>SHA-256</dt><dd className="mono">{selected.sha256.slice(0, 24)}…</dd>
        </dl>
      </Card>
    </div>
  );
}
