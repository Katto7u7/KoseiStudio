import { useEffect, useMemo, useRef, useState } from "react";
import { EngineModel, Job, Lora, Project, del, get, post, upload } from "../lib/api";
import { formatDate, formatDuration, usePoll } from "../lib/hooks";
import { LossChart } from "../ui/charts";
import { BrainCircuit, Plus, Upload } from "lucide-react";
import { Card, Confirm, Drawer, Empty, Field, JobPill, Notice, PageHead, Progress, Requirement, Requirements, Segmented, useAction } from "../ui/kit";

const ARCH_PRESET: Record<string, { resolution: number; steps: number; lr: number; dim: number; alpha: number; repeats: number; note: string }> = {
  sd15: { resolution: 512, steps: 1500, lr: 1e-4, dim: 16, alpha: 8, repeats: 10, note: "Entrena el UNet y el codificador de texto." },
  sdxl: { resolution: 1024, steps: 1500, lr: 1e-4, dim: 16, alpha: 8, repeats: 10, note: "Se entrena solo el UNet para caber en 12 GB de VRAM." },
  flux: { resolution: 768, steps: 1000, lr: 1e-4, dim: 16, alpha: 16, repeats: 10, note: "Requiere los componentes de Flux y mucha memoria." },
};

const LORA_STATUS: Record<string, string> = { queued: "En cola", training: "Entrenando", failed: "Falló", cancelled: "Cancelado" };

type Preset = "rapido" | "equilibrado" | "fidelidad";
const PRESETS: Record<Preset, { label: string; steps: number; dim: number; alpha: number }> = {
  rapido: { label: "Rápido", steps: 1000, dim: 16, alpha: 8 },
  equilibrado: { label: "Equilibrado", steps: 1500, dim: 16, alpha: 8 },
  fidelidad: { label: "Máxima fidelidad", steps: 2000, dim: 32, alpha: 16 },
};

function LoraStatus({ lora }: { lora: Lora }) {
  if (lora.origin === "external") return <span className="pill pill-warn">Externo</span>;
  if (lora.status === "ready") return <span className="pill pill-good">{lora.train_image_count} fotos de train</span>;
  return <span className={`pill ${lora.status === "failed" ? "pill-bad" : lora.status === "cancelled" ? "pill-warn" : ""}`}>{LORA_STATUS[lora.status] ?? lora.status}</span>;
}

export default function LoraPage({ projectId }: { projectId: string }) {
  const { data: loras, refresh } = usePoll(() => get<Lora[]>(`/loras?project_id=${projectId}`), 4000, [projectId]);
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), null, [projectId]);
  const own = useMemo(() => (loras ?? []).filter((l) => l.project_id === projectId), [loras, projectId]);
  const external = useMemo(() => (loras ?? []).filter((l) => l.project_id !== projectId), [loras, projectId]);
  const training = own.find((l) => l.job_status === "running" || l.job_status === "queued");
  const [selected, setSelected] = useState<string | null>(null);
  const shown = (loras ?? []).find((l) => l.id === selected) ?? training ?? own.find((l) => l.status === "ready") ?? own[0];
  const [creating, setCreating] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const { run } = useAction();
  const [removing, setRemoving] = useState<Lora | null>(null);

  const row = (l: Lora) => (
    <button key={l.id} type="button" className={`lora-row${shown?.id === l.id ? " active" : ""}`} onClick={() => setSelected(l.id)}>
      <span className="lora-icon" aria-hidden><BrainCircuit size={16} /></span>
      <span className="lora-text">
        <strong>{l.name}</strong>
        <span className="faint">{l.arch_label}{l.token ? ` · ${l.token}` : ""}</span>
      </span>
      <LoraStatus lora={l} />
    </button>
  );

  return (
    <>
      <PageHead
        title="LoRA"
        sub="El adaptador aprende cómo es tu sujeto a partir de tus fotos de entrenamiento. Solo funciona con la arquitectura sobre la que se entrenó."
        actions={
          <>
            <button className="btn" onClick={() => fileRef.current?.click()}><Upload size={15} /> Importar .safetensors</button>
            <button className="btn btn-primary" onClick={() => setCreating(true)}><Plus size={16} /> Nuevo entrenamiento</button>
            <input ref={fileRef} type="file" hidden accept=".safetensors" onChange={async (e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) { await run(() => upload(`/projects/${projectId}/loras/import`, [file], "file"), "LoRA importado"); refresh(); }
            }} />
          </>
        }
      />
      <div className="grid grid-side">
        <div className="stack">
          <Card title="De este proyecto" note="Con linaje verificable: se sabe con qué fotos se entrenó.">
            {own.length ? <div className="stack-sm">{own.map(row)}</div> : <Empty title="Aún no entrenas ninguno" action={<button className="btn btn-primary" onClick={() => setCreating(true)}>Entrenar el primero</button>} />}
          </Card>
          {external.length > 0 && (
            <Card title="Externos" note="Encontrados en la carpeta de LoRAs. Sin linaje verificable.">
              <div className="stack-sm">{external.map(row)}</div>
            </Card>
          )}
        </div>
        <div className="stack">
          {shown ? (
            shown.job_id ? <TrainingDetail key={shown.id} lora={shown} onRemove={() => setRemoving(shown)} /> : (
              <Card title={shown.name} note={`${shown.arch_label} · ${shown.base_model ?? "base desconocida"}`}>
                <Notice kind="warn" title="LoRA externo">No se entrenó en esta aplicación: no se puede verificar con qué imágenes aprendió. Si se usa en el experimento, se reporta como posible fuga de datos.</Notice>
              </Card>
            )
          ) : (
            <Empty title="Entrena un LoRA con tus fotos">Usa solo el split de entrenamiento, para que la prueba siga siendo independiente. Al terminar verás su curva de aprendizaje y cuánto se parecen sus imágenes a tu sujeto.</Empty>
          )}
        </div>
      </div>
      {creating && <TrainForm project={project} onClose={() => setCreating(false)} onStarted={(id) => { setCreating(false); setSelected(id); refresh(); }} />}
      {removing && (
        <Confirm
          danger title="Eliminar LoRA" action="Eliminar"
          body={<>Se borrará <strong>{removing.name}</strong> y sus puntos de control. No se permite si ya generó imágenes: rompería su procedencia.</>}
          onClose={() => setRemoving(null)}
          onConfirm={async () => {
            const target = removing;
            setRemoving(null);
            await run(() => del(`/loras/${target.id}`), "LoRA eliminado");
            if (selected === target.id) setSelected(null);
            refresh();
          }}
        />
      )}
    </>
  );
}

function TrainForm({ project, onStarted, onClose }: { project: Project | null; onStarted: (id: string) => void; onClose: () => void }) {
  const { data: models, error } = usePoll(() => get<EngineModel[]>("/engine/models"), 15000);
  const { data: history } = usePoll(() => get<Job[]>(`/jobs?project_id=${project?.id ?? ""}&limit=100`), null, [project?.id]);
  const [preset, setPreset] = useState<Preset>("fidelidad");
  const trainable = useMemo(() => (models ?? []).filter((m) => m.arch !== "unknown"), [models]);
  const [checkpoint, setCheckpoint] = useState("");
  const [name, setName] = useState("");
  const [token, setToken] = useState("ohwxperson");
  const [steps, setSteps] = useState(1500);
  const [repeats, setRepeats] = useState(10);
  const [dim, setDim] = useState(16);
  const [alpha, setAlpha] = useState(8);
  const [lr, setLr] = useState(1e-4);
  const [batch, setBatch] = useState(1);
  const [resolution, setResolution] = useState(1024);
  const [saveEvery, setSaveEvery] = useState(250);
  const { busy, run } = useAction();
  const model = trainable.find((m) => m.title === checkpoint);

  useEffect(() => {
    if (!checkpoint && trainable.length) setCheckpoint((trainable.find((m) => m.arch === "sdxl") ?? trainable[0]).title);
  }, [trainable, checkpoint]);
  useEffect(() => {
    const preset = model && ARCH_PRESET[model.arch];
    if (!preset) return;
    setResolution(preset.resolution); setSteps(preset.steps); setLr(preset.lr);
    setDim(preset.dim); setAlpha(preset.alpha); setRepeats(preset.repeats);
  }, [model?.arch]);

  const choosePreset = (next: Preset) => {
    setPreset(next);
    setSteps(PRESETS[next].steps); setDim(PRESETS[next].dim); setAlpha(PRESETS[next].alpha);
  };
  useEffect(() => { if (model) choosePreset(preset); }, [model?.arch]);

  const captions = project?.pipeline.train_captions ?? 0;
  const trainCount = project?.pipeline.by_split.train ?? 0;
  const pid = project?.id ?? "";
  const requirements: Requirement[] = [
    { ok: !!project?.pipeline.split_locked, text: "Partición bloqueada: el LoRA no verá validación ni prueba", to: `/p/${pid}/dataset`, action: "Ir a partición" },
    { ok: trainCount > 0, text: `${trainCount} fotos en entrenamiento`, to: `/p/${pid}/dataset`, action: "Ir a dataset" },
    { ok: trainCount > 0 && captions >= trainCount, text: captions >= trainCount ? "Todas tienen descripción" : `${trainCount - captions} sin descripción: usarán solo el token`, to: `/p/${pid}/dataset`, action: "Describir" },
  ];
  const blockers = requirements.slice(0, 2).filter((r) => !r.ok);

  // Segundos por paso medidos en el ultimo entrenamiento terminado de la misma arquitectura.
  const perStep = useMemo(() => {
    if (!model) return null;
    const done = history?.find((j) => j.kind === "train_lora" && j.status === "succeeded" && j.params?.arch === model.arch && j.started_at && j.finished_at && j.total > 0);
    return done ? (done.finished_at! - done.started_at!) / done.total : null;
  }, [history, model]);

  const submit = async () => {
    const res = await run(() => post<{ lora_id: string }>(`/projects/${project!.id}/loras`, {
      name, checkpoint, token, steps, repeats, network_dim: dim, network_alpha: alpha,
      learning_rate: lr, batch_size: batch, resolution, save_every: saveEvery, seed: 42,
    }), "Entrenamiento en cola");
    if (res) onStarted(res.lora_id);
  };

  return (
    <Drawer
      title="Nuevo entrenamiento"
      onClose={onClose}
      footer={
        <div className="stack-sm">
          <button className="btn btn-primary btn-block btn-large" disabled={busy || !checkpoint || !name || token.length < 2 || blockers.length > 0} onClick={submit}>
            Iniciar entrenamiento
          </button>
          <span className="field-hint" style={{ textAlign: "center" }}>
            {perStep ? `Unos ${formatDuration(perStep * steps)}, según tu último entrenamiento con esta arquitectura.` : "Durante el entrenamiento el motor de generación libera la GPU."}
          </span>
        </div>
      }
    >
      <Requirements items={requirements} />
      {error && <Notice kind="warn">{error}</Notice>}
      <Field label="Modelo base" hint={model ? ARCH_PRESET[model.arch]?.note : undefined}>
        <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
          {trainable.map((m) => <option key={m.title} value={m.title}>{m.name} · {m.arch_label}</option>)}
        </select>
      </Field>
      <div className="fields-2">
        <Field label="Nombre" hint="Letras, números, _ y -."><input value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_-]/g, "_"))} placeholder="persona_xl_v2" /></Field>
        <Field label="Token" help="Palabra inventada que invoca a tu sujeto en los prompts. Debe ser rara para no chocar con palabras que el modelo ya conoce.">
          <input value={token} onChange={(e) => setToken(e.target.value.replace(/[^A-Za-z0-9_]/g, ""))} />
        </Field>
      </div>
      <div className="field">
        <span>Objetivo</span>
        <Segmented label="Objetivo" value={preset} onChange={choosePreset} options={(Object.keys(PRESETS) as Preset[]).map((k) => ({ value: k, label: PRESETS[k].label }))} />
        <span className="field-hint">{steps} pasos, red de {dim}. Más pasos suelen dar más parecido, hasta que el modelo empieza a copiar tus fotos.</span>
      </div>
      <details className="advanced">
        <summary>Ajustes avanzados</summary>
        <div className="fields-2">
          <Field label="Pasos"><input type="number" min={10} value={steps} onChange={(e) => setSteps(+e.target.value)} /></Field>
          <Field label="Resolución"><input type="number" step={64} min={256} max={1536} value={resolution} onChange={(e) => setResolution(+e.target.value)} /></Field>
          <Field label="Dimensión de red" help="Capacidad del adaptador. Más alta aprende más detalle y pesa más."><input type="number" min={1} max={256} value={dim} onChange={(e) => setDim(+e.target.value)} /></Field>
          <Field label="Alpha"><input type="number" min={0.1} step={0.5} value={alpha} onChange={(e) => setAlpha(+e.target.value)} /></Field>
          <Field label="Tasa de aprendizaje"><input type="number" step={0.00001} min={0.00001} value={lr} onChange={(e) => setLr(+e.target.value)} /></Field>
          <Field label="Tamaño de lote"><input type="number" min={1} max={8} value={batch} onChange={(e) => setBatch(+e.target.value)} /></Field>
          <Field label="Repeticiones por foto"><input type="number" min={1} value={repeats} onChange={(e) => setRepeats(+e.target.value)} /></Field>
          <Field label="Guardar cada (pasos)" help="Cada cuántos pasos se guarda un punto de control para comparar después."><input type="number" min={10} value={saveEvery} onChange={(e) => setSaveEvery(+e.target.value)} /></Field>
        </div>
      </details>
    </Drawer>
  );
}

function TrainingDetail({ lora, onRemove }: { lora: Lora; onRemove: () => void }) {
  const active = lora.job_status === "running" || lora.job_status === "queued";
  const { data: job } = usePoll(() => get<Job>(`/jobs/${lora.job_id}`), active ? 2000 : null, [lora.job_id, active]);
  const { data: losses } = usePoll(() => get<{ step: number; loss: number }[]>(`/loras/${lora.id}/losses`), active ? 5000 : null, [lora.id, active]);
  const { data: log } = usePoll(() => fetch(`/api/jobs/${lora.job_id}/log?tail=12`).then((r) => r.text()), active ? 5000 : null, [lora.job_id, active]);
  const { run, busy } = useAction();
  if (!job) return null;
  const elapsed = job.started_at ? (job.finished_at ?? Date.now() / 1000) - job.started_at : 0;
  const eta = job.current > 5 && job.total ? (elapsed / job.current) * (job.total - job.current) : null;
  return (
    <Card
      title={lora.name}
      note={`${lora.arch_label} sobre ${lora.base_model ?? "?"}, ${formatDate(lora.created_at)}`}
      actions={<div className="row">
        <JobPill status={job.status} />
        {active && <button className="btn btn-small btn-danger" disabled={busy} onClick={() => run(() => post(`/jobs/${job.id}/cancel`), "Cancelación solicitada")}>Detener</button>}
        {!active && job.status !== "succeeded" && <button className="btn btn-small" disabled={busy} onClick={() => run(() => post(`/jobs/${job.id}/retry`), "Reintento en cola")}>Reintentar</button>}
        {!active && <button className="btn btn-small" onClick={onRemove}>Eliminar</button>}
      </div>}
    >
      <div className="stack">
        <div>
          <div className="row-between" style={{ marginBottom: 6 }}>
            <span className="muted">{job.message}</span>
            <span className="faint mono">{formatDuration(elapsed)}{eta !== null && active ? `, faltan ${formatDuration(eta)}` : ""}</span>
          </div>
          <Progress value={job.current} total={job.total} />
        </div>
        {job.error && <Notice kind="bad" title="El entrenamiento falló">{job.error}</Notice>}
        <div>
          <h3 style={{ marginBottom: 6 }}>Pérdida media</h3>
          <LossChart points={losses ?? []} />
        </div>
        {lora.status === "ready" && lora.origin === "trained" && <FidelityPanel lora={lora} />}
        {log && <details className="advanced"><summary>Registro</summary><pre className="mono faint pre" style={{ margin: 0, fontSize: 11 }}>{log}</pre></details>}
        {job.result?.checkpoints?.length > 0 && <p className="faint">Puntos de control guardados: {job.result.checkpoints.join(", ")}</p>}
      </div>
    </Card>
  );
}

function FidelityPanel({ lora }: { lora: Lora }) {
  const { data: models } = usePoll(() => get<EngineModel[]>("/engine/models"), null);
  const [subject, setSubject] = useState("man");
  const [jobId, setJobId] = useState<string | null>(null);
  const { data: job } = usePoll(() => (jobId ? get<Job>(`/jobs/${jobId}`) : Promise.resolve(null)), jobId ? 2500 : null, [jobId]);
  const running = job?.status === "queued" || job?.status === "running";
  const { data: evaluation, refresh } = usePoll(() => get<any>(`/loras/${lora.id}/evaluation`), running ? 5000 : null, [lora.id, running]);
  const { busy, run } = useAction();
  const base = models?.find((m) => m.title.split(" [")[0] === lora.base_model);
  const checkpoints: any[] = evaluation?.checkpoints ?? [];
  const best = checkpoints.reduce((a, c) => ((c.face_mean ?? c.clip_mean ?? -1) > (a?.face_mean ?? a?.clip_mean ?? -1) ? c : a), null as any);
  return (
    <div className="stack" style={{ borderTop: "1px solid var(--line)", paddingTop: 14 }}>
      <div>
        <h3>Fidelidad por punto de control</h3>
        <p className="faint">Genera las mismas escenas con cada punto de control y mide cuánto se parecen al sujeto real. Un LoRA sobreentrenado puede empeorar al final.</p>
      </div>
      <div className="row" style={{ alignItems: "flex-end" }}>
        <Field label="Palabra de la clase"><input value={subject} onChange={(e) => setSubject(e.target.value)} style={{ width: 140 }} /></Field>
        <button className="btn" disabled={busy || running || !base} onClick={async () => {
          const res = await run(() => post<{ job_id: string }>(`/loras/${lora.id}/evaluate`, { checkpoint: base!.title, subject }), "Medición en cola");
          if (res) setJobId(res.job_id);
        }}>{running ? "Midiendo..." : "Medir fidelidad"}</button>
      </div>
      {job?.error && <Notice kind="bad">{job.error}</Notice>}
      {checkpoints.length > 0 && (
        <div className="stack">
          {evaluation.real_baseline != null && <p className="faint">Referencia: entre fotos reales del sujeto el parecido facial es {evaluation.real_baseline}. El umbral de misma persona es {evaluation.threshold}.</p>}
          {checkpoints.map((c) => (
            <div key={c.file} style={{ display: "grid", gap: 8 }}>
              <div className="row-between">
                <div className="row">
                  <strong className="mono">{c.is_final ? "Final" : `Paso ${c.step}`}</strong>
                  {best?.file === c.file && <span className="pill pill-good">Mayor parecido</span>}
                  <span className="muted">Rostro {c.face_mean ?? "—"}, misma persona {c.same_person_rate != null ? `${Math.round(c.same_person_rate * 100)} %` : "—"}, CLIP {c.clip_mean ?? "—"}</span>
                </div>
                {!c.is_final && <button className="btn btn-small" disabled={busy} onClick={() => run(() => post(`/loras/${lora.id}/promote`, { file: c.file }), "Punto de control publicado como LoRA").then(refresh)}>Usar este</button>}
              </div>
              <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(92px, 1fr))", gap: 6 }}>
                {c.shots.map((shot: any) => (
                  <a key={shot.file} href={`/api/loras/${lora.id}/evaluation/${c.file.replace(".safetensors", "")}/${shot.file}`} target="_blank" rel="noreferrer" title={`Rostro ${shot.face ?? "no detectado"}, CLIP ${shot.clip}`}>
                    <img src={`/api/loras/${lora.id}/evaluation/${c.file.replace(".safetensors", "")}/${shot.file}`} alt={`Muestra de ${c.is_final ? "final" : `paso ${c.step}`}`} loading="lazy" style={{ width: "100%", aspectRatio: "1", objectFit: "cover", borderRadius: 3, display: "block" }} />
                  </a>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
