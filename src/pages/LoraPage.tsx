import { useEffect, useMemo, useRef, useState } from "react";
import { EngineModel, Job, Lora, Project, get, post, upload } from "../lib/api";
import { formatDate, formatDuration, usePoll } from "../lib/hooks";
import { LossChart } from "../ui/charts";
import { Card, Empty, Field, JobPill, Notice, PageHead, Progress, useAction } from "../ui/kit";

const ARCH_PRESET: Record<string, { resolution: number; steps: number; lr: number; dim: number; alpha: number; repeats: number; note: string }> = {
  sd15: { resolution: 512, steps: 1500, lr: 1e-4, dim: 16, alpha: 8, repeats: 10, note: "Unos 10 minutos en una RTX 3060." },
  sdxl: { resolution: 1024, steps: 1500, lr: 1e-4, dim: 16, alpha: 8, repeats: 10, note: "Se entrena solo el UNet para caber en 12 GB. Lleva más de media hora." },
  flux: { resolution: 768, steps: 1000, lr: 1e-4, dim: 16, alpha: 16, repeats: 10, note: "Requiere los componentes de Flux. Lleva más de una hora en 12 GB." },
};

export default function LoraPage({ projectId }: { projectId: string }) {
  const { data: loras, refresh } = usePoll(() => get<Lora[]>(`/loras?project_id=${projectId}`), 4000, [projectId]);
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), null, [projectId]);
  const training = useMemo(() => loras?.find((l) => l.job_status === "running" || l.job_status === "queued"), [loras]);
  const [selected, setSelected] = useState<string | null>(null);
  const shown = loras?.find((l) => l.id === (selected ?? training?.id));
  const fileRef = useRef<HTMLInputElement>(null);
  const { run } = useAction();
  return (
    <>
      <PageHead
        title="LoRA"
        sub="Adaptadores específicos del dataset. Un LoRA solo funciona con la arquitectura sobre la que se entrenó."
        actions={
          <>
            <button className="btn" onClick={() => fileRef.current?.click()}>Importar .safetensors</button>
            <input ref={fileRef} type="file" hidden accept=".safetensors" onChange={async (e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) { await run(() => upload(`/projects/${projectId}/loras/import`, [file], "file"), "LoRA importado"); refresh(); }
            }} />
          </>
        }
      />
      <div className="grid grid-side">
        <TrainForm project={project} onStarted={(id) => { setSelected(id); refresh(); }} />
        <div className="stack">
          {shown && shown.job_id && <TrainingDetail lora={shown} />}
          <Card title="Adaptadores">
            {loras?.length ? (
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Nombre</th><th>Arquitectura</th><th>Base</th><th>Linaje</th><th /></tr></thead>
                  <tbody>
                    {loras.map((l) => (
                      <tr key={l.id}>
                        <td><strong>{l.name}</strong>{l.token && <div className="faint mono">{l.token}</div>}</td>
                        <td>{l.arch_label}</td>
                        <td className="muted">{l.base_model ?? "—"}</td>
                        <td>
                          {l.origin === "trained"
                            ? <span className="pill pill-good">{l.status === "ready" ? `${l.train_image_count} imágenes de train` : l.status}</span>
                            : <span className="pill pill-warn">Externo, no verificable</span>}
                        </td>
                        <td style={{ textAlign: "right" }}>{l.job_id && <button className="btn btn-small" onClick={() => setSelected(l.id)}>Detalle</button>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <Empty title="Sin adaptadores">Entrena uno con el formulario.</Empty>}
          </Card>
        </div>
      </div>
    </>
  );
}

function TrainForm({ project, onStarted }: { project: Project | null; onStarted: (id: string) => void }) {
  const { data: models, error } = usePoll(() => get<EngineModel[]>("/engine/models"), 15000);
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

  const blockers: string[] = [];
  if (project && !project.pipeline.split_locked) blockers.push("Bloquea la partición del dataset antes de entrenar.");
  if (project && !project.pipeline.by_split.train) blockers.push("No hay imágenes en el split de entrenamiento.");
  const captions = project?.pipeline.train_captions ?? 0;
  const trainCount = project?.pipeline.by_split.train ?? 0;

  return (
    <Card title="Nuevo entrenamiento">
      <div className="fields">
        {blockers.map((b) => <Notice key={b} kind="warn">{b}</Notice>)}
        {!blockers.length && captions < trainCount && <Notice kind="warn">{trainCount - captions} imágenes de train no tienen descripción: se usará solo el token.</Notice>}
        {error && <Notice kind="warn">{error}</Notice>}
        <Field label="Modelo base" hint={model ? ARCH_PRESET[model.arch]?.note : undefined}>
          <select value={checkpoint} onChange={(e) => setCheckpoint(e.target.value)}>
            {trainable.map((m) => <option key={m.title} value={m.title}>{m.name} ({m.arch_label})</option>)}
          </select>
        </Field>
        <div className="fields-2">
          <Field label="Nombre del LoRA"><input value={name} onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9_-]/g, "_"))} placeholder="persona_xl_v1" /></Field>
          <Field label="Token"><input value={token} onChange={(e) => setToken(e.target.value.replace(/[^A-Za-z0-9_]/g, ""))} /></Field>
        </div>
        <div className="fields-2">
          <Field label="Pasos"><input type="number" min={10} value={steps} onChange={(e) => setSteps(+e.target.value)} /></Field>
          <Field label="Resolución"><input type="number" step={64} min={256} max={1536} value={resolution} onChange={(e) => setResolution(+e.target.value)} /></Field>
        </div>
        <details className="advanced">
          <summary>Ajustes avanzados</summary>
          <div className="fields">
            <div className="fields-2">
              <Field label="Dimensión de red"><input type="number" min={1} max={256} value={dim} onChange={(e) => setDim(+e.target.value)} /></Field>
              <Field label="Alpha"><input type="number" min={0.1} step={0.5} value={alpha} onChange={(e) => setAlpha(+e.target.value)} /></Field>
              <Field label="Tasa de aprendizaje"><input type="number" step={0.00001} min={0.00001} value={lr} onChange={(e) => setLr(+e.target.value)} /></Field>
              <Field label="Tamaño de lote"><input type="number" min={1} max={8} value={batch} onChange={(e) => setBatch(+e.target.value)} /></Field>
              <Field label="Repeticiones por imagen"><input type="number" min={1} value={repeats} onChange={(e) => setRepeats(+e.target.value)} /></Field>
              <Field label="Guardar cada (pasos)"><input type="number" min={10} value={saveEvery} onChange={(e) => setSaveEvery(+e.target.value)} /></Field>
            </div>
          </div>
        </details>
        <button
          className="btn btn-primary btn-block"
          disabled={busy || !checkpoint || !name || token.length < 2 || blockers.length > 0}
          onClick={async () => {
            const res = await run(() => post<{ lora_id: string }>(`/projects/${project!.id}/loras`, {
              name, checkpoint, token, steps, repeats, network_dim: dim, network_alpha: alpha,
              learning_rate: lr, batch_size: batch, resolution, save_every: saveEvery, seed: 42,
            }), "Entrenamiento en cola");
            if (res) onStarted(res.lora_id);
          }}
        >Iniciar entrenamiento</button>
        <p className="field-hint">Durante el entrenamiento, el motor de generación descarga su modelo para liberar VRAM.</p>
      </div>
    </Card>
  );
}

function TrainingDetail({ lora }: { lora: Lora }) {
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
        {log && <details className="advanced"><summary>Registro</summary><pre className="mono faint pre" style={{ margin: 0, fontSize: 11 }}>{log}</pre></details>}
        {job.result?.checkpoints?.length > 0 && <p className="faint">Puntos de control guardados: {job.result.checkpoints.join(", ")}</p>}
      </div>
    </Card>
  );
}
