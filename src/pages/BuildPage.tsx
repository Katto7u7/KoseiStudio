import { useState } from "react";
import { Project, get, post } from "../lib/api";
import { formatBytes, formatDate, usePoll } from "../lib/hooks";
import { Card, Empty, Field, JobPill, Notice, PageHead, Progress, useAction } from "../ui/kit";

const TASK_LABEL: Record<string, string> = { detect: "Detección", segment: "Segmentación", classify: "Clasificación" };

export default function BuildPage({ projectId }: { projectId: string }) {
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), 5000, [projectId]);
  const { data: builds, refresh } = usePoll(() => get<any[]>(`/projects/${projectId}/builds`), 3000, [projectId]);
  const [name, setName] = useState("dataset_extendido");
  const [nSynth, setNSynth] = useState<number | "">("");
  const [seed, setSeed] = useState(0);
  const [tasks, setTasks] = useState<string[]>(["detect", "segment"]);
  const { busy, run } = useAction();
  const approvedSynth = project?.pipeline.by_status.synthetic.approved ?? 0;
  const approvedReal = project?.pipeline.by_status.real.approved ?? 0;
  const toggle = (t: string) => setTasks(tasks.includes(t) ? tasks.filter((x) => x !== t) : [...tasks, t]);

  return (
    <>
      <PageHead title="Dataset extendido" sub="Une lo real y lo sintético aprobados, en formato YOLO. Las sintéticas entran solo a train; val y test quedan solo con imágenes reales." />
      <div className="grid grid-side">
        <Card title="Nueva construcción">
          <div className="fields">
            <Notice kind="info">Disponibles: {approvedReal} reales y {approvedSynth} sintéticas aprobadas.</Notice>
            <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} /></Field>
            <Field label="Sintéticas a incluir" hint="Vacío incluye todas las aprobadas.">
              <input type="number" min={0} max={approvedSynth} value={nSynth} onChange={(e) => setNSynth(e.target.value === "" ? "" : +e.target.value)} />
            </Field>
            <Field label="Seed de selección"><input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} /></Field>
            <div className="field">
              <span>Tareas a exportar</span>
              {Object.entries(TASK_LABEL).map(([t, label]) => (
                <label key={t} className="row" style={{ color: "var(--paper)", fontSize: 13 }}>
                  <input type="checkbox" checked={tasks.includes(t)} onChange={() => toggle(t)} /> {label}
                </label>
              ))}
            </div>
            <button className="btn btn-primary btn-block" disabled={busy || !name.trim() || !tasks.length} onClick={() => run(() => post(`/projects/${projectId}/builds`, { name, n_synthetic: nSynth === "" ? null : nSynth, seed, tasks }), "Construcción en cola").then(refresh)}>
              Construir y validar
            </button>
          </div>
        </Card>
        <Card title="Construcciones">
          {builds?.length ? (
            <div className="stack">
              {builds.map((b) => <BuildRow key={b.id} build={b} />)}
            </div>
          ) : <Empty title="Sin construcciones">Aprueba imágenes en Etiquetas y construye el primer dataset.</Empty>}
        </Card>
      </div>
    </>
  );
}

function BuildRow({ build }: { build: any }) {
  const s = build.summary;
  const active = build.job_status === "queued" || build.job_status === "running";
  return (
    <div style={{ borderBottom: "1px solid var(--line)", paddingBottom: 14 }}>
      <div className="row-between" style={{ marginBottom: 8 }}>
        <div>
          <strong>{build.name}</strong>
          <div className="faint">{formatDate(build.created_at)}{s ? `, ${formatBytes(s.size_bytes)}` : ""}</div>
        </div>
        <div className="row">
          <JobPill status={build.job_status} />
          {build.status === "ready" && <a className="btn btn-small btn-primary" href={`/api/builds/${build.id}/download`}>Descargar ZIP</a>}
        </div>
      </div>
      {active && <Progress value={build.job_current} total={build.job_total} />}
      {build.job_error && <Notice kind="bad">{build.job_error}</Notice>}
      {s?.validation && (
        <p className="faint" style={{ marginBottom: 8 }}>
          {s.validation.ok ? `ZIP validado: ${s.validation.checked_images} archivos de imagen abiertos, cada uno con su etiqueta.` : `Validación con problemas: ${s.validation.problems.join("; ")}`}
        </p>
      )}
      {s?.counts && Object.entries<any>(s.counts).map(([task, splits]) => (
        <div className="table-wrap" key={task}>
          <table className="table">
            <thead><tr><th>{TASK_LABEL[task] ?? task}</th><th className="num">Reales</th><th className="num">Sintéticas</th><th className="num">Instancias</th><th className="num">Omitidas</th></tr></thead>
            <tbody>
              {Object.entries<any>(splits).map(([split, c]) => (
                <tr key={split}><td>{split}</td><td className="num">{c.real}</td><td className="num">{c.synthetic}</td><td className="num">{task === "classify" ? "—" : c.instances}</td><td className="num">{c.skipped}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
