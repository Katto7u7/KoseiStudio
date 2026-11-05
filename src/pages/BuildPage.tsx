import { useState } from "react";
import { Project, get, post } from "../lib/api";
import { formatBytes, formatDate, usePoll } from "../lib/hooks";
import { Download, PackagePlus } from "lucide-react";
import { Card, Check, Drawer, Empty, Field, JobPill, Notice, PageHead, Progress, Requirement, Requirements, splitLabel, useAction } from "../ui/kit";

const TASK_LABEL: Record<string, string> = { detect: "Detección", segment: "Segmentación", classify: "Clasificación" };

export default function BuildPage({ projectId }: { projectId: string }) {
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), 5000, [projectId]);
  const { data: builds, refresh } = usePoll(() => get<any[]>(`/projects/${projectId}/builds`), 3000, [projectId]);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("dataset_extendido");
  const [nSynth, setNSynth] = useState<number | "">("");
  const [seed, setSeed] = useState(0);
  const [tasks, setTasks] = useState<string[]>(["detect", "segment"]);
  const { busy, run } = useAction();
  const pl = project?.pipeline;
  const approvedSynth = pl?.by_status.synthetic.approved ?? 0;
  const approvedReal = pl?.by_status.real.approved ?? 0;
  const toggle = (t: string) => setTasks(tasks.includes(t) ? tasks.filter((x) => x !== t) : [...tasks, t]);
  const requirements: Requirement[] = [
    { ok: !!pl?.split_locked, text: "Partición bloqueada", to: `/p/${projectId}/dataset`, action: "Ir a partición" },
    { ok: approvedReal > 0, text: `${approvedReal} reales aprobadas`, to: `/p/${projectId}/review`, action: "Revisar" },
    { ok: approvedSynth > 0, text: `${approvedSynth} sintéticas aprobadas`, to: `/p/${projectId}/review`, action: "Revisar" },
  ];

  return (
    <>
      <PageHead
        title="Dataset extendido"
        sub="Une lo real y lo sintético aprobados en formato YOLO. Las sintéticas entran solo a entrenamiento; validación y prueba quedan solo con imágenes reales."
        actions={<button className="btn btn-primary" onClick={() => setCreating(true)}><PackagePlus size={16} /> Nueva construcción</button>}
      />
      {builds?.length ? (
        <div className="stack">{builds.map((b) => <BuildRow key={b.id} build={b} />)}</div>
      ) : (
        <div className="grid grid-2">
          <Empty title="Aún no construyes ningún dataset" action={<button className="btn btn-primary" onClick={() => setCreating(true)}>Construir el primero</button>}>
            Obtendrás un ZIP listo para YOLO, con detección, segmentación o clasificación por separado, su data.yaml y un manifiesto que registra el origen de cada imagen. Se abre y se valida antes de dártelo.
          </Empty>
          <Card title="Antes de construir"><Requirements items={requirements} /></Card>
        </div>
      )}
      {creating && (
        <Drawer
          title="Nueva construcción"
          onClose={() => setCreating(false)}
          footer={
            <button className="btn btn-primary btn-block btn-large" disabled={busy || !name.trim() || !tasks.length || approvedReal === 0} onClick={async () => {
              await run(() => post(`/projects/${projectId}/builds`, { name, n_synthetic: nSynth === "" ? null : nSynth, seed, tasks }), "Construcción en cola");
              setCreating(false);
              refresh();
            }}>Construir y validar</button>
          }
        >
          <Requirements items={requirements} />
          <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} maxLength={60} /></Field>
          <Field label="Sintéticas a incluir" hint={`Hay ${approvedSynth} aprobadas. Vacío incluye todas.`}>
            <input type="number" min={0} max={approvedSynth} value={nSynth} onChange={(e) => setNSynth(e.target.value === "" ? "" : +e.target.value)} />
          </Field>
          <Field label="Semilla de selección" help="Decide cuáles sintéticas entran cuando no se usan todas. La misma semilla repite la misma selección.">
            <input type="number" min={0} value={seed} onChange={(e) => setSeed(+e.target.value)} />
          </Field>
          <div className="field">
            <span>Tareas a exportar</span>
            {Object.entries(TASK_LABEL).map(([t, label]) => <Check key={t} checked={tasks.includes(t)} onChange={() => toggle(t)}>{label}</Check>)}
          </div>
        </Drawer>
      )}
    </>
  );
}

function BuildRow({ build }: { build: any }) {
  const s = build.summary;
  const active = build.job_status === "queued" || build.job_status === "running";
  return (
    <div className="card">
      <div className="row-between" style={{ marginBottom: 8 }}>
        <div>
          <strong>{build.name}</strong>
          <div className="faint">{formatDate(build.created_at)}{s ? `, ${formatBytes(s.size_bytes)}` : ""}</div>
        </div>
        <div className="row">
          <JobPill status={build.job_status} />
          {build.status === "ready" && <a className="btn btn-primary" href={`/api/builds/${build.id}/download`}><Download size={15} /> Descargar ZIP</a>}
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
                <tr key={split}><td>{splitLabel(split)}</td><td className="num">{c.real}</td><td className="num">{c.synthetic}</td><td className="num">{task === "classify" ? "—" : c.instances}</td><td className="num">{c.skipped}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
