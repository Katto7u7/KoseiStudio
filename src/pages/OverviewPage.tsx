import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Job, Project, del, get } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { FlowMap, Mosaic } from "../ui/flow";
import { Card, Confirm, CountUp, JobPill, Notice, SkeletonPage, Stat, jobLabel, useAction } from "../ui/kit";

interface Next { title: string; why: string; to: string; action: string }

function nextStep(p: Project): Next | null {
  const pl = p.pipeline;
  const pendingReal = (pl.by_status.real.candidate ?? 0) + (pl.by_status.real.needs_review ?? 0) + (pl.by_status.real.no_detection ?? 0);
  if (pl.real_images === 0) return { title: "Importa tus imágenes reales", why: "Todo parte de ellas: el modelo aprende de estas fotos.", to: "dataset", action: "Importar imágenes" };
  if (!pl.classes) return { title: "Define qué vas a detectar", why: "Sin una clase, el etiquetado no sabe qué buscar.", to: "dataset", action: "Definir clases" };
  if (!pl.split_locked) return { title: "Separa las imágenes de prueba", why: "Una parte queda fuera del entrenamiento para poder medir sin trampa.", to: "dataset", action: "Definir partición" };
  if (pendingReal > 0) return { title: `Verifica ${pendingReal} imágenes reales`, why: "Son la verdad de referencia del experimento: un programa no puede aprobarlas por ti.", to: "review", action: "Revisar etiquetas" };
  if (pl.loras_ready === 0) return { title: "Entrena tu LoRA", why: "Es lo que enseña al modelo cómo es tu sujeto.", to: "lora", action: "Entrenar" };
  if (pl.synthetic_images === 0) return { title: "Genera tus primeras imágenes", why: "El LoRA ya está listo para crear escenas nuevas.", to: "generate", action: "Generar" };
  if (!pl.by_status.synthetic.approved) return { title: "Elige las sintéticas que sirven", why: "Solo lo que apruebas entra al dataset.", to: "review", action: "Revisar sintéticas" };
  if (!pl.quality_reports) return { title: "Evalúa lo generado", why: "Mide el parecido al sujeto, la diversidad y las repeticiones.", to: "quality", action: "Evaluar" };
  if (!pl.builds_ready) return { title: "Construye el dataset extendido", why: "Une lo real y lo sintético en un ZIP listo para YOLO.", to: "build", action: "Construir" };
  if (!pl.experiments_ready) return { title: "Comprueba si tu detector mejora", why: "Compara real contra real más sintético sobre la prueba real.", to: "experiment", action: "Crear experimento" };
  return null;
}

export default function OverviewPage({ projectId, runningKinds }: { projectId: string; runningKinds: string[] }) {
  const { data: project, error } = usePoll(() => get<Project>(`/projects/${projectId}`), 4000, [projectId]);
  const { data: jobs } = usePoll(() => get<Job[]>(`/jobs?project_id=${projectId}&limit=6`), 4000, [projectId]);
  const [deleting, setDeleting] = useState(false);
  const { run } = useAction();
  const navigate = useNavigate();

  if (error && !project) return <Notice kind="bad" title="No se pudo abrir el proyecto">{error}</Notice>;
  if (!project) return <SkeletonPage />;
  const pl = project.pipeline;
  const next = nextStep(project);
  const approvedSynth = pl.by_status.synthetic.approved ?? 0;
  const approvedReal = pl.by_status.real.approved ?? 0;
  const total = pl.real_images + approvedSynth;
  const pendingAll = (pl.by_status.real.candidate ?? 0) + (pl.by_status.real.needs_review ?? 0) + (pl.by_status.synthetic.candidate ?? 0) + (pl.by_status.synthetic.needs_review ?? 0);

  return (
    <>
      <section className="hero">
        <div className="stack" style={{ gap: 20 }}>
          <div>
            <div className="hero-eyebrow">Proyecto</div>
            <h1 style={{ marginTop: 2 }}>{project.name}</h1>
            {project.description && <p className="page-sub">{project.description}</p>}
          </div>
          <div className="kpis">
            <div className="kpi"><div className="kpi-value"><CountUp value={pl.real_images} /></div><div className="kpi-label"><i style={{ background: "var(--real-fill)" }} />Reales</div></div>
            <div className="kpi"><div className="kpi-value"><CountUp value={pl.synthetic_images} /></div><div className="kpi-label"><i style={{ background: "var(--synth-fill)" }} />Generadas</div></div>
            <div className="kpi"><div className="kpi-value"><CountUp value={approvedReal + approvedSynth} /></div><div className="kpi-label">Aprobadas</div></div>
            <div className="kpi"><div className="kpi-value"><CountUp value={pendingAll} /></div><div className="kpi-label">Por revisar</div></div>
          </div>
          <div className="stack-sm">
            <div className="hero-title">
              {approvedSynth > 0
                ? <>Tu dataset creció de {pl.real_images} a <CountUp value={total} /> imágenes</>
                : pl.real_images > 0 ? "Aún no hay sintéticas aprobadas en tu dataset" : "Tu dataset aún está vacío"}
            </div>
            {pl.real_images > 0 && (
              <>
                <div className="mix" style={{ height: 10 }} role="img" aria-label={`${pl.real_images} reales y ${approvedSynth} sintéticas aprobadas`}>
                  <i style={{ width: `${(pl.real_images / Math.max(total, 1)) * 100}%`, background: "var(--real-fill)" }} />
                  {approvedSynth > 0 && <i style={{ width: `${(approvedSynth / total) * 100}%`, background: "var(--synth-fill)" }} />}
                </div>
                <div className="legend">
                  <span><i style={{ background: "var(--real-fill)" }} />{pl.real_images} reales</span>
                  <span><i style={{ background: "var(--synth-fill)" }} />{approvedSynth} sintéticas aprobadas</span>
                </div>
              </>
            )}
          </div>
        </div>
        <Mosaic projectId={projectId} />
      </section>

      {next ? (
        <div className="next-step">
          <div>
            <div className="eyebrow">Siguiente paso</div>
            <strong>{next.title}</strong>
            <div className="eyebrow" style={{ marginTop: 2 }}>{next.why}</div>
          </div>
          <Link className="btn btn-primary btn-large" to={next.to}>{next.action} <ArrowRight size={16} /></Link>
        </div>
      ) : (
        <div style={{ marginBottom: 16 }}><Notice kind="good" title="Flujo completo">Todas las etapas tienen resultados. Puedes generar más imágenes o repetir el experimento.</Notice></div>
      )}

      <div className="stack">
        <Card title="Avance del proyecto">
          <FlowMap pipeline={pl} runningKinds={runningKinds} />
        </Card>

        <Card title="Partición" note={pl.split_locked ? `Bloqueada. Huella ${project.split_hash?.slice(0, 12)}` : "Aún sin bloquear"}>
          <div className="stat-row">
            <Stat value={pl.by_split.train} label="Entrenamiento" />
            <Stat value={pl.by_split.val} label="Validación" />
            <Stat value={pl.by_split.test} label="Prueba" />
          </div>
        </Card>

        <Card title="Actividad reciente">
          {jobs && jobs.length > 0 ? (
            <div className="table-wrap">
              <table className="table">
                <thead><tr><th>Trabajo</th><th>Estado</th><th>Inicio</th><th>Detalle</th></tr></thead>
                <tbody>
                  {jobs.map((j) => (
                    <tr key={j.id}>
                      <td>{jobLabel(j)}</td>
                      <td><JobPill status={j.status} /></td>
                      <td className="faint">{formatDate(j.started_at ?? j.created_at)}</td>
                      <td className="muted">{j.error?.slice(0, 140) || j.message}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p className="muted">Aquí verás cada trabajo que ejecutes en este proyecto.</p>}
        </Card>

        <div className="row-between" style={{ marginTop: 8 }}>
          <span className="faint">Creado {formatDate(project.created_at)}</span>
          <button className="btn btn-small btn-danger" onClick={() => setDeleting(true)}>Eliminar proyecto</button>
        </div>
      </div>

      {deleting && (
        <Confirm
          danger title="Eliminar proyecto" action="Eliminar definitivamente"
          body={<>Se borrarán de forma permanente las imágenes, etiquetas, exportaciones y experimentos de <strong>{project.name}</strong>. Los LoRAs entrenados se conservan.</>}
          onClose={() => setDeleting(false)}
          onConfirm={async () => {
            setDeleting(false);
            const ok = await run(async () => { await del(`/projects/${projectId}`); return true; }, "Proyecto eliminado");
            if (ok) navigate("/");
          }}
        />
      )}
    </>
  );
}
