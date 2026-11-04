import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Job, Project, del, get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { Card, Confirm, JobPill, Notice, PageHead, Stat, jobLabel, useAction } from "../ui/kit";

function nextStep(p: Project): { text: string; to: string; action: string } | null {
  const pl = p.pipeline;
  const pendingReal = (pl.by_status.real.candidate ?? 0) + (pl.by_status.real.needs_review ?? 0) + (pl.by_status.real.no_detection ?? 0);
  if (pl.real_images === 0) return { text: "Importa tus imágenes reales para empezar.", to: "dataset", action: "Importar imágenes" };
  if (!pl.classes) return { text: "Define al menos una clase para poder etiquetar.", to: "dataset", action: "Definir clases" };
  if (!pl.split_locked) return { text: "Reparte las imágenes en train, val y test, y bloquea la partición.", to: "dataset", action: "Definir partición" };
  if (pendingReal > 0) return { text: `Verifica las etiquetas de ${pendingReal} imágenes reales: son la verdad de referencia.`, to: "review", action: "Revisar etiquetas" };
  if (pl.loras_ready === 0) return { text: "Entrena un LoRA con el split de entrenamiento.", to: "lora", action: "Entrenar LoRA" };
  if (pl.synthetic_images === 0) return { text: "Genera las primeras imágenes sintéticas.", to: "generate", action: "Generar" };
  if (!pl.by_status.synthetic.approved) return { text: "Aprueba las imágenes sintéticas que quieras usar.", to: "review", action: "Revisar sintéticas" };
  if (!pl.quality_reports) return { text: "Evalúa la calidad de lo generado.", to: "quality", action: "Evaluar calidad" };
  if (!pl.builds_ready) return { text: "Construye y exporta el dataset extendido.", to: "build", action: "Construir dataset" };
  if (!pl.experiments_ready) return { text: "Compara real contra real más sintético sobre la prueba real.", to: "experiment", action: "Crear experimento" };
  return null;
}

export default function OverviewPage({ projectId }: { projectId: string }) {
  const { data: project, error } = usePoll(() => get<Project>(`/projects/${projectId}`), 4000, [projectId]);
  const { data: jobs } = usePoll(() => get<Job[]>(`/jobs?project_id=${projectId}&limit=8`), 4000, [projectId]);
  const [deleting, setDeleting] = useState(false);
  const { run } = useAction();
  const navigate = useNavigate();

  if (error) return <Notice kind="bad" title="No se pudo abrir el proyecto">{error}</Notice>;
  if (!project) return null;
  const pl = project.pipeline;
  const next = nextStep(project);
  const approved = (pl.by_status.real.approved ?? 0) + (pl.by_status.synthetic.approved ?? 0);

  return (
    <>
      <PageHead
        title={project.name}
        sub={project.description || undefined}
        actions={<button className="btn btn-danger" onClick={() => setDeleting(true)}>Eliminar proyecto</button>}
      />
      {next && (
        <div className="next-step">
          <div>{next.text}</div>
          <Link className="btn btn-primary" to={next.to}>{next.action}</Link>
        </div>
      )}
      {!next && <div style={{ marginBottom: 16 }}><Notice kind="good" title="Flujo completo">Todas las etapas tienen resultados. Puedes generar más imágenes o repetir el experimento.</Notice></div>}
      <div className="grid grid-2">
        <Card title="Imágenes">
          <div className="stat-row">
            <Stat value={pl.real_images} label="Reales" />
            <Stat value={pl.synthetic_images} label="Sintéticas" />
            <Stat value={approved} label="Aprobadas" />
          </div>
        </Card>
        <Card title="Partición" note={pl.split_locked ? `Bloqueada. Huella ${project.split_hash?.slice(0, 12)}` : "Sin bloquear"}>
          <div className="stat-row">
            <Stat value={pl.by_split.train} label="Train" />
            <Stat value={pl.by_split.val} label="Val" />
            <Stat value={pl.by_split.test} label="Test" />
          </div>
        </Card>
      </div>
      <div style={{ marginTop: 16 }}>
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
          ) : <p className="muted">Aún no se ha ejecutado ningún trabajo en este proyecto.</p>}
        </Card>
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
