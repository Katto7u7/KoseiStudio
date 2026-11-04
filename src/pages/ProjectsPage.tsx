import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Project, get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { Card, Empty, Field, Notice, PageHead, useAction } from "../ui/kit";

export default function ProjectsPage() {
  const { data: projects, error, loading } = usePoll(() => get<Project[]>("/projects"), 5000);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const { busy, run } = useAction();
  const navigate = useNavigate();

  const create = async (e: FormEvent) => {
    e.preventDefault();
    const project = await run(() => post<Project>("/projects", { name, description }));
    if (project) navigate(`/p/${project.id}`);
  };

  const form = (
    <Card title="Nuevo proyecto">
      <form className="fields" onSubmit={create}>
        <Field label="Nombre">
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Detección de personas" required maxLength={80} autoFocus />
        </Field>
        <Field label="Descripción (opcional)">
          <textarea rows={2} value={description} onChange={(e) => setDescription(e.target.value)} maxLength={1000} />
        </Field>
        <div className="row">
          <button className="btn btn-primary" disabled={busy || !name.trim()}>Crear proyecto</button>
          {projects && projects.length > 0 && <button type="button" className="btn" onClick={() => setCreating(false)}>Cancelar</button>}
        </div>
      </form>
    </Card>
  );

  return (
    <>
      <PageHead
        title="Proyectos"
        sub="Cada proyecto lleva un dataset real hasta un dataset extendido, etiquetado y comparado contra el original."
        actions={projects && projects.length > 0 && !creating ? <button className="btn btn-primary" onClick={() => setCreating(true)}>Nuevo proyecto</button> : undefined}
      />
      {error && <Notice kind="bad" title="No se pudo cargar la lista">{error}</Notice>}
      {creating && <div style={{ marginBottom: 16 }}>{form}</div>}
      {!loading && projects?.length === 0 && !creating && (
        <div className="stack">
          <Empty title="Crea tu primer proyecto">Después podrás importar tus imágenes reales y reanudar el trabajo cuando quieras.</Empty>
          {form}
        </div>
      )}
      <div className="project-grid">
        {projects?.map((p) => {
          const real = p.pipeline.real_images, synth = p.pipeline.synthetic_images;
          const total = real + synth || 1;
          return (
            <Link key={p.id} to={`/p/${p.id}`} className="card project-card">
              <h2>{p.name}</h2>
              {p.description && <p className="muted">{p.description}</p>}
              <div className="mix" aria-label={`${real} reales, ${synth} sintéticas`}>
                {real > 0 && <i style={{ width: `${(real / total) * 100}%`, background: "var(--viz-real)", borderRadius: synth ? "3px 0 0 3px" : 3 }} />}
                {synth > 0 && <i style={{ width: `${(synth / total) * 100}%`, background: "var(--viz-synth)", borderRadius: real ? "0 3px 3px 0" : 3 }} />}
                {real + synth === 0 && <i style={{ width: "100%", background: "var(--raised)", borderRadius: 3 }} />}
              </div>
              <div className="row-between">
                <div className="legend">
                  <span><i style={{ background: "var(--viz-real)" }} />{real} reales</span>
                  <span><i style={{ background: "var(--viz-synth)" }} />{synth} sintéticas</span>
                </div>
                <span className="faint">{formatDate(p.updated_at)}</span>
              </div>
            </Link>
          );
        })}
      </div>
    </>
  );
}
