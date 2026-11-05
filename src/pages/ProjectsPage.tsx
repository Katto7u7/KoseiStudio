import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ImageRow, Project, get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { FlowIntro } from "../ui/flow";
import { Drawer, Field, Notice, PageHead, Skeleton, useAction } from "../ui/kit";

function ProjectForm({ onClose }: { onClose?: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const { busy, run } = useAction();
  const navigate = useNavigate();
  const create = async (e: FormEvent) => {
    e.preventDefault();
    const project = await run(() => post<Project>("/projects", { name, description }));
    if (project) navigate(`/p/${project.id}`);
  };
  return (
    <form className="fields" onSubmit={create}>
      <Field label="Nombre del proyecto">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Detección de personas" required maxLength={80} autoFocus />
      </Field>
      <Field label="Descripción (opcional)" hint="Para qué modelo es este dataset o qué problema resuelve.">
        <textarea rows={3} value={description} onChange={(e) => setDescription(e.target.value)} maxLength={1000} />
      </Field>
      <div className="row">
        <button className="btn btn-primary" disabled={busy || !name.trim()}>Crear proyecto</button>
        {onClose && <button type="button" className="btn" onClick={onClose}>Cancelar</button>}
      </div>
    </form>
  );
}

function Cover({ projectId }: { projectId: string }) {
  const { data } = usePoll(() => get<{ items: ImageRow[] }>(`/projects/${projectId}/images?limit=4`), null, [projectId]);
  return (
    <div className="project-cover" aria-hidden>
      {data?.items.map((image) => <img key={image.id} src={image.thumb_url} alt="" loading="lazy" />)}
    </div>
  );
}

function ProjectCard({ project: p }: { project: Project }) {
  const real = p.pipeline.real_images, synth = p.pipeline.synthetic_images;
  const total = real + synth || 1;
  return (
    <Link to={`/p/${p.id}`} className="card project-card">
      <Cover projectId={p.id} />
      <div className="project-card-body">
        <div>
          <h2>{p.name}</h2>
          {p.description && <p className="muted" style={{ fontSize: 13.5, marginTop: 3 }}>{p.description}</p>}
        </div>
        <div className="mix" aria-label={`${real} reales, ${synth} sintéticas`}>
          {real > 0 && <i style={{ width: `${(real / total) * 100}%`, background: "var(--real-fill)" }} />}
          {synth > 0 && <i style={{ width: `${(synth / total) * 100}%`, background: "var(--synth-fill)" }} />}
        </div>
        <div className="row-between">
          <div className="legend">
            <span><i style={{ background: "var(--real-fill)" }} />{real} reales</span>
            <span><i style={{ background: "var(--synth-fill)" }} />{synth} sintéticas</span>
          </div>
          <span className="faint">{formatDate(p.updated_at)}</span>
        </div>
      </div>
    </Link>
  );
}

export default function ProjectsPage() {
  const { data: projects, error, loading } = usePoll(() => get<Project[]>("/projects"), 5000);
  const [creating, setCreating] = useState(false);
  const empty = !loading && projects?.length === 0;

  if (empty) {
    return (
      <>
        <section className="welcome">
          <div>
            <h1>Amplía tu dataset con imágenes de tu propio objeto o persona</h1>
            <p>Entrena un modelo con tus fotos, genera escenas nuevas y etiquétalas. Después comprueba si tu detector mejora, evaluándolo sobre imágenes reales.</p>
          </div>
          <FlowIntro />
        </section>
        <section className="card" style={{ maxWidth: 560 }}>
          <div className="card-head"><div><h2>Crea tu primer proyecto</h2><div className="card-note">Podrás retomarlo cuando quieras: todo se guarda.</div></div></div>
          <ProjectForm />
        </section>
      </>
    );
  }

  return (
    <>
      <PageHead
        title="Proyectos"
        sub="Cada proyecto lleva un dataset real hasta un dataset más grande, etiquetado y comparado contra el original."
        actions={<button className="btn btn-primary" onClick={() => setCreating(true)}>Nuevo proyecto</button>}
      />
      {error && <Notice kind="bad" title="No se pudo cargar la lista">{error}</Notice>}
      {loading && !projects && <div className="project-grid"><Skeleton height={200} /><Skeleton height={200} /></div>}
      <div className="project-grid">
        {projects?.map((p) => <ProjectCard key={p.id} project={p} />)}
      </div>
      {creating && (
        <Drawer title="Nuevo proyecto" onClose={() => setCreating(false)}>
          <ProjectForm onClose={() => setCreating(false)} />
        </Drawer>
      )}
    </>
  );
}
