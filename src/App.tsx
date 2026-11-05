import { useEffect, useMemo } from "react";
import { BrowserRouter, Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { Job, Pipeline, Project, SystemStatus, get, post } from "./lib/api";
import { usePoll, useStored } from "./lib/hooks";
import { Progress, ToastProvider, jobLabel, useAction } from "./ui/kit";
import BuildPage from "./pages/BuildPage";
import DatasetPage from "./pages/DatasetPage";
import ExperimentPage from "./pages/ExperimentPage";
import GeneratePage from "./pages/GeneratePage";
import LoraPage from "./pages/LoraPage";
import OverviewPage from "./pages/OverviewPage";
import ProjectsPage from "./pages/ProjectsPage";
import QualityPage from "./pages/QualityPage";
import ReviewPage from "./pages/ReviewPage";
import SystemPage from "./pages/SystemPage";

interface Stage { path: string; label: string; kinds: string[]; done: (p: Pipeline) => boolean; count?: (p: Pipeline) => number | string }

export const STAGES: Stage[] = [
  { path: "dataset", label: "Dataset", kinds: ["ingest", "caption", "features"], done: (p) => p.split_locked, count: (p) => p.real_images },
  { path: "lora", label: "LoRA", kinds: ["train_lora", "lora_eval"], done: (p) => p.loras_ready > 0, count: (p) => p.loras_ready },
  { path: "generate", label: "Generación", kinds: ["generate"], done: (p) => p.synthetic_images > 0, count: (p) => p.synthetic_images },
  { path: "review", label: "Etiquetas", kinds: ["label"], done: (p) => (p.by_status.synthetic.approved ?? 0) > 0,
    count: (p) => (p.by_status.real.needs_review ?? 0) + (p.by_status.synthetic.needs_review ?? 0) + (p.by_status.synthetic.candidate ?? 0) + (p.by_status.real.candidate ?? 0) },
  { path: "quality", label: "Calidad", kinds: ["quality"], done: (p) => p.quality_reports > 0 },
  { path: "build", label: "Dataset extendido", kinds: ["build"], done: (p) => p.builds_ready > 0, count: (p) => p.builds_ready },
  { path: "experiment", label: "Experimento", kinds: ["experiment"], done: (p) => p.experiments_ready > 0, count: (p) => p.experiments_ready },
];

function Rail({ system, jobs }: { system: SystemStatus | null; jobs: Job[] }) {
  const location = useLocation();
  const navigate = useNavigate();
  const projectId = location.pathname.match(/^\/p\/([0-9a-f]{16})/)?.[1];
  const { data: projects } = usePoll(() => get<Project[]>("/projects"), 5000, [location.pathname]);
  const project = projects?.find((p) => p.id === projectId);
  const [theme, setTheme] = useStored<"dark" | "light">("kosei_theme", "dark");
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);

  return (
    <aside className="rail">
      <Link to="/" className="brand" aria-label="KoseiStudio, proyectos">
        <span className="brand-mark" aria-hidden />
        <span className="brand-name">KoseiStudio</span>
      </Link>

      {projects && projects.length > 0 && (
        <div className="project-switch">
          <select aria-label="Proyecto activo" value={projectId ?? ""} onChange={(e) => navigate(e.target.value ? `/p/${e.target.value}` : "/")}>
            <option value="">Todos los proyectos</option>
            {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </div>
      )}

      <nav className="stages" aria-label="Etapas del proyecto">
        {project && (
          <>
            <NavLink end to={`/p/${project.id}`} className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>Resumen</NavLink>
            <div className="stages-title">Flujo de trabajo</div>
            {STAGES.map((stage) => {
              const running = jobs.some((j) => j.project_id === project.id && stage.kinds.includes(j.kind));
              const done = stage.done(project.pipeline);
              const count = stage.count?.(project.pipeline);
              return (
                <NavLink key={stage.path} to={`/p/${project.id}/${stage.path}`}
                  className={({ isActive }) => `stage${isActive ? " active" : ""}${done ? " done" : ""}${running ? " running" : ""}`}>
                  <span className="stage-dot" aria-hidden />
                  <span>{stage.label}</span>
                  {count ? <span className="stage-count">{count}</span> : <span />}
                </NavLink>
              );
            })}
          </>
        )}
      </nav>

      <div className="rail-foot">
        <NavLink to="/system" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
          <span>Sistema</span>
          <span className={`pill ${system?.engine.ready ? "pill-good" : "pill-warn"}`}>{system?.engine.ready ? "Motor listo" : "Motor cargando"}</span>
        </NavLink>
        <button className="rail-link" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
          Tema {theme === "dark" ? "claro" : "oscuro"}
        </button>
      </div>
    </aside>
  );
}

function JobDock({ jobs, onChange }: { jobs: Job[]; onChange: () => void }) {
  const { run, busy } = useAction();
  if (jobs.length === 0) return null;
  return (
    <div className="dock" aria-label="Trabajos en curso">
      {jobs.slice(0, 3).map((job) => (
        <div className="dock-row" key={job.id}>
          <strong>{jobLabel(job)}</strong>
          <span className="dock-msg" title={job.message}>{job.status === "queued" ? "En cola, esperando su turno" : job.message}</span>
          <Progress value={job.current} total={job.status === "queued" ? 0 : job.total} />
          <button className="btn btn-small" disabled={busy} onClick={() => run(() => post(`/jobs/${job.id}/cancel`), "Cancelación solicitada").then(onChange)}>Cancelar</button>
        </div>
      ))}
      {jobs.length > 3 && <span className="faint">y {jobs.length - 3} más en cola</span>}
    </div>
  );
}

function Shell() {
  const { data: system, refresh } = usePoll(() => get<SystemStatus>("/system"), 2500);
  const location = useLocation();
  const jobs = useMemo(() => system?.active_jobs ?? [], [system]);
  useEffect(() => { document.querySelector(".main")?.scrollTo(0, 0); }, [location.pathname]);
  return (
    <div className="shell">
      <Rail system={system} jobs={jobs} />
      <div className="main">
        {system && !system.engine.ready && (
          <div className="engine-banner">
            <span className="pill pill-warn">Motor cargando</span>
            El motor de generación tarda unos minutos en arrancar. Mientras tanto puedes usar el resto de la aplicación.
          </div>
        )}
        <main className="page">
          <Routes>
            <Route path="/" element={<ProjectsPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="/p/:projectId/*" element={<ProjectRoutes />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <JobDock jobs={jobs} onChange={refresh} />
      </div>
    </div>
  );
}

function ProjectRoutes() {
  const { projectId = "" } = useParams();
  return (
    <Routes>
      <Route index element={<OverviewPage projectId={projectId} />} />
      <Route path="dataset" element={<DatasetPage projectId={projectId} />} />
      <Route path="lora" element={<LoraPage projectId={projectId} />} />
      <Route path="generate" element={<GeneratePage projectId={projectId} />} />
      <Route path="review" element={<ReviewPage projectId={projectId} />} />
      <Route path="quality" element={<QualityPage projectId={projectId} />} />
      <Route path="build" element={<BuildPage projectId={projectId} />} />
      <Route path="experiment" element={<ExperimentPage projectId={projectId} />} />
    </Routes>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <BrowserRouter>
        <Shell />
      </BrowserRouter>
    </ToastProvider>
  );
}
