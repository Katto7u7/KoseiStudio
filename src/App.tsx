import { useEffect, useMemo, useRef } from "react";
import { BrowserRouter, Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { Job, Project, SystemStatus, get, post } from "./lib/api";
import { formatDuration, usePoll, useStored } from "./lib/hooks";
import { STAGES, flowProgress } from "./ui/flow";
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

type Theme = "light" | "dark";

function ThemeToggle() {
  const [theme, setTheme] = useStored<Theme>("kosei_theme_v2", "light");
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);
  const choose = (next: Theme, button: HTMLElement) => {
    if (next === theme) return;
    const apply = () => {
      document.documentElement.setAttribute("data-theme", next);
      setTheme(next);
    };
    const rect = button.getBoundingClientRect();
    document.documentElement.style.setProperty("--tx", `${rect.left + rect.width / 2}px`);
    document.documentElement.style.setProperty("--ty", `${rect.top + rect.height / 2}px`);
    const doc = document as Document & { startViewTransition?: (cb: () => void) => unknown };
    if (doc.startViewTransition && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) doc.startViewTransition(apply);
    else apply();
  };
  return (
    <div className="theme-toggle" role="group" aria-label="Tema">
      <button aria-pressed={theme === "light"} onClick={(e) => choose("light", e.currentTarget)}>Claro</button>
      <button aria-pressed={theme === "dark"} onClick={(e) => choose("dark", e.currentTarget)}>Oscuro</button>
    </div>
  );
}

function Rail({ system, jobs }: { system: SystemStatus | null; jobs: Job[] }) {
  const location = useLocation();
  const navigate = useNavigate();
  const projectId = location.pathname.match(/^\/p\/([0-9a-f]{16})/)?.[1];
  const { data: projects } = usePoll(() => get<Project[]>("/projects"), 5000, [location.pathname]);
  const project = projects?.find((p) => p.id === projectId);
  const reached = project ? flowProgress(project.pipeline) : 0;
  const ready = system?.engine.ready;

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

      <nav className="rail-nav" aria-label="Etapas del proyecto">
        {project && (
          <>
            <NavLink end to={`/p/${project.id}`} className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>Resumen</NavLink>
            <div className="rail-title">Flujo de trabajo</div>
            <div className="flow">
              <span className="flow-track" aria-hidden />
              <span className="flow-fill" aria-hidden style={{ height: reached > 1 ? `calc((100% - 36px) * ${(reached - 1) / (STAGES.length - 1)})` : 0 }} />
              {STAGES.map((stage) => {
                const running = jobs.some((j) => j.project_id === project.id && stage.kinds.includes(j.kind));
                const count = stage.count?.(project.pipeline);
                return (
                  <NavLink key={stage.path} to={`/p/${project.id}/${stage.path}`}
                    className={({ isActive }) => `stage${isActive ? " active" : ""}${stage.done(project.pipeline) ? " done" : ""}${running ? " running" : ""}`}>
                    <span className="stage-dot" aria-hidden />
                    <span>{stage.label}</span>
                    {count ? <span className="stage-count">{count}</span> : <span />}
                  </NavLink>
                );
              })}
            </div>
          </>
        )}
      </nav>

      <div className="rail-foot">
        <NavLink to="/system" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
          <span>Sistema</span>
          <span className="rail-status"><span className={`dot ${ready ? "dot-good" : "dot-warn"}`} aria-hidden />{ready ? "Motor listo" : "Motor cargando"}</span>
        </NavLink>
        <ThemeToggle />
      </div>
    </aside>
  );
}

/** Tiempo restante a partir del avance que esta sesion observa de verdad. */
function useEtas(jobs: Job[]) {
  const seen = useRef(new Map<string, { t: number; current: number }>());
  const out = new Map<string, number>();
  const now = Date.now() / 1000;
  for (const job of jobs) {
    if (job.status !== "running" || !job.total) continue;
    const first = seen.current.get(job.id);
    if (!first) {
      seen.current.set(job.id, { t: now, current: job.current });
      continue;
    }
    const advanced = job.current - first.current;
    if (advanced >= 2 && now - first.t > 8) out.set(job.id, ((now - first.t) / advanced) * (job.total - job.current));
  }
  return out;
}

function JobDock({ jobs, onChange }: { jobs: Job[]; onChange: () => void }) {
  const { run, busy } = useAction();
  const etas = useEtas(jobs);
  if (jobs.length === 0) return null;
  return (
    <div className="dock" aria-label="Trabajos en curso">
      {jobs.slice(0, 3).map((job) => {
        const eta = etas.get(job.id);
        const pct = job.total ? `${Math.min(100, Math.round((job.current / job.total) * 100))} %` : "";
        return (
          <div className="dock-row" key={job.id}>
            <strong>{jobLabel(job)}</strong>
            <span className="dock-msg" title={job.message}>{job.status === "queued" ? "En cola, esperando su turno" : job.message}</span>
            <Progress value={job.current} total={job.status === "queued" ? 0 : job.total} live />
            <span className="dock-num" title={eta ? `Faltan unos ${formatDuration(eta)}` : undefined}>{eta ? formatDuration(eta) : pct}</span>
            <button className="btn btn-small" disabled={busy} onClick={() => run(() => post(`/jobs/${job.id}/cancel`), "Cancelación solicitada").then(onChange)}>Cancelar</button>
          </div>
        );
      })}
      {jobs.length > 3 && <span className="faint">y {jobs.length - 3} más en cola</span>}
    </div>
  );
}

function Shell() {
  const { data: system, refresh } = usePoll(() => get<SystemStatus>("/system"), 2500);
  const location = useLocation();
  const jobs = useMemo(() => system?.active_jobs ?? [], [system]);
  useEffect(() => { document.querySelector(".main")?.scrollTo({ top: 0 }); }, [location.pathname]);
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
        <main className="page" key={location.pathname}>
          <Routes>
            <Route path="/" element={<ProjectsPage />} />
            <Route path="/system" element={<SystemPage />} />
            <Route path="/p/:projectId/*" element={<ProjectRoutes runningKinds={jobs.map((j) => j.kind)} />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <JobDock jobs={jobs} onChange={refresh} />
      </div>
    </div>
  );
}

function ProjectRoutes({ runningKinds }: { runningKinds: string[] }) {
  const { projectId = "" } = useParams();
  return (
    <Routes>
      <Route index element={<OverviewPage projectId={projectId} runningKinds={runningKinds} />} />
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
