import { useEffect, useMemo, useRef } from "react";
import { BrowserRouter, Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom";
import { ChevronRight, Cpu, FolderKanban, LayoutDashboard, Moon, Sun } from "lucide-react";
import { Job, Project, SystemStatus, get, post } from "./lib/api";
import { formatDuration, usePoll, useStored } from "./lib/hooks";
import { STAGES, StageIcon, flowProgress } from "./ui/flow";
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

function useTheme() {
  const [theme, setTheme] = useStored<Theme>("kosei_theme_v3", "dark");
  useEffect(() => { document.documentElement.setAttribute("data-theme", theme); }, [theme]);
  return { theme, setTheme };
}

function ThemeButton({ theme, setTheme }: { theme: Theme; setTheme: (t: Theme) => void }) {
  const next: Theme = theme === "dark" ? "light" : "dark";
  const toggle = (button: HTMLElement) => {
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
    <button className="icon-btn" aria-label={`Cambiar a tema ${next === "dark" ? "oscuro" : "claro"}`} title={`Tema ${next === "dark" ? "oscuro" : "claro"}`} onClick={(e) => toggle(e.currentTarget)}>
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}

function Rail({ projects, system }: { projects: Project[] | null; system: SystemStatus | null }) {
  const location = useLocation();
  const navigate = useNavigate();
  const projectId = location.pathname.match(/^\/p\/([0-9a-f]{16})/)?.[1];
  const project = projects?.find((p) => p.id === projectId);
  const reached = project ? flowProgress(project.pipeline) : 0;
  const jobs = system?.active_jobs ?? [];

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

      <nav className="rail-nav" aria-label="Navegación">
        {project ? (
          <>
            <NavLink end to={`/p/${project.id}`} className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
              <LayoutDashboard size={16} /> Resumen
            </NavLink>
            <div className="rail-title">Flujo de trabajo</div>
            <div className="flow">
              <span className="flow-track" aria-hidden />
              <span className="flow-fill" aria-hidden style={{ height: reached > 1 ? `calc((100% - 40px) * ${(reached - 1) / (STAGES.length - 1)})` : 0 }} />
              {STAGES.map((stage) => {
                const running = jobs.some((j) => j.project_id === project.id && stage.kinds.includes(j.kind));
                const count = stage.count?.(project.pipeline);
                return (
                  <NavLink key={stage.path} to={`/p/${project.id}/${stage.path}`}
                    className={({ isActive }) => `stage${isActive ? " active" : ""}${stage.done(project.pipeline) ? " done" : ""}${running ? " running" : ""}`}>
                    <span className="stage-node" aria-hidden><StageIcon path={stage.path} /></span>
                    <span>{stage.label}</span>
                    {count ? <span className="stage-count">{count}</span> : <span />}
                  </NavLink>
                );
              })}
            </div>
          </>
        ) : (
          <NavLink end to="/" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
            <FolderKanban size={16} /> Proyectos
          </NavLink>
        )}
      </nav>

      <div className="rail-foot">
        <NavLink to="/system" className={({ isActive }) => `rail-link${isActive ? " active" : ""}`}>
          <Cpu size={16} /> <span className="grow">Sistema</span>
        </NavLink>
      </div>
    </aside>
  );
}

const PAGE_LABEL: Record<string, string> = Object.fromEntries(STAGES.map((s) => [s.path, s.label]));

function Topbar({ projects, system, theme, setTheme }: { projects: Project[] | null; system: SystemStatus | null; theme: Theme; setTheme: (t: Theme) => void }) {
  const location = useLocation();
  const match = location.pathname.match(/^\/p\/([0-9a-f]{16})(?:\/([a-z]+))?/);
  const project = match ? projects?.find((p) => p.id === match[1]) : undefined;
  const section = match?.[2];
  const gpu = system?.gpu;
  const ready = system?.engine.ready;
  return (
    <header className="topbar">
      <nav className="crumbs" aria-label="Ruta">
        <Link to="/">Proyectos</Link>
        {location.pathname === "/system" && <><ChevronRight size={14} /><span className="here">Sistema</span></>}
        {project && (
          <>
            <ChevronRight size={14} />
            {section ? <Link to={`/p/${project.id}`}>{project.name}</Link> : <span className="here">{project.name}</span>}
            {section && <><ChevronRight size={14} /><span className="here">{PAGE_LABEL[section] ?? section}</span></>}
          </>
        )}
      </nav>
      <div className="top-actions">
        <Link to="/system" className="chip" title="Estado del motor de generación">
          <span className={`dot ${ready ? "dot-good" : "dot-warn"}`} aria-hidden />
          {ready ? "Motor listo" : "Motor cargando"}
        </Link>
        {gpu && (
          <Link to="/system" className="chip" title={gpu.name}>
            GPU <span className="mono">{(gpu.memory_used_mib / 1024).toFixed(1)}/{(gpu.memory_total_mib / 1024).toFixed(0)} GB</span>
          </Link>
        )}
        <ThemeButton theme={theme} setTheme={setTheme} />
      </div>
    </header>
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
  const { theme, setTheme } = useTheme();
  const location = useLocation();
  const { data: raw, refresh } = usePoll(() => get<SystemStatus>("/system"), 2500);
  // Un solo sondeo lento no debe anunciar que el motor se cayo: se exigen dos seguidos.
  const misses = useRef(0);
  const system = useMemo(() => {
    if (!raw) return raw;
    misses.current = raw.engine.ready ? 0 : misses.current + 1;
    return misses.current === 1 ? { ...raw, engine: { ...raw.engine, ready: true } } : raw;
  }, [raw]);
  const { data: projects } = usePoll(() => get<Project[]>("/projects"), 5000, [location.pathname]);
  const jobs = useMemo(() => system?.active_jobs ?? [], [system]);
  useEffect(() => { document.querySelector(".main")?.scrollTo({ top: 0 }); }, [location.pathname]);
  return (
    <div className="shell">
      <Rail projects={projects} system={system} />
      <div className="main">
        <Topbar projects={projects} system={system} theme={theme} setTheme={setTheme} />
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
