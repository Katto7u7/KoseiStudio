import { useState } from "react";
import { Job, SystemStatus, get, post } from "../lib/api";
import { formatBytes, formatDate, usePoll } from "../lib/hooks";
import { Card, JobPill, Notice, PageHead, Progress, Stat, jobLabel, useAction } from "../ui/kit";

export default function SystemPage() {
  const { data: system, error } = usePoll(() => get<SystemStatus>("/system"), 2500);
  const { data: jobs, refresh } = usePoll(() => get<Job[]>("/jobs?limit=40"), 4000);
  const [logFor, setLogFor] = useState<string | null>(null);
  const { data: log } = usePoll(() => (logFor ? fetch(`/api/jobs/${logFor}/log?tail=200`).then((r) => r.text()) : Promise.resolve("")), logFor ? 4000 : null, [logFor]);
  const { busy, run } = useAction();
  if (error) return <Notice kind="bad" title="No se puede leer el estado del sistema">{error}</Notice>;
  if (!system) return null;
  const gpu = system.gpu;
  const missing = Object.entries(system.weights).filter(([, ok]) => !ok).map(([k]) => k);
  return (
    <>
      <PageHead title="Sistema" sub="Estado real del motor, la GPU, los pesos y todos los trabajos." />
      <div className="grid grid-2">
        <Card title="Motor de generación" note={system.engine.ready ? "Stable Diffusion WebUI Forge" : "Aún no responde; tarda unos minutos al arrancar."}>
          <div className="stack">
            <span className={`pill ${system.engine.ready ? "pill-good" : "pill-warn"}`} style={{ width: "fit-content" }}>{system.engine.ready ? "Listo" : "Cargando"}</span>
            {system.engine.checkpoint && <div className="muted">Modelo activo: {system.engine.checkpoint}</div>}
          </div>
        </Card>
        <Card title="GPU" note={gpu?.name ?? "No se detecta una GPU NVIDIA en el contenedor"}>
          {gpu ? (
            <div className="stack">
              <div className="stat-row">
                <Stat value={`${(gpu.memory_used_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(0)} GB`} label="VRAM en uso" />
                <Stat value={`${gpu.utilization_pct} %`} label="Uso" />
                <Stat value={`${gpu.temperature_c} °C`} label="Temperatura" />
              </div>
              <Progress value={gpu.memory_used_mib} total={gpu.memory_total_mib} />
            </div>
          ) : <Notice kind="warn">Sin GPU no se puede generar ni entrenar.</Notice>}
        </Card>
        <Card title="Pesos de los modelos">
          {missing.length ? <Notice kind="warn" title="Faltan pesos">{missing.join(", ")}. Ejecuta docker compose up -d con internet para descargarlos.</Notice> : <Notice kind="good" title="Todos presentes" />}
          <div className="row" style={{ marginTop: 10 }}>
            {Object.entries(system.weights).map(([k, ok]) => <span key={k} className={`pill ${ok ? "" : "pill-warn"}`}>{k}</span>)}
            <span className={`pill ${system.kohya ? "" : "pill-warn"}`}>kohya</span>
          </div>
        </Card>
        <Card title="Disco">
          <div className="stat-row">
            <Stat value={formatBytes(system.disk.free_bytes)} label="Libre" />
            <Stat value={formatBytes(system.disk.total_bytes)} label="Total" />
          </div>
        </Card>
      </div>
      <div style={{ marginTop: 16 }}>
        <Card title="Trabajos">
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>Trabajo</th><th>Estado</th><th>Inicio</th><th>Detalle</th><th /></tr></thead>
              <tbody>
                {jobs?.map((j) => (
                  <tr key={j.id}>
                    <td>{jobLabel(j)}</td>
                    <td><JobPill status={j.status} /></td>
                    <td className="faint">{formatDate(j.started_at ?? j.created_at)}</td>
                    <td className="muted">{j.error?.slice(0, 160) || j.message}</td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                      <button className="btn btn-small" onClick={() => setLogFor(logFor === j.id ? null : j.id)}>Registro</button>{" "}
                      {["failed", "cancelled", "interrupted"].includes(j.status) && (
                        <button className="btn btn-small" disabled={busy} onClick={() => run(() => post(`/jobs/${j.id}/retry`), "Reintento en cola").then(refresh)}>Reintentar</button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {logFor && <pre className="mono faint pre" style={{ marginTop: 12, fontSize: 11, maxHeight: 320, overflow: "auto" }}>{log || "Este trabajo no tiene registro."}</pre>}
        </Card>
      </div>
    </>
  );
}
