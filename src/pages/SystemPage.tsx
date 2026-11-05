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
      <PageHead title="Sistema" sub="Estado real del motor de generación, la GPU, los modelos y todos los trabajos." />
      <div className="grid grid-2">
        <Card title="Motor de generación" note={system.engine.ready ? "Stable Diffusion WebUI Forge" : "Aún no responde; tarda unos minutos al arrancar."}>
          <div className="stack">
            <span className={`pill ${system.engine.ready ? "pill-good" : "pill-warn"}`} style={{ width: "fit-content" }}>{system.engine.ready ? "Listo" : "Cargando"}</span>
            {system.engine.checkpoint && <div className="muted">Modelo activo: {system.engine.checkpoint}</div>}
          </div>
        </Card>
        <Card title="GPU" note={gpu?.name ?? "No se detecta una GPU NVIDIA en el contenedor"}>
          {gpu ? (
            <div className="gauge-row">
              <Meter label="Memoria de video" value={gpu.memory_used_mib} total={gpu.memory_total_mib} text={`${(gpu.memory_used_mib / 1024).toFixed(1)} de ${(gpu.memory_total_mib / 1024).toFixed(0)} GB`} />
              <Meter label="Uso" value={gpu.utilization_pct} total={100} text={`${gpu.utilization_pct} %`} warm />
              <Meter label="Temperatura" value={gpu.temperature_c} total={95} text={`${gpu.temperature_c} °C`} warm />
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
          <Meter label="Espacio usado" value={system.disk.total_bytes - system.disk.free_bytes} total={system.disk.total_bytes} text={`${formatBytes(system.disk.free_bytes)} libres de ${formatBytes(system.disk.total_bytes)}`} />
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

function Meter({ label, value, total, text, warm }: { label: string; value: number; total: number; text: string; warm?: boolean }) {
  const pct = Math.max(0, Math.min(100, (value / Math.max(total, 1)) * 100));
  return (
    <div className="meter">
      <div className="row-between"><span className="muted" style={{ fontSize: 13 }}>{label}</span><span className="mono" style={{ fontSize: 13 }}>{text}</span></div>
      <div className={`meter-track${warm ? " warm" : ""}`} role="meter" aria-label={label} aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${pct}%` }} /></div>
    </div>
  );
}
