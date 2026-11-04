import { useEffect, useState } from "react";
import { get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { ConditionDots } from "../ui/charts";
import { Card, Empty, Field, JobPill, Notice, PageHead, Progress, useAction } from "../ui/kit";

interface Condition { name: string; n_synthetic: number; use_real: boolean }

const VERDICT: Record<string, { text: string; tone: "good" | "warn" | "bad" | "info" }> = {
  mejora: { text: "Mejora", tone: "good" },
  empeora: { text: "Empeora", tone: "bad" },
  sin_cambio_concluyente: { text: "Sin cambio concluyente", tone: "info" },
  no_concluyente: { text: "No concluyente", tone: "info" },
  sin_datos: { text: "Sin datos", tone: "info" },
};
const METRIC_LABEL: Record<string, string> = { map50_95: "mAP50-95", map50: "mAP50", precision: "Precisión", recall: "Recall", f1: "F1" };

export default function ExperimentPage({ projectId }: { projectId: string }) {
  const { data: experiments, refresh } = usePoll(() => get<any[]>(`/projects/${projectId}/experiments`), 3000, [projectId]);
  const [name, setName] = useState("Real contra real + sintético");
  const [conditions, setConditions] = useState<Condition[]>([
    { name: "A_real", n_synthetic: 0, use_real: true },
    { name: "B_real_100", n_synthetic: 100, use_real: true },
    { name: "C_real_300", n_synthetic: 300, use_real: true },
    { name: "D_solo_300", n_synthetic: 300, use_real: false },
  ]);
  const [seeds, setSeeds] = useState("0,1,2");
  const [epochs, setEpochs] = useState(100);
  const [imgsz, setImgsz] = useState(640);
  const [batch, setBatch] = useState(16);
  const [preflight, setPreflight] = useState<any>(null);
  const { busy, run } = useAction();
  const seedList = seeds.split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => !Number.isNaN(n));
  const payload = { name, conditions, baseline: conditions[0]?.name, seeds: seedList, epochs, imgsz, batch, model: "yolo11n.yaml" };

  useEffect(() => {
    if (!seedList.length || conditions.length < 2) return;
    const t = setTimeout(() => {
      post<any>(`/projects/${projectId}/experiments/preflight`, payload).then(setPreflight).catch(() => setPreflight(null));
    }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId, JSON.stringify(conditions), seeds]);

  const setCondition = (i: number, patch: Partial<Condition>) => setConditions(conditions.map((c, j) => (j === i ? { ...c, ...patch } : c)));

  return (
    <>
      <PageHead title="Experimento" sub="Misma arquitectura y mismos hiperparámetros; solo cambian los datos de train. Se evalúa siempre en la prueba real bloqueada, sin pesos preentrenados." />
      <div className="grid grid-side">
        <Card title="Nuevo experimento">
          <div className="fields">
            <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} maxLength={80} /></Field>
            <div className="field">
              <span>Condiciones. La primera es la base de comparación.</span>
              <table className="table">
                <thead><tr><th>Nombre</th><th>Sintéticas</th><th>Usa reales</th><th /></tr></thead>
                <tbody>
                  {conditions.map((c, i) => (
                    <tr key={i}>
                      <td><input aria-label="Nombre de la condición" value={c.name} onChange={(e) => setCondition(i, { name: e.target.value.replace(/[^A-Za-z0-9_-]/g, "_") })} /></td>
                      <td><input aria-label="Sintéticas" type="number" min={0} value={c.n_synthetic} onChange={(e) => setCondition(i, { n_synthetic: +e.target.value })} style={{ width: 70 }} /></td>
                      <td><input aria-label="Usa reales" type="checkbox" checked={c.use_real} onChange={(e) => setCondition(i, { use_real: e.target.checked })} /></td>
                      <td><button className="btn btn-small" aria-label="Quitar condición" disabled={conditions.length <= 2} onClick={() => setConditions(conditions.filter((_, j) => j !== i))}>×</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button className="btn btn-small" disabled={conditions.length >= 8} onClick={() => setConditions([...conditions, { name: `cond_${conditions.length + 1}`, n_synthetic: 0, use_real: true }])}>Agregar condición</button>
            </div>
            <Field label="Seeds de entrenamiento" hint="Separadas por coma. Con 3 o más se calcula un intervalo de confianza útil."><input value={seeds} onChange={(e) => setSeeds(e.target.value)} /></Field>
            <div className="fields-3">
              <Field label="Épocas"><input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} /></Field>
              <Field label="Imagen"><input type="number" step={32} min={160} value={imgsz} onChange={(e) => setImgsz(+e.target.value)} /></Field>
              <Field label="Lote"><input type="number" min={1} value={batch} onChange={(e) => setBatch(+e.target.value)} /></Field>
            </div>
            {preflight?.problems?.map((p: string) => <Notice key={p} kind="bad">{p}</Notice>)}
            {preflight?.warnings?.map((w: string) => <Notice key={w} kind="warn">{w}</Notice>)}
            {preflight?.ok && <Notice kind="good" title="Validación previa correcta">{`Train ${preflight.counts.train}, val ${preflight.counts.val}, test ${preflight.counts.test}. ${preflight.synthetic_approved} sintéticas aprobadas.`}</Notice>}
            <button className="btn btn-primary btn-block" disabled={busy || !preflight?.ok || !name.trim()} onClick={() => run(() => post(`/projects/${projectId}/experiments`, payload), "Experimento en cola").then(refresh)}>
              Iniciar {conditions.length * seedList.length} corridas
            </button>
          </div>
        </Card>
        <div className="stack">
          {experiments?.length
            ? experiments.map((e) => <ExperimentResult key={e.id} experiment={e} />)
            : <Empty title="Sin experimentos">El resultado puede ser mejora, ningún cambio o empeoramiento; se reporta lo que salga.</Empty>}
        </div>
      </div>
    </>
  );
}

function ExperimentResult({ experiment: e }: { experiment: any }) {
  const summary = e.summary;
  const active = e.job_status === "running" || e.job_status === "queued";
  const conditions: [string, any][] = summary ? Object.entries(summary.conditions) : [];
  const rows = conditions.map(([name, c]) => ({ name, values: c.map50_95.values, mean: c.map50_95.mean }));
  return (
    <Card title={e.name} note={formatDate(e.created_at)} actions={<JobPill status={e.job_status ?? e.status} />}>
      <div className="stack">
        {active && (
          <div>
            <div className="faint" style={{ marginBottom: 6 }}>{e.job_message}</div>
            <Progress value={e.job_current} total={e.job_total} />
          </div>
        )}
        {e.job_error && <Notice kind="bad" title="El experimento se detuvo">{e.job_error}</Notice>}
        {summary?.comparisons && Object.entries<any>(summary.comparisons).map(([name, c]) => {
          const verdict = VERDICT[c.verdict] ?? VERDICT.sin_datos;
          return (
            <Notice key={name} kind={verdict.tone} title={`${name} frente a ${summary.baseline}: ${verdict.text}`}>
              {c.mean_delta !== undefined && `Diferencia de mAP50-95 ${c.mean_delta >= 0 ? "+" : ""}${c.mean_delta} en ${c.pairs} pares${c.ci95 ? `, IC 95 % [${c.ci95[0]}, ${c.ci95[1]}]` : ""}.`}
            </Notice>
          );
        })}
        {summary?.preflight?.warnings?.map((w: string) => <Notice key={w} kind="warn">{w}</Notice>)}
        {rows.length > 0 && (
          <div>
            <h3 style={{ marginBottom: 6 }}>mAP50-95 en prueba real, por condición</h3>
            <ConditionDots rows={rows} label="mAP50-95 por condición y seed" />
          </div>
        )}
        {conditions.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Condición</th><th className="num">Train real</th><th className="num">Train sint.</th>
                  {Object.values(METRIC_LABEL).map((m) => <th key={m} className="num">{m}</th>)}</tr>
              </thead>
              <tbody>
                {conditions.map(([name, c]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td className="num">{c.counts.train.real}</td>
                    <td className="num">{c.counts.train.synthetic}</td>
                    {Object.keys(METRIC_LABEL).map((m) => (
                      <td key={m} className="num">{c[m].mean.toFixed(3)}{c[m].std != null ? ` ±${c[m].std.toFixed(3)}` : ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {e.runs?.some((r: any) => r.status === "failed") && (
          <Notice kind="bad" title="Corridas fallidas">{e.runs.filter((r: any) => r.status === "failed").map((r: any) => `${r.condition}, seed ${r.seed}: ${r.error}`).join("\n")}</Notice>
        )}
      </div>
    </Card>
  );
}
