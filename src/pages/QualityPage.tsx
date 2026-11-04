import { get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { BarList } from "../ui/charts";
import { Card, Empty, Notice, PageHead, Stat, useAction } from "../ui/kit";

interface ReportRow { id: string; status: string; job_status: string; job_error: string | null; created_at: number; report: any }

function Metric({ name, value, meaning, limit }: { name: string; value: string | number; meaning: string; limit: string }) {
  return (
    <div className="card" style={{ display: "grid", gap: 6 }}>
      <div className="stat-value">{value}</div>
      <h3>{name}</h3>
      <p className="muted" style={{ fontSize: 13 }}>{meaning}</p>
      <p className="faint">Límite: {limit}</p>
    </div>
  );
}

export default function QualityPage({ projectId }: { projectId: string }) {
  const { data: reports, refresh } = usePoll(() => get<ReportRow[]>(`/projects/${projectId}/quality`), 4000, [projectId]);
  const { busy, run } = useAction();
  const running = reports?.find((r) => r.status === "running" && (r.job_status === "running" || r.job_status === "queued"));
  const failed = reports?.[0]?.job_status === "failed" ? reports[0] : null;
  const latest = reports?.find((r) => r.status === "ready");
  const r = latest?.report;

  return (
    <>
      <PageHead
        title="Calidad"
        sub="Compara lo sintético con el split de entrenamiento real, nunca con val ni test. Ninguna métrica prueba por sí sola que una imagen sirva: la prueba final es el experimento."
        actions={<button className="btn btn-primary" disabled={busy || !!running} onClick={() => run(() => post(`/projects/${projectId}/quality`), "Evaluación en cola").then(refresh)}>{running ? "Evaluando..." : "Evaluar ahora"}</button>}
      />
      {failed && <div style={{ marginBottom: 16 }}><Notice kind="bad" title="La última evaluación falló">{failed.job_error}</Notice></div>}
      {!r && !running && !failed && <Empty title="Sin evaluaciones">Genera imágenes y evalúalas para ver estas métricas.</Empty>}
      {r && (
        <div className="stack">
          <div className="row-between">
            <div className="stat-row">
              <Stat value={r.samples.real_train} label="Reales de train" />
              <Stat value={r.samples.synthetic} label="Sintéticas evaluadas" />
            </div>
            <span className="faint">{formatDate(latest!.created_at)}</span>
          </div>
          {r.warnings.map((w: string) => <Notice key={w} kind="warn">{w}</Notice>)}
          <div className="grid grid-3">
            <Metric name="FID" value={r.inception.fid} meaning="Distancia entre las distribuciones de características Inception de reales y sintéticas. Menor es más parecido."
              limit="muy sesgado con menos de 2048 muestras; útil solo para comparar corridas de igual tamaño." />
            <Metric name="KID" value={`${r.inception.kid_mean} ± ${r.inception.kid_std}`} meaning="Parecido a FID, sin sesgo de tamaño de muestra. Menor es más parecido."
              limit={`calculado en subconjuntos de ${r.inception.kid_subset_size} imágenes.`} />
            <Metric name="Precisión k-NN" value={r.precision_recall.precision} meaning="Parte de las sintéticas que cae dentro de la región de las reales. Alta indica fidelidad."
              limit="depende de los embeddings CLIP y de cuántas reales haya." />
            <Metric name="Cobertura k-NN (recall)" value={r.precision_recall.recall} meaning="Parte de las reales cubierta por las sintéticas. Alta indica que no se ignoran variantes."
              limit="alta varianza con pocas imágenes." />
            <Metric name="Diversidad" value={`${r.diversity.synthetic} / ${r.diversity.real}`} meaning="Distancia CLIP media entre pares, sintéticas / reales. Muy baja indica repetición."
              limit="no distingue diversidad útil de ruido." />
            <Metric name="Similitud a lo real" value={r.similarity_to_real.p50} meaning={`Similitud mediana a la real más cercana. ${r.similarity_to_real.memorized} imágenes parecen copias de una real.`}
              limit="alta no significa mejor: puede indicar memorización." />
          </div>
          <div className="grid grid-2">
            <Card title="Coherencia con la clase" note="Similitud CLIP entre la imagen y el nombre de la clase, frente a las reales. Es una señal, no una verificación.">
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Clase</th><th className="num">n</th><th className="num">Sintéticas</th><th className="num">Reales</th></tr></thead>
                  <tbody>
                    {Object.entries<any>(r.per_class).map(([name, row]) => (
                      <tr key={name}><td>{name}</td><td className="num">{row.count}</td><td className="num">{row.clip_consistency.p50}</td><td className="num">{row.real_baseline.p50}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
            <Card title="Repetición y artefactos">
              <div className="stack">
                <div className="stat-row">
                  <Stat value={r.duplicates.near_duplicates} label="Casi duplicadas" />
                  <Stat value={`${(r.duplicates.rate * 100).toFixed(1)} %`} label="Tasa" />
                </div>
                {Object.keys(r.visual.artifact_flags).length
                  ? <BarList rows={Object.entries<number>(r.visual.artifact_flags)} />
                  : <p className="muted">No se marcaron artefactos automáticos.</p>}
              </div>
            </Card>
            <Card title="Comparación visual" note="Mediana de sintéticas frente a reales de train.">
              <table className="table">
                <thead><tr><th>Medida</th><th className="num">Sintéticas</th><th className="num">Reales</th></tr></thead>
                <tbody>
                  <tr><td>Nitidez</td><td className="num">{r.visual.sharpness_synthetic.p50}</td><td className="num">{r.visual.sharpness_real.p50}</td></tr>
                  <tr><td>Brillo</td><td className="num">{r.visual.brightness_synthetic.p50}</td><td className="num">{r.visual.brightness_real.p50}</td></tr>
                  {r.labeling.box_area_synthetic.p50 !== undefined && (
                    <tr><td>Área de caja</td><td className="num">{r.labeling.box_area_synthetic.p50}</td><td className="num">{r.labeling.box_area_real.p50 ?? "—"}</td></tr>
                  )}
                </tbody>
              </table>
            </Card>
            <Card title="Etiquetado" note={r.labeling.no_detection_rate != null ? `${(r.labeling.no_detection_rate * 100).toFixed(1)} % sin detección` : undefined}>
              <BarList rows={Object.entries<number>(r.labeling.statuses)} />
            </Card>
          </div>
          <p className="faint">Inception Score omitido. {r.omitted.inception_score}</p>
        </div>
      )}
    </>
  );
}
