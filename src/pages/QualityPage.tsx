import { Project, get, post } from "../lib/api";
import { formatDate, usePoll } from "../lib/hooks";
import { BarList } from "../ui/charts";
import { Gauge } from "lucide-react";
import { Card, Empty, FidelityScale, Notice, PageHead, Requirements, Stat, useAction } from "../ui/kit";

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
  const { data: project } = usePoll(() => get<Project>(`/projects/${projectId}`), 10000, [projectId]);
  const { busy, run } = useAction();
  const pl = project?.pipeline;
  const labeledSynth = pl ? pl.synthetic_images - (pl.by_status.synthetic.pending ?? 0) - (pl.by_status.synthetic.rejected ?? 0) : 0;
  const requirements = [
    { ok: (pl?.by_split.train ?? 0) >= 2, text: `${pl?.by_split.train ?? 0} reales en entrenamiento (mínimo 2)`, to: `/p/${projectId}/dataset`, action: "Ir a dataset" },
    { ok: labeledSynth >= 2, text: `${labeledSynth} sintéticas etiquetadas (mínimo 2)`, to: `/p/${projectId}/generate`, action: "Generar" },
  ];
  const running = reports?.find((r) => r.status === "running" && (r.job_status === "running" || r.job_status === "queued"));
  const failed = reports?.[0]?.job_status === "failed" ? reports[0] : null;
  const latest = reports?.find((r) => r.status === "ready");
  const r = latest?.report;

  return (
    <>
      <PageHead
        title="Calidad"
        sub="Compara lo sintético con el split de entrenamiento real, nunca con val ni test. Ninguna métrica prueba por sí sola que una imagen sirva: la prueba final es el experimento."
        actions={<button className="btn btn-primary" disabled={busy || !!running || requirements.some((r) => !r.ok)} onClick={() => run(() => post(`/projects/${projectId}/quality`), "Evaluación en cola").then(refresh)}><Gauge size={16} /> {running ? "Evaluando..." : "Evaluar ahora"}</button>}
      />
      {failed && <div style={{ marginBottom: 16 }}><Notice kind="bad" title="La última evaluación falló">{failed.job_error}</Notice></div>}
      {!r && !running && !failed && (
        <div className="grid grid-2">
          <Empty title="Aún no hay evaluaciones">Verás cuánto se parecen tus imágenes al sujeto, si cubren la variedad de lo real, cuántas se repiten y qué artefactos tienen. Cada métrica viene con lo que mide y sus límites.</Empty>
          <Card title="Para evaluar"><Requirements items={requirements} /></Card>
        </div>
      )}
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
          {r.fidelity?.images > 0 && (
            <Card title="Fidelidad al sujeto" note="Identidad facial con SFace y parecido visual con CLIP, contra las imágenes reales de train.">
              <div className="stat-row">
                <Stat value={r.fidelity.same_person_rate != null ? `${(r.fidelity.same_person_rate * 100).toFixed(0)} %` : "—"} label={`Reconocidas como el sujeto (rostro ≥ ${r.fidelity.threshold})`} />
                <Stat value={r.fidelity.face_mean ?? "—"} label="Parecido facial medio" />
                <Stat value={r.fidelity.real_baseline ?? "—"} label="Referencia: entre fotos reales" />
                <Stat value={r.fidelity.clip_mean ?? "—"} label="Parecido visual medio (CLIP)" />
                <Stat value={`${r.fidelity.face_found} / ${r.fidelity.images}`} label="Con rostro detectado" />
              </div>
              {r.fidelity.face_mean != null && <div style={{ marginTop: 14 }}><FidelityScale value={r.fidelity.face_mean} threshold={r.fidelity.threshold} baseline={r.fidelity.real_baseline} /></div>}
              <p className="faint" style={{ marginTop: 10 }}>Límite: el umbral se calibró con una muestra pequeña, y el parecido facial no aplica a objetos. Para objetos usa CLIP junto con la inspección visual.</p>
            </Card>
          )}
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
