import { useMemo, useRef, useState } from "react";

/* Graficas SVG propias. Marcas delgadas, un solo eje, ejes discretos y tooltip real. */

export function BarList({ rows, color }: { rows: [string, number][]; color?: string }) {
  const max = Math.max(1, ...rows.map((r) => r[1]));
  return (
    <div className="bars">
      {rows.map(([label, value]) => (
        <div className="bar-row" key={label} title={`${label}: ${value}`}>
          <span className="bar-label">{label}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${(value / max) * 100}%`, background: color }} /></div>
          <span className="bar-value">{value}</span>
        </div>
      ))}
    </div>
  );
}

function niceTicks(min: number, max: number, count = 4): number[] {
  const span = max - min || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = span / count / step;
  const size = step * (err >= 5 ? 10 : err >= 2 ? 5 : err >= 1.5 ? 2 : 1);
  const ticks: number[] = [];
  for (let v = Math.ceil(min / size) * size; v <= max + size * 1e-6; v += size) ticks.push(Number(v.toPrecision(6)));
  return ticks;
}

/** Curva de perdida del entrenamiento. Una sola serie: el titulo de la tarjeta la nombra. */
export function LossChart({ points }: { points: { step: number; loss: number }[] }) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const W = 640, H = 200, L = 46, R = 12, T = 10, B = 24;
  const { xs, ys, ticks, stepMax } = useMemo(() => {
    const stepMax = Math.max(1, ...points.map((p) => p.step));
    const losses = points.map((p) => p.loss);
    const lo = Math.min(...losses), hi = Math.max(...losses);
    const pad = (hi - lo) * 0.1 || hi * 0.1 || 0.01;
    const yMin = Math.max(0, lo - pad), yMax = hi + pad;
    return {
      xs: (s: number) => L + (s / stepMax) * (W - L - R),
      ys: (v: number) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B),
      ticks: niceTicks(yMin, yMax),
      stepMax,
    };
  }, [points]);
  if (points.length < 2) return <p className="faint">La curva aparece al registrarse los primeros pasos.</p>;
  const path = points.map((p, i) => `${i ? "L" : "M"}${xs(p.step).toFixed(1)},${ys(p.loss).toFixed(1)}`).join("");
  const active = hover !== null ? points[hover] : null;
  const onMove = (e: React.MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const px = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0, dist = Infinity;
    points.forEach((p, i) => { const d = Math.abs(xs(p.step) - px); if (d < dist) { dist = d; best = i; } });
    setHover(best);
  };
  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Pérdida por paso, última ${points[points.length - 1].loss.toFixed(4)}`} onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} />
            <text x={L - 6} y={ys(t) + 4} textAnchor="end">{t.toFixed(3)}</text>
          </g>
        ))}
        <text x={L} y={H - 6}>0</text>
        <text x={W - R} y={H - 6} textAnchor="end">{stepMax} pasos</text>
        <path d={path} fill="none" stroke="var(--paper)" strokeWidth="2" strokeLinejoin="round" />
        {active && (
          <g>
            <line x1={xs(active.step)} x2={xs(active.step)} y1={T} y2={H - B} stroke="var(--line-strong)" />
            <circle cx={xs(active.step)} cy={ys(active.loss)} r="4" fill="var(--paper)" stroke="var(--panel)" strokeWidth="2" />
          </g>
        )}
      </svg>
      {active && (
        <div className="chart-tip" style={{ left: `${(xs(active.step) / W) * 100}%`, top: `${(ys(active.loss) / H) * 100}%` }}>
          paso {active.step}: {active.loss.toFixed(4)}
        </div>
      )}
    </div>
  );
}

/** Mapa PCA de embeddings CLIP. Real y sintetico difieren tambien en la forma de la marca. */
export function FeatureMap({ points }: { points: { id: string; x: number; y: number; origin: string; split: string | null }[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const S = 420, P = 14;
  const sx = (v: number) => P + ((v + 1) / 2) * (S - 2 * P);
  const sy = (v: number) => P + ((1 - v) / 2) * (S - 2 * P);
  const active = hover !== null ? points[hover] : null;
  return (
    <div>
      <div className="legend" style={{ marginBottom: 8 }}>
        <span><i style={{ background: "var(--viz-real)", borderRadius: "50%" }} />Real ({points.filter((p) => p.origin === "real").length})</span>
        <span><i style={{ background: "var(--viz-synth)" }} />Sintética ({points.filter((p) => p.origin !== "real").length})</span>
      </div>
      <div style={{ position: "relative", maxWidth: 520 }}>
        <svg className="chart" viewBox={`0 0 ${S} ${S}`} width="100%" role="img" aria-label="Mapa de similitud visual de las imágenes" style={{ background: "var(--carbon)", borderRadius: 6 }}>
          {points.map((p, i) => p.origin === "real"
            ? <circle key={p.id} cx={sx(p.x)} cy={sy(p.y)} r={5} fill="var(--viz-real)" stroke="var(--carbon)" strokeWidth={1.5} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
            : <rect key={p.id} x={sx(p.x) - 4.5} y={sy(p.y) - 4.5} width={9} height={9} fill="var(--viz-synth)" stroke="var(--carbon)" strokeWidth={1.5} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />)}
        </svg>
        {active && (
          <div className="chart-tip" style={{ left: `${(sx(active.x) / S) * 100}%`, top: `${(sy(active.y) / S) * 100}%` }}>
            {active.origin === "real" ? `real, ${active.split ?? "sin split"}` : "sintética"}
          </div>
        )}
      </div>
    </div>
  );
}

/** Metrica por condicion: cada punto es una seed y la barra vertical marca la media. */
export function ConditionDots({ rows, label }: { rows: { name: string; values: number[]; mean: number }[]; label: string }) {
  const [hover, setHover] = useState<string | null>(null);
  const W = 640, RowH = 34, L = 110, R = 62, T = 22;
  const H = T + rows.length * RowH + 6;
  const all = rows.flatMap((r) => r.values);
  const hi = Math.max(0.01, ...all);
  const xMax = Math.min(1, hi * 1.15);
  const sx = (v: number) => L + (v / xMax) * (W - L - R);
  const ticks = niceTicks(0, xMax, 4);
  return (
    <div style={{ position: "relative" }}>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={label}>
        {ticks.map((t) => (
          <g key={t}>
            <line className="gridline" x1={sx(t)} x2={sx(t)} y1={T - 4} y2={H - 4} />
            <text x={sx(t)} y={12} textAnchor="middle">{t.toFixed(2)}</text>
          </g>
        ))}
        {rows.map((row, i) => {
          const y = T + i * RowH + RowH / 2;
          return (
            <g key={row.name}>
              <text x={L - 12} y={y + 4} textAnchor="end" style={{ fill: "var(--paper)" }}>{row.name}</text>
              <line x1={sx(row.mean)} x2={sx(row.mean)} y1={y - 11} y2={y + 11} stroke="var(--paper)" strokeWidth="2" />
              {row.values.map((v, j) => (
                <circle key={j} cx={sx(v)} cy={y} r={5} fill="var(--mist)" stroke="var(--panel)" strokeWidth="2"
                  onMouseEnter={() => setHover(`${row.name}, seed ${j + 1}: ${v.toFixed(4)}`)} onMouseLeave={() => setHover(null)} />
              ))}
              <text x={W - R + 8} y={y + 4} style={{ fill: "var(--paper)" }}>{row.mean.toFixed(3)}</text>
            </g>
          );
        })}
      </svg>
      <div className="faint" style={{ minHeight: 18 }}>{hover ?? "Punto: una seed. Línea: media."}</div>
    </div>
  );
}
