import { Link } from "react-router-dom";
import { ImageRow, Pipeline, get } from "../lib/api";
import { usePoll } from "../lib/hooks";

export interface Stage {
  path: string;
  label: string;
  kinds: string[];
  done: (p: Pipeline) => boolean;
  count?: (p: Pipeline) => number;
  value: (p: Pipeline) => string;
}

const pendingReview = (p: Pipeline) =>
  (p.by_status.real.needs_review ?? 0) + (p.by_status.synthetic.needs_review ?? 0) +
  (p.by_status.real.candidate ?? 0) + (p.by_status.synthetic.candidate ?? 0);

export const STAGES: Stage[] = [
  { path: "dataset", label: "Dataset", kinds: ["ingest", "caption", "features"], done: (p) => p.split_locked, count: (p) => p.real_images, value: (p) => `${p.real_images} reales` },
  { path: "lora", label: "LoRA", kinds: ["train_lora", "lora_eval"], done: (p) => p.loras_ready > 0, count: (p) => p.loras_ready, value: (p) => (p.loras_ready ? `${p.loras_ready} listo${p.loras_ready === 1 ? "" : "s"}` : "Sin entrenar") },
  { path: "generate", label: "Generación", kinds: ["generate"], done: (p) => p.synthetic_images > 0, count: (p) => p.synthetic_images, value: (p) => `${p.synthetic_images} sintéticas` },
  { path: "review", label: "Etiquetas", kinds: ["label"], done: (p) => (p.by_status.synthetic.approved ?? 0) > 0, count: pendingReview, value: (p) => `${pendingReview(p)} por revisar` },
  { path: "quality", label: "Calidad", kinds: ["quality"], done: (p) => p.quality_reports > 0, value: (p) => (p.quality_reports ? "Evaluada" : "Sin evaluar") },
  { path: "build", label: "Dataset extendido", kinds: ["build"], done: (p) => p.builds_ready > 0, count: (p) => p.builds_ready, value: (p) => (p.builds_ready ? `${p.builds_ready} exportado${p.builds_ready === 1 ? "" : "s"}` : "Sin construir") },
  { path: "experiment", label: "Experimento", kinds: ["experiment"], done: (p) => p.experiments_ready > 0, count: (p) => p.experiments_ready, value: (p) => (p.experiments_ready ? `${p.experiments_ready} completo${p.experiments_ready === 1 ? "" : "s"}` : "Sin correr") },
];

/** Ultima etapa consecutiva completada. La linea del riel se llena hasta ahi. */
export function flowProgress(p: Pipeline): number {
  let done = 0;
  for (const stage of STAGES) {
    if (!stage.done(p)) break;
    done += 1;
  }
  return done;
}

/** Mapa del proyecto. Los conectores solo se mueven si hay un trabajo corriendo en esa etapa. */
export function FlowMap({ pipeline, runningKinds }: { pipeline: Pipeline; runningKinds: string[] }) {
  return (
    <div className="flowmap" role="list" aria-label="Avance del proyecto">
      {STAGES.map((stage, i) => {
        const done = stage.done(pipeline);
        const running = stage.kinds.some((k) => runningKinds.includes(k));
        return (
          <Link key={stage.path} role="listitem" to={stage.path} className={`flow-node${done ? " done" : ""}${running ? " running" : ""}`}>
            <span className="flow-node-top">
              <span className="flow-node-dot" aria-hidden>{done ? "✓" : i + 1}</span>
              <span className="flow-wire" aria-hidden />
            </span>
            <span className="flow-node-label">{stage.label}</span>
            <span className="flow-node-value">{running ? "En curso" : stage.value(pipeline)}</span>
          </Link>
        );
      })}
    </div>
  );
}

/** Imagenes reales del proyecto; real y sintetico se distinguen por el color inferior. */
export function Mosaic({ projectId, count = 12 }: { projectId: string; count?: number }) {
  const half = Math.ceil(count / 2);
  const { data } = usePoll(async () => {
    const [synth, real] = await Promise.all([
      get<{ items: ImageRow[] }>(`/projects/${projectId}/images?origin=synthetic&sort=fidelity&limit=${half}`),
      get<{ items: ImageRow[] }>(`/projects/${projectId}/images?origin=real&limit=${count}`),
    ]);
    const takeReal = real.items.slice(0, count - synth.items.length);
    const merged: ImageRow[] = [];
    for (let i = 0; i < Math.max(synth.items.length, takeReal.length); i++) {
      if (takeReal[i]) merged.push(takeReal[i]);
      if (synth.items[i]) merged.push(synth.items[i]);
    }
    return merged.slice(0, count);
  }, 15000, [projectId, count]);
  if (!data?.length) return null;
  return (
    <div className="mosaic" aria-hidden>
      {data.map((image, i) => (
        <div key={image.id} className={`mosaic-cell is-${image.origin}`} style={{ animationDelay: `${i * 35}ms` }}>
          <img src={image.thumb_url} alt="" loading="lazy" />
        </div>
      ))}
    </div>
  );
}

/** Explica el recorrido en la bienvenida. No representa ningun trabajo ni muestra cifras. */
export function FlowIntro() {
  const nodes = [
    { x: 8, label: "Tus fotos" },
    { x: 142, label: "LoRA" },
    { x: 276, label: "Imágenes nuevas" },
    { x: 410, label: "Dataset" },
  ];
  return (
    <svg className="intro-svg" viewBox="0 0 520 170" role="img" aria-label="Tus fotos entrenan un LoRA, que genera imágenes nuevas etiquetadas, y juntas forman un dataset más grande">
      <path className="intro-wire" d="M 94 66 H 410" />
      <path className="intro-pulse" d="M 94 66 H 410" />
      {nodes.map((n) => (
        <g key={n.label}>
          <rect className="intro-card" x={n.x} y={22} width={92} height={88} rx={10} />
          <text x={n.x + 46} y={136} textAnchor="middle">{n.label}</text>
        </g>
      ))}
      {/* Sujeto simplificado: el mismo en todas las etapas. */}
      <g fill="#4FD1BE"><circle cx={54} cy={54} r={13} /><path d="M 30 96 a 24 22 0 0 1 48 0 z" /></g>
      <g fill="none" stroke="#C9D3EA" strokeWidth={2}><circle cx={188} cy={66} r={22} /><path d="M 166 66 h 44 M 188 44 v 44" /></g>
      <g fill="#F7A64D"><circle cx={322} cy={54} r={13} /><path d="M 298 96 a 24 22 0 0 1 48 0 z" /></g>
      <rect className="intro-box" x={291} y={34} width={62} height={68} rx={3} />
      <g>
        {[0, 1, 2].map((r) => [0, 1, 2].map((c) => (
          <rect key={`${r}${c}`} x={424 + c * 22} y={36 + r * 22} width={18} height={18} rx={3} fill={(r + c) % 2 ? "#F7A64D" : "#4FD1BE"} />
        )))}
      </g>
    </svg>
  );
}
