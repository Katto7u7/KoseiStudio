/* Cliente de la API. Todo error llega como ApiError con el mensaje del backend. */
export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Error ${res.status}`;
    const type = res.headers.get("content-type") ?? "";
    if (type.includes("json")) {
      const body = await res.json();
      if (typeof body?.detail === "string") message = body.detail;
      else if (Array.isArray(body?.detail)) {
        message = body.detail.map((d: { loc?: string[]; msg: string }) => `${d.loc?.slice(1).join(".")}: ${d.msg}`).join("; ");
      }
    } else if (res.status === 413) {
      message = "El archivo supera el limite de carga";
    } else if (res.status === 502 || res.status === 504) {
      message = "El servidor aun esta arrancando. Espera unos segundos y reintenta.";
    }
    throw new ApiError(message, res.status);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const get = <T,>(path: string) => fetch(`/api${path}`).then((r) => parse<T>(r));

export const post = <T,>(path: string, body?: unknown) =>
  fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  }).then((r) => parse<T>(r));

export const put = <T,>(path: string, body: unknown) =>
  fetch(`/api${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => parse<T>(r));

export const patch = <T,>(path: string, body: unknown) =>
  fetch(`/api${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => parse<T>(r));

export const del = (path: string) => fetch(`/api${path}`, { method: "DELETE" }).then((r) => parse<void>(r));

export const upload = <T,>(path: string, files: File[], field = "files") => {
  const form = new FormData();
  files.forEach((f) => form.append(field, f));
  return fetch(`/api${path}`, { method: "POST", body: form }).then((r) => parse<T>(r));
};

export type Origin = "real" | "synthetic";
export type ImageStatus = "pending" | "candidate" | "needs_review" | "no_detection" | "approved" | "rejected";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";

export interface Klass { id: number; name: string; idx: number; detector: string; detector_prompt: string }

export interface Pipeline {
  real_images: number;
  synthetic_images: number;
  by_status: { real: Record<string, number>; synthetic: Record<string, number> };
  by_split: { train: number; val: number; test: number; none: number };
  split_locked: boolean;
  train_captions: number;
  classes: number;
  loras_ready: number;
  quality_reports: number;
  builds_ready: number;
  experiments_ready: number;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  split_config: { train: number; val: number; test: number; seed: number } | null;
  split_locked_at: number | null;
  split_hash: string | null;
  created_at: number;
  updated_at: number;
  pipeline: Pipeline;
  classes?: Klass[];
}

export interface Job {
  id: string;
  project_id: string | null;
  kind: string;
  status: JobStatus;
  current: number;
  total: number;
  message: string;
  error: string | null;
  result: any;
  params: any;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface ImageRow {
  id: string;
  origin: Origin;
  filename: string;
  width: number;
  height: number;
  format: string;
  split: "train" | "val" | "test" | null;
  status: ImageStatus;
  class_id: number | null;
  class_name?: string | null;
  class_source: string | null;
  caption: string | null;
  sha256: string;
  generation: any;
  qc: any;
  annotation_count?: number;
  top_confidence?: number | null;
  thumb_url: string;
  file_url: string;
}

export interface Annotation {
  id: number;
  class_id: number;
  class_name: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  polygon: number[] | null;
  confidence: number | null;
  source: string;
  mask_source: string | null;
  verified: number;
}

export interface EngineModel {
  title: string;
  name: string;
  arch: "sd15" | "sdxl" | "flux" | "unknown";
  arch_label: string;
  defaults: {
    width: number; height: number; steps: number; cfg_scale: number;
    distilled_cfg_scale?: number; sampler: string; scheduler: string; batch_size: number;
  };
}

export interface Lora {
  id: string;
  project_id: string | null;
  name: string;
  arch: string;
  arch_label: string;
  base_model: string | null;
  token: string | null;
  origin: "trained" | "external";
  status: string;
  config: any;
  job_id: string | null;
  job_status: JobStatus | null;
  train_image_count: number;
  created_at: number;
}

export interface SystemStatus {
  engine: { ready: boolean; checkpoint: string | null };
  gpu: { name: string; memory_total_mib: number; memory_used_mib: number; memory_free_mib: number; utilization_pct: number; temperature_c: number } | null;
  disk: { free_bytes: number; total_bytes: number };
  weights: Record<string, boolean>;
  kohya: boolean;
  active_jobs: Job[];
}

export const JOB_LABEL: Record<string, string> = {
  ingest: "Importación",
  caption: "Descripciones BLIP",
  features: "Características",
  train_lora: "Entrenamiento LoRA",
  generate: "Generación",
  label: "Etiquetado",
  quality: "Evaluación de calidad",
  build: "Construcción de dataset",
  experiment: "Experimento",
};

export const STATUS_LABEL: Record<string, string> = {
  pending: "Pendiente",
  candidate: "Candidata",
  needs_review: "Revisar",
  no_detection: "Sin detección",
  approved: "Aprobada",
  rejected: "Rechazada",
};
