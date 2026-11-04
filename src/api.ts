import type {
  DatasetEntry,
  GenerateRequest,
  GenerateResponse,
  HealthResponse,
  OutputEntry,
  TrainingCaptionResponse,
  TrainingStartRequest,
  TrainingStatus,
  TrainingUploadResponse,
} from "./types";

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const data = await response.json();
      const message =
        typeof data?.detail === "string" ? data.detail : JSON.stringify(data ?? {});
      throw new Error(message || "Request failed");
    }
    const text = await response.text();
    throw new Error(text || "Request failed");
  }
  return response.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return handleResponse<HealthResponse>(await fetch("/api/health"));
}

export async function fetchModels(): Promise<string[]> {
  const data = await handleResponse<{ models: string[] }>(await fetch("/api/models"));
  return data.models ?? [];
}

export async function fetchLoras(): Promise<string[]> {
  const data = await handleResponse<{ loras: string[] }>(await fetch("/api/loras"));
  return data.loras ?? [];
}

export async function fetchSamplers(): Promise<string[]> {
  const data = await handleResponse<{ samplers: string[] }>(await fetch("/api/samplers"));
  return data.samplers ?? [];
}

export async function fetchOutputs(): Promise<OutputEntry[]> {
  return handleResponse<OutputEntry[]>(await fetch("/api/outputs"));
}

export async function generateImages(
  payload: GenerateRequest,
  signal?: AbortSignal
): Promise<GenerateResponse> {
  return handleResponse<GenerateResponse>(
    await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    })
  );
}

// ─── Dataset ──────────────────────────────────────────────────────────
export async function fetchDataset(): Promise<DatasetEntry[]> {
  return handleResponse<DatasetEntry[]>(await fetch("/api/dataset"));
}

export async function exportDataset(ids: string[] | null): Promise<Blob> {
  const response = await fetch("/api/dataset/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!response.ok) throw new Error("Export failed");
  return response.blob();
}

// ─── Training ─────────────────────────────────────────────────────────
export async function uploadTrainingImages(files: File[]): Promise<TrainingUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return handleResponse<TrainingUploadResponse>(
    await fetch("/api/train/upload", { method: "POST", body: form })
  );
}

export async function generateCaptions(
  job_id: string,
  prefix: string
): Promise<TrainingCaptionResponse> {
  return handleResponse<TrainingCaptionResponse>(
    await fetch("/api/train/caption", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id, prefix }),
    })
  );
}

export async function startTraining(req: TrainingStartRequest): Promise<{ job_id: string }> {
  return handleResponse<{ job_id: string }>(
    await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    })
  );
}

export async function fetchTrainingStatus(job_id: string): Promise<TrainingStatus> {
  return handleResponse<TrainingStatus>(await fetch(`/api/train/status/${job_id}`));
}

export async function cancelTraining(job_id: string): Promise<void> {
  await fetch(`/api/train/cancel/${job_id}`, { method: "POST" });
}
