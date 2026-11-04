export type GenerateRequest = {
  prompt: string;
  steps: number;
  width: number;
  height: number;
  batch: number;
  sampler: string;
  sd_model: string;
  lora: string;
  negative_prompt?: string;
  cfg_scale?: number;
  target_class?: string;
  class_id?: number;
};

export type GenerateResult = {
  id: string;
  image_url: string;
};

export type GenerateResponse = {
  results: GenerateResult[];
  prompt: string;
  negative_prompt: string;
  sampler: string;
  sd_model: string;
  width: number;
  height: number;
  steps: number;
};

export type HealthResponse = {
  status: string;
  sd_host: string;
  sd_port: number;
};

export type OutputEntry = {
  id: string;
  image_url: string;
  metadata_url: string | null;
  timestamp: string;
};

export type DatasetEntry = {
  id: string;
  image_url: string;
  class_label: string | null;
  has_detection: boolean;
  has_segmentation: boolean;
  lora_used: string | null;
  timestamp: string;
};

export type TrainingUploadResponse = {
  job_id: string;
  image_count: number;
  filenames: string[];
};

export type CaptionEntry = {
  filename: string;
  caption: string;
};

export type TrainingCaptionResponse = {
  captions: CaptionEntry[];
};

export type TrainingStartRequest = {
  job_id: string;
  output_name: string;
  base_model: string;
  token_prefix: string;
  n_repeats: number;
  network_dim: number;
  learning_rate: number;
  max_steps: number;
  batch_size: number;
  captions: CaptionEntry[];
};

export type TrainingStatus = {
  status: "idle" | "running" | "done" | "error" | "cancelled";
  current_step: number;
  total_steps: number;
  log_tail: string[];
  elapsed: number;
  output_name?: string;
};
