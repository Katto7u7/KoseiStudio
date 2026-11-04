import { useEffect, useMemo, useRef, useState } from "react";
import { fetchLoras, fetchModels, fetchSamplers, generateImages } from "../api";
import type { GenerateResult } from "../types";
import Controls from "./Controls";
import Gallery from "./Gallery";
import ImageViewer from "./ImageViewer";
import Toast from "./Toast";
import { useToast } from "./useToast";

const DEFAULT_NEGATIVE_PROMPT =
  "ugly, out of frame, deformations, weird face, extra hands, extra fingers, mutilations";

function ls<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw !== null ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function persist<T>(key: string, setter: (v: T) => void): (v: T) => void {
  return (v: T) => {
    localStorage.setItem(key, JSON.stringify(v));
    setter(v);
  };
}

type LoadState = "idle" | "loading" | "success" | "error";

export default function StudioSection() {
  const [prompt, setPromptRaw]                 = useState<string>(() => ls("af_prompt", ""));
  const [negativePrompt, setNegativePromptRaw] = useState<string>(() => ls("af_negativePrompt", DEFAULT_NEGATIVE_PROMPT));
  const [steps, setStepsRaw]                   = useState<number>(() => ls("af_steps", 20));
  const [cfgScale, setCfgScaleRaw]             = useState<number>(() => ls("af_cfgScale", 7));
  const [width, setWidthRaw]                   = useState<number>(() => ls("af_width", 512));
  const [height, setHeightRaw]                 = useState<number>(() => ls("af_height", 512));
  const [batch, setBatchRaw]                   = useState<number>(() => ls("af_batch", 1));
  const [sampler, setSamplerRaw]               = useState<string>(() => ls("af_sampler", "Euler a"));
  const [sdModel, setSdModelRaw]               = useState<string>(() => ls("af_sdModel", ""));
  const [lora, setLoraRaw]                     = useState<string>(() => ls("af_lora", "NINGUNO"));
  const [targetClass, setTargetClassRaw]       = useState<string>(() => ls("af_targetClass", "object"));
  const [classId, setClassIdRaw]               = useState<number>(() => ls("af_classId", 0));

  const setPrompt        = persist("af_prompt",         setPromptRaw);
  const setNegativePrompt= persist("af_negativePrompt", setNegativePromptRaw);
  const setSteps         = persist("af_steps",          setStepsRaw);
  const setCfgScale      = persist("af_cfgScale",       setCfgScaleRaw);
  const setWidth         = persist("af_width",          setWidthRaw);
  const setHeight        = persist("af_height",         setHeightRaw);
  const setBatch         = persist("af_batch",          setBatchRaw);
  const setSampler       = persist("af_sampler",        setSamplerRaw);
  const setSdModel       = persist("af_sdModel",        setSdModelRaw);
  const setLora          = persist("af_lora",           setLoraRaw);
  const setTargetClass   = persist("af_targetClass",    setTargetClassRaw);
  const setClassId       = persist("af_classId",        setClassIdRaw);

  const [models, setModels]     = useState<string[]>([]);
  const [loras, setLoras]       = useState<string[]>([]);
  const [samplers, setSamplers] = useState<string[]>(["Euler a"]);
  const [outputs, setOutputs]   = useState<GenerateResult[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loadState, setLoadState] = useState<LoadState>("idle");
  const [error, setError]         = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const abortRef = useRef<AbortController | null>(null);
  const { toasts, addToast, removeToast } = useToast();

  useEffect(() => {
    let mounted = true;
    Promise.all([fetchModels(), fetchLoras(), fetchSamplers()])
      .then(([modelsData, lorasData, samplersData]) => {
        if (!mounted) return;
        setModels(modelsData);
        if (modelsData.length > 0 && !sdModel) setSdModel(modelsData[0]);
        setLoras(["NINGUNO", ...lorasData]);
        if (samplersData.length > 0) setSamplers(samplersData);
      })
      .catch((err) => { if (mounted) setError(err instanceof Error ? err.message : "No se pudo cargar la config"); });
    return () => { mounted = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && loadState !== "loading") {
        handleGenerate();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadState, prompt, sdModel, steps, width, height, batch, sampler, lora, negativePrompt, cfgScale, targetClass, classId]);

  const activeOutput = useMemo(() => {
    if (outputs.length === 0) return null;
    if (activeId) return outputs.find((o) => o.id === activeId) ?? outputs[outputs.length - 1];
    return outputs[outputs.length - 1];
  }, [activeId, outputs]);

  const handleGenerate = async () => {
    if (!prompt.trim() || !sdModel) return;
    setError(null);
    setLoadState("loading");
    setElapsedSeconds(0);

    const controller = new AbortController();
    abortRef.current = controller;
    const timer = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);

    try {
      const response = await generateImages(
        { prompt, steps, width, height, batch, sampler, sd_model: sdModel, lora,
          negative_prompt: negativePrompt, cfg_scale: cfgScale,
          target_class: targetClass, class_id: classId },
        controller.signal
      );
      setOutputs((prev) => [...prev, ...response.results]);
      if (response.results.length > 0) setActiveId(response.results[response.results.length - 1].id);
      setLoadState("success");
      addToast(`${response.results.length} imagen${response.results.length !== 1 ? "es" : ""} generada${response.results.length !== 1 ? "s" : ""}`, "success");
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        addToast("Generación cancelada", "info");
        setLoadState("idle");
      } else {
        setError(err instanceof Error ? err.message : "Falló la generación");
        setLoadState("error");
      }
    } finally {
      clearInterval(timer);
      abortRef.current = null;
    }
  };

  const handleCancel = () => abortRef.current?.abort();

  const handleClear = () => {
    setOutputs([]);
    setActiveId(null);
    setLoadState("idle");
    setError(null);
  };

  const isBusy = loadState === "loading";
  const statusLabel = { loading: "Generando", success: "Listo", error: "Error", idle: "Listo" }[loadState];
  const statusClass = { loading: "status-busy", success: "status-ready", error: "status-error", idle: "status-ready" }[loadState];

  return (
    <div className="section-content">
      <main className="grid">
        <Controls
          prompt={prompt} setPrompt={setPrompt}
          negativePrompt={negativePrompt} setNegativePrompt={setNegativePrompt}
          steps={steps} setSteps={setSteps}
          cfgScale={cfgScale} setCfgScale={setCfgScale}
          width={width} setWidth={setWidth}
          height={height} setHeight={setHeight}
          batch={batch} setBatch={setBatch}
          sampler={sampler} setSampler={setSampler}
          sdModel={sdModel} setSdModel={setSdModel}
          lora={lora} setLora={setLora}
          targetClass={targetClass} setTargetClass={setTargetClass}
          classId={classId} setClassId={setClassId}
          models={models} loras={loras} samplers={samplers}
          isBusy={isBusy} loadState={loadState} error={error}
          onGenerate={handleGenerate} onCancel={handleCancel} onRetry={handleGenerate}
        />

        <section className="panel output stagger-2">
          <div className="panel-header">
            <span className="panel-header-title">Output</span>
            <span className="panel-header-sub">Ctrl+Enter para generar</span>
          </div>

          <ImageViewer activeOutput={activeOutput} isBusy={isBusy} elapsedSeconds={elapsedSeconds} />

          <Gallery outputs={outputs} activeId={activeId} onSelect={setActiveId} onClear={handleClear} />

          <div className="meta-strip">
            <div className="meta-item">
              <span className="meta-key">Modelo</span>
              <span className="meta-val" title={sdModel}>{sdModel || "—"}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Sampler</span>
              <span className="meta-val">{sampler}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Canvas</span>
              <span className="meta-val">{width}×{height}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Pasos / Batch</span>
              <span className="meta-val">{steps} / {batch}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Clase</span>
              <span className="meta-val">{targetClass || "—"}</span>
            </div>
            <div className="meta-item">
              <span className="meta-key">Estado</span>
              <span className={`meta-val ${statusClass}`}>{statusLabel}</span>
            </div>
          </div>
        </section>
      </main>

      <Toast toasts={toasts} onRemove={removeToast} />
    </div>
  );
}
