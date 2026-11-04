import { useEffect, useRef, useState } from "react";
import { cancelTraining, fetchTrainingStatus } from "../../api";
import type { TrainingStatus } from "../../types";

interface Props {
  jobId: string;
  onReset: () => void;
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export default function StepProgress({ jobId, onReset }: Props) {
  const [status, setStatus] = useState<TrainingStatus>({
    status: "running",
    current_step: 0,
    total_steps: 0,
    log_tail: [],
    elapsed: 0,
  });
  const [cancelling, setCancelling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const s = await fetchTrainingStatus(jobId);
        setStatus(s);
        if (s.status === "done" || s.status === "error" || s.status === "cancelled") {
          if (pollRef.current) clearInterval(pollRef.current);
        }
      } catch {
        // network error — keep polling
      }
    };

    poll();
    pollRef.current = setInterval(poll, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobId]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status.log_tail]);

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await cancelTraining(jobId);
    } finally {
      setCancelling(false);
    }
  };

  const progress = status.total_steps > 0
    ? Math.round((status.current_step / status.total_steps) * 100)
    : 0;

  const isDone = status.status === "done";
  const isError = status.status === "error";
  const isCancelled = status.status === "cancelled";
  const isFinished = isDone || isError || isCancelled;

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-secondary)" }}>
          {isDone ? "✓ Completado" : isError ? "✗ Error" : isCancelled ? "— Cancelado" : `Step ${status.current_step} / ${status.total_steps}`}
        </span>
        <span className="elapsed-badge">{formatTime(status.elapsed)}</span>
      </div>

      <div className="progress-bar" style={{ marginBottom: 16 }}>
        <div
          className="progress-fill"
          style={{
            width: `${progress}%`,
            background: isDone ? "var(--success)" : isError ? "var(--danger)" : undefined,
          }}
        />
      </div>

      {status.log_tail.length > 0 && (
        <div>
          <span className="section-label">Log</span>
          <div className="log-output" ref={logRef} style={{ marginTop: 8 }}>
            {status.log_tail.join("\n")}
          </div>
        </div>
      )}

      {isDone && status.output_name && (
        <div className="callout callout-info" style={{ marginTop: 16 }}>
          LoRA <strong>{status.output_name}.safetensors</strong> guardado y disponible en Studio.
        </div>
      )}

      {isError && (
        <div className="alert" style={{ marginTop: 12 }}>
          El entrenamiento falló. Revisa el log para más detalles.
        </div>
      )}

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end", marginTop: 20 }}>
        {!isFinished && (
          <button
            className="danger"
            style={{ width: "auto", padding: "9px 18px" }}
            onClick={handleCancel}
            disabled={cancelling}
            type="button"
          >
            {cancelling ? "Cancelando..." : "Cancelar training"}
          </button>
        )}
        {isFinished && (
          <button
            className="primary"
            style={{ width: "auto", padding: "9px 20px" }}
            onClick={onReset}
            type="button"
          >
            Nuevo training
          </button>
        )}
      </div>
    </div>
  );
}
