import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { ImageRow, JOB_LABEL, Job, STATUS_LABEL } from "../lib/api";

/* ---------- Avisos ---------- */
type ToastKind = "good" | "bad" | "info";
interface ToastItem { id: number; text: string; kind: ToastKind }
const ToastCtx = createContext<(text: string, kind?: ToastKind) => void>(() => undefined);
export const useToast = () => useContext(ToastCtx);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const next = useRef(1);
  const push = useCallback((text: string, kind: ToastKind = "info") => {
    const id = next.current++;
    setItems((v) => [...v, { id, text, kind }]);
    setTimeout(() => setItems((v) => v.filter((t) => t.id !== id)), kind === "bad" ? 9000 : 4500);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => <div key={t.id} className={`toast toast-${t.kind}`}>{t.text}</div>)}
      </div>
    </ToastCtx.Provider>
  );
}

/** Ejecuta una accion y muestra su error sin que cada pagina repita el try/catch. */
export function useAction() {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const run = useCallback(async <T,>(fn: () => Promise<T>, success?: string): Promise<T | undefined> => {
    setBusy(true);
    try {
      const value = await fn();
      if (success) toast(success, "good");
      return value;
    } catch (err) {
      toast(err instanceof Error ? err.message : "La acción falló", "bad");
      return undefined;
    } finally {
      setBusy(false);
    }
  }, [toast]);
  return { busy, run };
}

/* ---------- Confirmacion ---------- */
export function Confirm({ title, body, action, danger, onConfirm, onClose }: {
  title: string; body: ReactNode; action: string; danger?: boolean; onConfirm: () => void; onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <h2>{title}</h2>
        <div className="muted">{body}</div>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <button className="btn" onClick={onClose} autoFocus>Cancelar</button>
          <button className={danger ? "btn btn-danger" : "btn btn-primary"} onClick={onConfirm}>{action}</button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Piezas ---------- */
export function PageHead({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-head">
      <div>
        <h1>{title}</h1>
        {sub && <p className="page-sub">{sub}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  );
}

export function Card({ title, note, actions, children }: { title?: string; note?: ReactNode; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="card">
      {(title || actions) && (
        <div className="card-head">
          <div>
            {title && <h2>{title}</h2>}
            {note && <div className="card-note">{note}</div>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Notice({ kind = "info", title, children }: { kind?: "info" | "warn" | "bad" | "good"; title?: string; children?: ReactNode }) {
  return (
    <div className={`notice notice-${kind}`} role={kind === "bad" ? "alert" : undefined}>
      {title && <strong>{title}</strong>}
      {children && <div className="pre">{children}</div>}
    </div>
  );
}

export function Stat({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div>
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function Progress({ value, total }: { value: number; total: number }) {
  if (!total) return <div className="progress indeterminate" role="progressbar"><i /></div>;
  const pct = Math.min(100, Math.round((value / total) * 100));
  return <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${pct}%` }} /></div>;
}

export function StatusPill({ status }: { status: string }) {
  const tone = status === "approved" ? "pill-good" : status === "rejected" ? "pill-bad"
    : status === "needs_review" || status === "no_detection" ? "pill-warn" : "";
  return <span className={`pill ${tone}`}>{STATUS_LABEL[status] ?? status}</span>;
}

const JOB_STATUS: Record<string, string> = {
  queued: "En cola", running: "En curso", succeeded: "Completado", failed: "Falló",
  cancelled: "Cancelado", interrupted: "Interrumpido",
};

export function JobPill({ status }: { status: string }) {
  const tone = status === "succeeded" ? "pill-good" : status === "failed" ? "pill-bad"
    : status === "interrupted" || status === "cancelled" ? "pill-warn" : "";
  return <span className={`pill ${tone}`}>{JOB_STATUS[status] ?? status}</span>;
}

export const jobLabel = (job: Job) => JOB_LABEL[job.kind] ?? job.kind;

export function OriginPill({ origin }: { origin: "real" | "synthetic" }) {
  return <span className={`pill ${origin === "real" ? "pill-real" : "pill-synth"}`}>{origin === "real" ? "Real" : "Sintética"}</span>;
}

/** Miniatura con la marca de deteccion del propio etiquetador. */
export function Thumb({ image, selected, onClick }: { image: ImageRow; selected?: boolean; onClick?: () => void }) {
  const tag = useMemo(() => {
    if (image.top_confidence != null) return `${image.class_name ?? "?"} ${image.top_confidence.toFixed(2)}`;
    if (image.annotation_count) return `${image.class_name ?? "?"} ✓`;
    return null;
  }, [image]);
  return (
    <button
      type="button"
      className={`thumb origin-${image.origin}${selected ? " selected" : ""}`}
      onClick={onClick}
      aria-pressed={selected}
      aria-label={`${image.filename}, ${STATUS_LABEL[image.status] ?? image.status}`}
    >
      <img src={image.thumb_url} alt="" loading="lazy" />
      {image.split && <span className="pill thumb-status">{image.split}</span>}
      {tag && <span className="tag">{tag}</span>}
    </button>
  );
}

export function Tabs<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string; count?: number }[];
}) {
  return (
    <div className="tabs" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} className={`tab${o.value === value ? " active" : ""}`} onClick={() => onChange(o.value)}>
          {o.label}{o.count !== undefined && <span className="faint"> {o.count}</span>}
        </button>
      ))}
    </div>
  );
}
