import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
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

/** Lleva un numero a su valor real con una transicion breve. */
export function useCountUp(target: number, duration = 650): number {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduced || typeof requestAnimationFrame === "undefined" || from.current === target) {
      from.current = target;
      setValue(target);
      return;
    }
    const start = performance.now();
    const origin = from.current;
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(Math.round(origin + (target - origin) * eased));
      if (t < 1) frame = requestAnimationFrame(tick);
      else from.current = target;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration]);
  return value;
}

export function CountUp({ value }: { value: number }) {
  return <>{useCountUp(value)}</>;
}

/* ---------- Superposiciones ---------- */
function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

export function Confirm({ title, body, action, danger, onConfirm, onClose }: {
  title: string; body: ReactNode; action: string; danger?: boolean; onConfirm: () => void; onClose: () => void;
}) {
  useEscape(onClose);
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

export function Drawer({ title, onClose, footer, children }: { title: string; onClose: () => void; footer?: ReactNode; children: ReactNode }) {
  useEscape(onClose);
  return (
    <>
      <div className="drawer-back" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label={title}>
        <div className="drawer-head">
          <h2>{title}</h2>
          <button className="btn btn-small" onClick={onClose}>Cerrar</button>
        </div>
        <div className="drawer-body">{children}</div>
        {footer && <div className="drawer-foot">{footer}</div>}
      </aside>
    </>
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
      {children && <div>{children}</div>}
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
      <div className="stat-value">{typeof value === "number" ? <CountUp value={value} /> : value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

/** Ayuda breve junto a un termino. Se abre al pasar el cursor o con el teclado. */
export function Help({ children }: { children: ReactNode }) {
  return (
    <span className="help">
      <button type="button" className="help-btn" aria-label="Ayuda" onClick={(e) => e.preventDefault()}>?</button>
      <span className="help-pop" role="tooltip">{children}</span>
    </span>
  );
}

export function Field({ label, hint, help, children }: { label: string; hint?: ReactNode; help?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}{help && <Help>{help}</Help>}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function Check({ checked, onChange, children }: { checked: boolean; onChange: (v: boolean) => void; children: ReactNode }) {
  return (
    <label className="check-row">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{children}</span>
    </label>
  );
}

export function Segmented<T extends string>({ value, onChange, options, label }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>
      ))}
    </div>
  );
}

export function Progress({ value, total, live }: { value: number; total: number; live?: boolean }) {
  if (!total) return <div className="progress indeterminate" role="progressbar"><i /></div>;
  const pct = Math.min(100, Math.round((value / total) * 100));
  return <div className={`progress${live ? " live" : ""}`} role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}><i style={{ width: `${pct}%` }} /></div>;
}

export function Skeleton({ height = 120 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-hidden />;
}

export function SkeletonPage() {
  return (
    <div className="stack" aria-busy="true">
      <Skeleton height={34} />
      <Skeleton height={180} />
      <Skeleton height={260} />
    </div>
  );
}

export interface Requirement { ok: boolean; text: string; to?: string; action?: string }

/** Que falta para poder hacer algo, con el camino para resolverlo. */
export function Requirements({ items }: { items: Requirement[] }) {
  return (
    <div className="reqs">
      {items.map((item) => (
        <div key={item.text} className={`req ${item.ok ? "ok" : "todo"}`}>
          <span className="req-text">
            <span className="req-mark" aria-hidden>{item.ok ? "✓" : ""}</span>
            <span>{item.text}<span className="sr-hidden">{item.ok ? " (listo)" : " (falta)"}</span></span>
          </span>
          {!item.ok && item.to && <Link className="btn btn-small" to={item.to}>{item.action ?? "Ir"}</Link>}
        </div>
      ))}
    </div>
  );
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

export const jobLabel = (job: Pick<Job, "kind">) => JOB_LABEL[job.kind] ?? job.kind;

export function OriginPill({ origin }: { origin: "real" | "synthetic" }) {
  return <span className={`pill ${origin === "real" ? "pill-real" : "pill-synth"}`}>{origin === "real" ? "Real" : "Sintética"}</span>;
}

const SPLIT_LABEL: Record<string, string> = { train: "Entrenamiento", val: "Validación", test: "Prueba" };
export const splitLabel = (split: string) => SPLIT_LABEL[split] ?? split;

/** Miniatura con la marca de deteccion del propio etiquetador. */
export function Thumb({ image, selected, onClick, compact, index = 0 }: { image: ImageRow; selected?: boolean; onClick?: () => void; compact?: boolean; index?: number }) {
  const tag = useMemo(() => {
    if (compact) return null;
    if (image.top_confidence != null) return `${image.class_name ?? "?"} ${image.top_confidence.toFixed(2)}`;
    if (image.annotation_count) return `${image.class_name ?? "?"} ✓`;
    return null;
  }, [image, compact]);
  return (
    <button
      type="button"
      className={`thumb origin-${image.origin}${selected ? " selected" : ""}`}
      style={{ animationDelay: `${Math.min(index, 14) * 22}ms` }}
      onClick={onClick}
      aria-pressed={selected}
      aria-label={`${image.filename}, ${STATUS_LABEL[image.status] ?? image.status}`}
    >
      <img src={image.thumb_url} alt="" loading="lazy" />
      {image.split && !compact && <span className="pill thumb-status">{splitLabel(image.split)}</span>}
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

export function Stepper<T extends string>({ value, onChange, steps }: {
  value: T; onChange: (v: T) => void; steps: { value: T; label: string; done?: boolean }[];
}) {
  return (
    <div className="stepper" role="tablist">
      {steps.map((s, i) => (
        <button key={s.value} role="tab" aria-selected={s.value === value} className={`step${s.value === value ? " active" : ""}${s.done ? " done" : ""}`} onClick={() => onChange(s.value)}>
          <span className="step-mark" aria-hidden>{s.done ? "✓" : i + 1}</span>
          {s.label}
        </button>
      ))}
    </div>
  );
}

/** Regla horizontal del parecido al sujeto, con el umbral y la referencia real marcados. */
export function FidelityScale({ value, threshold, baseline }: { value: number; threshold: number; baseline?: number | null }) {
  const pct = (v: number) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <div className="fidelity" style={{ paddingTop: 16 }}>
      <div className="fidelity-scale" role="img" aria-label={`Parecido ${value}, umbral ${threshold}${baseline != null ? `, referencia ${baseline}` : ""}`}>
        <div className="fidelity-fill" style={{ width: pct(value) }} />
        <div className="fidelity-mark" style={{ left: pct(threshold) }}><span>umbral {threshold}</span></div>
        {baseline != null && <div className="fidelity-mark" style={{ left: pct(baseline), background: "var(--real-fill)" }}><span>real {baseline}</span></div>}
      </div>
    </div>
  );
}
