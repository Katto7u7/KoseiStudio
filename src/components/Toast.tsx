import type { Toast as ToastItem } from "./useToast";

interface Props {
  toasts: ToastItem[];
  onRemove: (id: string) => void;
}

export default function Toast({ toasts, onRemove }: Props) {
  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          <span>{t.message}</span>
          <button onClick={() => onRemove(t.id)} aria-label="Cerrar">×</button>
        </div>
      ))}
    </div>
  );
}
