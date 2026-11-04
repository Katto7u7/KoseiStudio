import { useEffect } from "react";
import type { GenerateResult } from "../types";

interface Props {
  outputs: GenerateResult[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onClear: () => void;
}

export default function Gallery({ outputs, activeId, onSelect, onClear }: Props) {
  useEffect(() => {
    if (outputs.length === 0) return;

    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const currentIndex = outputs.findIndex((o) => o.id === activeId);
      if (currentIndex === -1) return;
      if (e.key === "ArrowLeft" && currentIndex > 0) {
        onSelect(outputs[currentIndex - 1].id);
      }
      if (e.key === "ArrowRight" && currentIndex < outputs.length - 1) {
        onSelect(outputs[currentIndex + 1].id);
      }
    };

    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [outputs, activeId, onSelect]);

  if (outputs.length === 0) return null;

  return (
    <>
      <div className="gallery-header">
        <span className="gallery-count">{outputs.length} imagen{outputs.length !== 1 ? "es" : ""}</span>
        <button className="ghost small" onClick={onClear} type="button">
          Limpiar
        </button>
      </div>
      <div className="thumb-grid">
        {outputs.map((item) => (
          <button
            key={item.id}
            className={`thumb ${item.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(item.id)}
            type="button"
          >
            <img src={item.image_url} alt={`Miniatura ${item.id}`} />
          </button>
        ))}
      </div>
    </>
  );
}
