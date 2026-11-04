import { useCallback, useEffect, useState } from "react";
import { exportDataset, fetchDataset } from "../../api";
import type { DatasetEntry } from "../../types";

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function DatasetSection() {
  const [entries, setEntries]     = useState<DatasetEntry[]>([]);
  const [selected, setSelected]   = useState<Set<string>>(new Set());
  const [loading, setLoading]     = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [classFilter, setClassFilter] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDataset();
      setEntries(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo cargar el dataset");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const allClasses = Array.from(new Set(entries.map((e) => e.class_label).filter(Boolean))) as string[];

  const filtered = classFilter
    ? entries.filter((e) => e.class_label === classFilter)
    : entries;

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === filtered.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(filtered.map((e) => e.id)));
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      const ids = selected.size > 0 ? Array.from(selected) : null;
      const blob = await exportDataset(ids);
      downloadBlob(blob, `kosei-dataset-${Date.now()}.zip`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error al exportar");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="section-content">
      <div className="panel stagger-1" style={{ marginBottom: 16 }}>
        <div className="panel-header">
          <span className="panel-header-title">Dataset</span>
          <span className="panel-header-sub">{entries.length} imágenes generadas</span>
        </div>

        {error && <div className="alert">{error}</div>}

        {/* Filters */}
        {allClasses.length > 0 && (
          <div className="filter-chips">
            <button
              className={`chip${classFilter === null ? " active" : ""}`}
              onClick={() => setClassFilter(null)}
              type="button"
            >
              Todas
            </button>
            {allClasses.map((c) => (
              <button
                key={c}
                className={`chip${classFilter === c ? " active" : ""}`}
                onClick={() => setClassFilter(c)}
                type="button"
              >
                {c}
              </button>
            ))}
          </div>
        )}

        {/* Toolbar */}
        <div className="dataset-toolbar">
          <button className="ghost small" onClick={toggleAll} type="button">
            {selected.size === filtered.length && filtered.length > 0 ? "Deseleccionar todo" : "Seleccionar todo"}
          </button>
          <button
            className="primary"
            style={{ width: "auto", padding: "7px 14px" }}
            onClick={handleExport}
            disabled={exporting}
            type="button"
          >
            {exporting ? "Exportando..." : `Exportar ZIP${selected.size > 0 ? ` (${selected.size})` : ""}`}
          </button>
          <button className="ghost small" onClick={load} type="button">↺ Recargar</button>
          <span className="count-badge">{filtered.length} imágenes</span>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: "32px", color: "var(--text-muted)", fontSize: 13 }}>
            Cargando dataset...
          </div>
        ) : filtered.length === 0 ? (
          <div className="callout callout-info">
            No hay imágenes en el dataset aún. Genera imágenes en el Studio para empezar.
          </div>
        ) : (
          <div className="dataset-grid">
            {filtered.map((entry) => (
              <div
                key={entry.id}
                className={`dataset-card${selected.has(entry.id) ? " selected" : ""}`}
                onClick={() => toggleSelect(entry.id)}
              >
                <img src={entry.image_url} alt={entry.id} loading="lazy" />
                <div className="dataset-card-meta">
                  {entry.class_label && (
                    <span className="class-badge">{entry.class_label}</span>
                  )}
                  {entry.has_detection && (
                    <span style={{ fontSize: 10, color: "var(--text-muted)" }} title="Detección">⬡</span>
                  )}
                  {entry.has_segmentation && (
                    <span style={{ fontSize: 10, color: "var(--text-muted)" }} title="Segmentación">◈</span>
                  )}
                </div>
                <div className="dataset-card-check">
                  {selected.has(entry.id) ? "✓" : ""}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
