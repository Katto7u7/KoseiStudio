import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Annotation, ImageRow, Klass, Origin, Project, get, post, put } from "../lib/api";
import { usePoll, useStored } from "../lib/hooks";
import { Card, Confirm, Empty, Notice, OriginPill, PageHead, StatusPill, Tabs, Thumb, useAction } from "../ui/kit";

const FILTERS = [
  { value: "needs_review", label: "Revisar" },
  { value: "candidate", label: "Candidatas" },
  { value: "no_detection", label: "Sin detección" },
  { value: "approved", label: "Aprobadas" },
  { value: "rejected", label: "Rechazadas" },
] as const;
type Filter = (typeof FILTERS)[number]["value"];

interface Box { class_id: number; cx: number; cy: number; w: number; h: number; polygon?: number[] | null; confidence?: number | null; source?: string }

export default function ReviewPage({ projectId }: { projectId: string }) {
  const [origin, setOrigin] = useStored<Origin>(`kosei_rev_origin_${projectId}`, "real");
  const [status, setStatus] = useState<Filter>("needs_review");
  const { data: project, refresh: reloadProject } = usePoll(() => get<Project>(`/projects/${projectId}`), 4000, [projectId]);
  const { data, refresh } = usePoll(
    () => get<{ total: number; items: ImageRow[] }>(`/projects/${projectId}/images?origin=${origin}&status=${status}&limit=200`),
    5000, [projectId, origin, status],
  );
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [bulk, setBulk] = useState(false);
  const { busy, run } = useAction();
  const items = data?.items ?? [];
  const counts = project?.pipeline.by_status[origin] ?? {};
  const pending = counts.pending ?? 0;

  const advance = useCallback(() => {
    setSelectedId((current) => {
      const index = items.findIndex((i) => i.id === current);
      const next = items[index + 1] ?? items[index - 1];
      return next && next.id !== current ? next.id : null;
    });
    refresh();
    reloadProject();
  }, [items, refresh, reloadProject]);

  useEffect(() => {
    if (!selectedId && items.length) setSelectedId(items[0].id);
  }, [items, selectedId]);
  useEffect(() => { setSelectedId(null); }, [origin, status]);

  return (
    <>
      <PageHead
        title="Etiquetas"
        sub="Las etiquetas automáticas son candidatas. Solo lo que apruebas aquí se exporta, y las reales aprobadas son la verdad de referencia del experimento."
        actions={
          <>
            <select aria-label="Origen" value={origin} onChange={(e) => setOrigin(e.target.value as Origin)} style={{ width: 150 }}>
              <option value="real">Reales</option>
              <option value="synthetic">Sintéticas</option>
            </select>
            <button className="btn" disabled={busy} onClick={() => run(() => post(`/projects/${projectId}/label`), "Etiquetado en cola")}>Etiquetar pendientes{pending ? ` (${pending})` : ""}</button>
            {status === "candidate" && items.length > 0 && <button className="btn btn-primary" disabled={busy} onClick={() => setBulk(true)}>Aprobar candidatas</button>}
          </>
        }
      />
      <Tabs value={status} onChange={setStatus} options={FILTERS.map((f) => ({ value: f.value, label: f.label, count: counts[f.value] ?? 0 }))} />
      {selectedId && project?.classes && <Inspector key={selectedId} imageId={selectedId} classes={project.classes} onDone={advance} />}
      <div style={{ marginTop: 16 }}>
        <Card title={`${data?.total ?? 0} imágenes`}>
          {items.length
            ? <div className="gallery">{items.map((i) => <Thumb key={i.id} image={i} selected={i.id === selectedId} onClick={() => setSelectedId(i.id)} />)}</div>
            : <Empty title="Nada por aquí">{pending ? `Hay ${pending} imágenes pendientes de etiquetar.` : "No hay imágenes en este estado."}</Empty>}
        </Card>
      </div>
      {bulk && (
        <Confirm
          title="Aprobar candidatas" action={`Aprobar ${counts.candidate ?? 0}`}
          body="Las candidatas superaron el umbral de confianza y no tienen avisos, pero aprobarlas sin verlas es aceptar el criterio automático. Revisa una muestra antes si son para el experimento."
          onClose={() => setBulk(false)}
          onConfirm={async () => {
            setBulk(false);
            await run(() => post<{ changed: number }>(`/projects/${projectId}/review/bulk`, { from_status: "candidate", origin, status: "approved" }), "Candidatas aprobadas");
            refresh();
            reloadProject();
          }}
        />
      )}
    </>
  );
}

function Inspector({ imageId, classes, onDone }: { imageId: string; classes: Klass[]; onDone: () => void }) {
  const [image, setImage] = useState<(ImageRow & { annotations: Annotation[] }) | null>(null);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [dirty, setDirty] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [negativeAsk, setNegativeAsk] = useState(false);
  const { busy, run } = useAction();

  const load = useCallback(async () => {
    const row = await get<ImageRow & { annotations: Annotation[] }>(`/images/${imageId}`);
    setImage(row);
    setBoxes(row.annotations.map((a) => ({ class_id: a.class_id, cx: a.cx, cy: a.cy, w: a.w, h: a.h, polygon: a.polygon, confidence: a.confidence, source: a.source })));
    setDirty(false);
    setSelected(null);
  }, [imageId]);
  useEffect(() => { load(); }, [load]);

  const save = useCallback(async () => {
    await put(`/images/${imageId}/annotations`, { boxes: boxes.map(({ class_id, cx, cy, w, h }) => ({ class_id, cx, cy, w, h })) });
    setDirty(false);
  }, [boxes, imageId]);

  const review = useCallback(async (status: "approved" | "rejected", negative = false) => {
    const ok = await run(async () => {
      if (dirty && status === "approved") await save();
      await post(`/images/${imageId}/review`, { status, negative });
      return true;
    });
    if (ok) onDone();
  }, [dirty, imageId, onDone, run, save]);

  const approve = useCallback(() => {
    if (boxes.length === 0) setNegativeAsk(true);
    else review("approved");
  }, [boxes.length, review]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.matches("input, textarea, select")) return;
      if (e.key === "a") approve();
      else if (e.key === "r") review("rejected");
      else if (e.key === "d") setDrawing((v) => !v);
      else if ((e.key === "Delete" || e.key === "Backspace") && selected !== null) {
        setBoxes((list) => list.filter((_, i) => i !== selected));
        setSelected(null);
        setDirty(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [approve, review, selected]);

  if (!image) return <div className="card" style={{ minHeight: 320 }} />;
  const reasons: string[] = image.qc?.label_reasons ?? [];
  const defaultClass = image.class_id ?? classes[0]?.id;

  return (
    <div className="inspector">
      <BoxCanvas
        image={image} boxes={boxes} selected={selected} drawing={drawing} classes={classes}
        onSelect={setSelected}
        onChange={(next) => { setBoxes(next); setDirty(true); }}
        onDrawn={(box) => {
          setBoxes((list) => [...list, { ...box, class_id: defaultClass, source: "humano" }]);
          setDirty(true);
          setDrawing(false);
          setSelected(boxes.length);
        }}
      />
      <div className="stack">
        <Card>
          <div className="stack">
            <div className="row-between">
              <div className="row"><OriginPill origin={image.origin} /><StatusPill status={image.status} /></div>
              {image.split && <span className="pill">{image.split}</span>}
            </div>
            {reasons.length > 0 && <Notice kind="warn" title="Por qué requiere revisión">{reasons.join("\n")}</Notice>}
            <div className="row">
              <button className="btn btn-primary" disabled={busy} onClick={approve}>Aprobar <span className="kbd" style={{ color: "inherit" }}>A</span></button>
              <button className="btn btn-danger" disabled={busy} onClick={() => review("rejected")}>Rechazar <span className="kbd">R</span></button>
            </div>
            <div className="row">
              <button className="btn btn-small" onClick={() => setDrawing((v) => !v)}>{drawing ? "Cancelar caja" : "Dibujar caja"} <span className="kbd">D</span></button>
              <button className="btn btn-small" disabled={!dirty || busy} onClick={() => run(save, "Cajas guardadas; máscaras recalculadas").then(load)}>Guardar y recalcular máscara</button>
            </div>
            {dirty && <p className="faint">Cambios sin guardar. Aprobar también los guarda.</p>}
          </div>
        </Card>
        <Card title={`${boxes.length} ${boxes.length === 1 ? "anotación" : "anotaciones"}`}>
          {boxes.length ? (
            <div className="stack" style={{ gap: 8 }}>
              {boxes.map((box, i) => (
                <div key={i} className="row-between" style={{ padding: "6px 8px", borderRadius: 4, background: selected === i ? "var(--raised)" : undefined, cursor: "pointer" }} onClick={() => setSelected(i)}>
                  <div>
                    <select aria-label="Clase" value={box.class_id} style={{ width: "auto", padding: "2px 4px" }} onClick={(e) => e.stopPropagation()} onChange={(e) => {
                      setBoxes(boxes.map((b, j) => (j === i ? { ...b, class_id: +e.target.value } : b)));
                      setDirty(true);
                    }}>
                      {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
                    </select>
                    <div className="faint">{box.source ?? "humano"}{box.confidence != null ? `, confianza ${box.confidence.toFixed(2)}` : ""}{box.polygon ? ", con máscara" : ""}</div>
                  </div>
                  <button className="btn btn-small" aria-label="Quitar caja" onClick={(e) => { e.stopPropagation(); setBoxes(boxes.filter((_, j) => j !== i)); setSelected(null); setDirty(true); }}>Quitar</button>
                </div>
              ))}
            </div>
          ) : <p className="muted">Sin cajas. Dibuja una, o aprueba si la imagen de verdad no contiene la clase.</p>}
        </Card>
        {image.generation && (
          <Card title="Generación">
            <dl className="meta-list">
              <dt>Prompt</dt><dd>{image.generation.prompt}</dd>
              <dt>Seed</dt><dd className="mono">{image.generation.seed}</dd>
              <dt>Modelo</dt><dd>{image.generation.checkpoint}</dd>
              <dt>Clase</dt><dd>{image.class_name}, por {image.class_source}</dd>
            </dl>
          </Card>
        )}
      </div>
      {negativeAsk && (
        <Confirm
          title="Aprobar sin cajas" action="Es un negativo real"
          body="La imagen se exportará sin etiquetas y el detector aprenderá que aquí no hay ningún objeto. Si el objeto sí aparece, cancela y dibuja la caja."
          onClose={() => setNegativeAsk(false)}
          onConfirm={() => { setNegativeAsk(false); review("approved", true); }}
        />
      )}
    </div>
  );
}

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));

function BoxCanvas({ image, boxes, selected, drawing, classes, onSelect, onChange, onDrawn }: {
  image: ImageRow; boxes: Box[]; selected: number | null; drawing: boolean; classes: Klass[];
  onSelect: (i: number | null) => void; onChange: (boxes: Box[]) => void; onDrawn: (box: Omit<Box, "class_id">) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [draft, setDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const drag = useRef<{ mode: "move" | "resize"; index: number; startX: number; startY: number; box: Box } | null>(null);
  const color = image.origin === "real" ? "var(--real)" : "var(--synth)";
  const className = useMemo(() => Object.fromEntries(classes.map((c) => [c.id, c.name])), [classes]);

  const point = (e: React.PointerEvent) => {
    const rect = ref.current!.getBoundingClientRect();
    return { x: clamp((e.clientX - rect.left) / rect.width), y: clamp((e.clientY - rect.top) / rect.height) };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = point(e);
    if (draft) { setDraft({ ...draft, x1: p.x, y1: p.y }); return; }
    const d = drag.current;
    if (!d) return;
    const dx = p.x - d.startX, dy = p.y - d.startY;
    const b = d.box;
    let next: Box;
    if (d.mode === "move") {
      next = { ...b, cx: clamp(b.cx + dx, b.w / 2, 1 - b.w / 2), cy: clamp(b.cy + dy, b.h / 2, 1 - b.h / 2) };
    } else {
      const x0 = b.cx - b.w / 2, y0 = b.cy - b.h / 2;
      const x1 = clamp(b.cx + b.w / 2 + dx, x0 + 0.01), y1 = clamp(b.cy + b.h / 2 + dy, y0 + 0.01);
      next = { ...b, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0 };
    }
    next.polygon = null;
    next.confidence = null;
    next.source = "humano";
    onChange(boxes.map((box, i) => (i === d.index ? next : box)));
  };

  const onPointerUp = () => {
    if (draft) {
      const x0 = Math.min(draft.x0, draft.x1), x1 = Math.max(draft.x0, draft.x1);
      const y0 = Math.min(draft.y0, draft.y1), y1 = Math.max(draft.y0, draft.y1);
      if (x1 - x0 > 0.01 && y1 - y0 > 0.01) onDrawn({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, w: x1 - x0, h: y1 - y0, source: "humano" });
      setDraft(null);
    }
    drag.current = null;
  };

  return (
    <div
      ref={ref} className="canvas-wrap" style={{ cursor: drawing ? "crosshair" : "default" }}
      onPointerDown={(e) => {
        if (drawing) {
          const p = point(e);
          setDraft({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
          ref.current?.setPointerCapture(e.pointerId);
        } else if (e.target === e.currentTarget || (e.target as Element).tagName === "svg" || (e.target as Element).tagName === "IMG") {
          onSelect(null);
        }
      }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
    >
      <img src={image.file_url} alt={image.filename} draggable={false} />
      <svg viewBox="0 0 1 1" preserveAspectRatio="none">
        {boxes.map((b, i) => b.polygon && (
          <polygon key={`p${i}`} className="anno-poly" fill={color} points={b.polygon.reduce((acc, v, k) => acc + (k % 2 ? `,${v} ` : `${v}`), "")} />
        ))}
        {boxes.map((b, i) => (
          <g key={i}>
            <rect
              className="anno-box" x={b.cx - b.w / 2} y={b.cy - b.h / 2} width={b.w} height={b.h}
              stroke={color} strokeDasharray={b.source === "humano" ? undefined : undefined}
              style={{ pointerEvents: drawing ? "none" : "auto", strokeWidth: selected === i ? 3 : 2 }}
              onPointerDown={(e) => {
                e.stopPropagation();
                onSelect(i);
                const p = point(e);
                drag.current = { mode: "move", index: i, startX: p.x, startY: p.y, box: b };
                ref.current?.setPointerCapture(e.pointerId);
              }}
            />
          </g>
        ))}
        {draft && (
          <rect className="anno-box" x={Math.min(draft.x0, draft.x1)} y={Math.min(draft.y0, draft.y1)} width={Math.abs(draft.x1 - draft.x0)} height={Math.abs(draft.y1 - draft.y0)} stroke="#fff" strokeDasharray="0.01 0.01" />
        )}
      </svg>
      {boxes.map((b, i) => {
        const left = `${(b.cx - b.w / 2) * 100}%`;
        const top = `${(b.cy - b.h / 2) * 100}%`;
        return (
          <span key={`l${i}`} className="anno-label" style={{ left, top: `max(14px, ${top})`, background: color }}>
            {className[b.class_id] ?? "?"}{b.confidence != null ? ` ${b.confidence.toFixed(2)}` : ""}
          </span>
        );
      })}
      {selected !== null && boxes[selected] && !drawing && (
        <span
          role="slider" aria-label="Cambiar tamaño de la caja" tabIndex={0}
          style={{
            position: "absolute", width: 12, height: 12, background: "#fff", border: "1px solid #14171B", cursor: "nwse-resize",
            left: `calc(${(boxes[selected].cx + boxes[selected].w / 2) * 100}% - 6px)`,
            top: `calc(${(boxes[selected].cy + boxes[selected].h / 2) * 100}% - 6px)`,
          }}
          onPointerDown={(e) => {
            e.stopPropagation();
            const p = point(e);
            drag.current = { mode: "resize", index: selected, startX: p.x, startY: p.y, box: boxes[selected] };
            ref.current?.setPointerCapture(e.pointerId);
          }}
        />
      )}
    </div>
  );
}
