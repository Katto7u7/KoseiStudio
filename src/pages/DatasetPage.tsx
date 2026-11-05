import { DragEvent, useEffect, useRef, useState } from "react";
import { ImageRow, Job, Klass, Project, del, get, patch, post, upload } from "../lib/api";
import { formatBytes, usePoll } from "../lib/hooks";
import { BarList, FeatureMap } from "../ui/charts";
import { Card, Confirm, Empty, Field, Notice, PageHead, SkeletonPage, Stat, Stepper, Thumb, splitLabel, useAction, useToast } from "../ui/kit";

type Tab = "images" | "analysis" | "split" | "captions" | "classes";

export default function DatasetPage({ projectId }: { projectId: string }) {
  const { data: project, refresh } = usePoll(() => get<Project>(`/projects/${projectId}`), 4000, [projectId]);
  const [tab, setTab] = useState<Tab | null>(null);
  if (!project) return <SkeletonPage />;
  const pl = project.pipeline;
  const steps: { value: Tab; label: string; done: boolean }[] = [
    { value: "images", label: "Importar", done: pl.real_images > 0 },
    { value: "classes", label: "Clases", done: pl.classes > 0 },
    { value: "split", label: "Partición", done: pl.split_locked },
    { value: "captions", label: "Descripciones", done: pl.by_split.train > 0 && pl.train_captions >= pl.by_split.train },
    { value: "analysis", label: "Análisis", done: false },
  ];
  // Se abre en el primer paso pendiente; despues manda la eleccion del usuario.
  const current = tab ?? steps.find((s) => !s.done)?.value ?? "images";
  return (
    <>
      <PageHead title="Dataset" sub="Tus imágenes reales. Los originales nunca se modifican: las etapas siguientes usan una copia normalizada a sRGB." />
      <Stepper value={current} onChange={setTab} steps={steps} />
      {current === "images" && <ImagesTab project={project} refresh={refresh} />}
      {current === "classes" && <ClassesTab project={project} refresh={refresh} />}
      {current === "split" && <SplitTab project={project} refresh={refresh} />}
      {current === "captions" && <CaptionsTab project={project} />}
      {current === "analysis" && <AnalysisTab projectId={project.id} />}
    </>
  );
}

function ImportReport({ result }: { result: any }) {
  return (
    <Notice kind={result.corrupt?.length ? "warn" : "good"} title={`Importación: ${result.imported} imágenes nuevas`}>
      {[
        result.labels_imported ? `${result.labels_imported} etiquetas importadas, pendientes de verificar.` : "",
        result.duplicates?.length ? `${result.duplicates.length} duplicados exactos omitidos.` : "",
        result.near_duplicates?.length ? `${result.near_duplicates.length} casi duplicados detectados: ${result.near_duplicates.slice(0, 3).map((n: any) => n.file).join(", ")}.` : "",
        result.corrupt?.length ? `${result.corrupt.length} archivos rechazados: ${result.corrupt.slice(0, 5).map((c: any) => `${c.file} (${c.reason})`).join("; ")}.` : "",
      ].filter(Boolean).join("\n")}
    </Notice>
  );
}

function ImagesTab({ project, refresh }: { project: Project; refresh: () => void }) {
  const [split, setSplit] = useState("");
  const query = `origin=real&limit=300${split ? `&split=${split}` : ""}`;
  const { data, refresh: reload } = usePoll(() => get<{ total: number; items: ImageRow[] }>(`/projects/${project.id}/images?${query}`), 5000, [project.id, query]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [report, setReport] = useState<any>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const { busy, run } = useAction();
  const toast = useToast();
  const locked = project.pipeline.split_locked;

  useEffect(() => {
    if (!jobId) return;
    const timer = setInterval(async () => {
      const job = await get<Job>(`/jobs/${jobId}`);
      if (["succeeded", "failed", "cancelled", "interrupted"].includes(job.status)) {
        clearInterval(timer);
        setJobId(null);
        if (job.status === "succeeded") setReport(job.result);
        else toast(job.error ?? "La importación no terminó", "bad");
        reload();
        refresh();
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [jobId, reload, refresh, toast]);

  const send = (files: File[]) => {
    if (!files.length) return;
    setReport(null);
    run(async () => {
      const res = await upload<{ job_id: string }>(`/projects/${project.id}/import`, files);
      setJobId(res.job_id);
    });
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    send(Array.from(e.dataTransfer.files));
  };

  return (
    <div className="stack">
      {!locked ? (
        <div
          className={`dropzone${over ? " over" : ""}`} role="button" tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => { if (e.key === "Enter") input.current?.click(); }}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <strong style={{ color: "var(--paper)" }}>{busy ? "Subiendo..." : jobId ? "Procesando imágenes..." : "Arrastra imágenes o un ZIP, o haz clic para elegirlos"}</strong>
          <span className="faint">JPG, PNG, WEBP, BMP o TIFF. Un ZIP con estructura YOLO conserva sus etiquetas; los .txt junto a cada imagen se toman como descripción.</span>
          <input ref={input} type="file" multiple hidden accept="image/*,.zip" onChange={(e) => { send(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
        </div>
      ) : <Notice kind="info" title="Partición bloqueada">No se pueden importar más imágenes reales: cambiaría la base del experimento.</Notice>}
      {report && <ImportReport result={report} />}
      <Card title={`${data?.total ?? 0} imágenes`} actions={
        <select aria-label="Filtrar por partición" value={split} onChange={(e) => setSplit(e.target.value)} style={{ width: 160 }}>
          <option value="">Todas las particiones</option>
          <option value="train">Entrenamiento</option>
          <option value="val">Validación</option>
          <option value="test">Prueba</option>
          <option value="none">Sin partición</option>
        </select>
      }>
        {data && data.items.length > 0
          ? <div className="gallery">{data.items.map((image) => <Thumb key={image.id} image={image} />)}</div>
          : <p className="muted">No hay imágenes con este filtro.</p>}
      </Card>
    </div>
  );
}

function ClassesTab({ project, refresh }: { project: Project; refresh: () => void }) {
  const [name, setName] = useState("");
  const [detector, setDetector] = useState("coco");
  const [prompt, setPrompt] = useState("");
  const { busy, run } = useAction();
  const [removing, setRemoving] = useState<Klass | null>(null);
  const add = async () => {
    const ok = await run(() => post(`/projects/${project.id}/classes`, { name, detector, detector_prompt: prompt }), "Clase agregada");
    if (ok) { setName(""); setPrompt(""); refresh(); }
  };
  return (
    <div className="grid grid-side">
      <Card title="Nueva clase">
        <div className="fields">
          <Field label="Nombre"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="persona" maxLength={60} /></Field>
          <Field label="Detector para el etiquetado" hint="COCO reconoce 80 categorías comunes. YOLO-World acepta cualquier descripción, con menor precisión.">
            <select value={detector} onChange={(e) => setDetector(e.target.value)}>
              <option value="coco">YOLO11 COCO</option>
              <option value="world">YOLO-World (vocabulario abierto)</option>
            </select>
          </Field>
          <Field label={detector === "coco" ? "Nombre en COCO (en inglés)" : "Descripción en inglés"} hint={detector === "coco" ? 'Por ejemplo "person" o "car". Vacío usa el nombre de la clase.' : 'Por ejemplo "orange handheld electronic device".'}>
            <input value={prompt} onChange={(e) => setPrompt(e.target.value)} maxLength={200} />
          </Field>
          <button className="btn btn-primary" disabled={busy || !name.trim()} onClick={add}>Agregar clase</button>
        </div>
      </Card>
      <Card title="Clases del proyecto" note="El índice es el que se escribe en las etiquetas YOLO exportadas.">
        {project.classes?.length ? (
          <table className="table">
            <thead><tr><th>Índice</th><th>Nombre</th><th>Detector</th><th>Consulta</th><th /></tr></thead>
            <tbody>
              {project.classes.map((c) => (
                <tr key={c.id}>
                  <td className="mono">{c.idx}</td><td>{c.name}</td>
                  <td className="muted">{c.detector === "coco" ? "YOLO11 COCO" : "YOLO-World"}</td>
                  <td className="muted">{c.detector_prompt || "—"}</td>
                  <td style={{ textAlign: "right" }}><button className="btn btn-small" onClick={() => setRemoving(c)}>Quitar</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="muted">Sin clases. Agrega al menos una para poder etiquetar.</p>}
      </Card>
      {removing && (
        <Confirm danger title="Quitar clase" action="Quitar" body={`Se quitará ${removing.name} del proyecto. No se permite si ya tiene anotaciones.`}
          onClose={() => setRemoving(null)}
          onConfirm={async () => { const c = removing; setRemoving(null); await run(() => del(`/projects/${project.id}/classes/${c.id}`), "Clase quitada"); refresh(); }} />
      )}
    </div>
  );
}

function SplitTab({ project, refresh }: { project: Project; refresh: () => void }) {
  const { data: status, refresh: reload } = usePoll(() => get<any>(`/projects/${project.id}/split`), null, [project.id]);
  const n = project.pipeline.real_images;
  const [train, setTrain] = useState(Math.round(n * 0.4));
  const [val, setVal] = useState(Math.round(n * 0.2));
  const [test, setTest] = useState(n - Math.round(n * 0.4) - Math.round(n * 0.2));
  const [seed, setSeed] = useState(0);
  const [confirm, setConfirm] = useState(false);
  const { busy, run } = useAction();
  const locked = project.pipeline.split_locked;
  const label = project.pipeline.by_split;
  return (
    <div className="grid grid-side">
      <Card title="Reparto" note={locked ? `Bloqueado. Huella ${project.split_hash?.slice(0, 16)}` : "Las proporciones se aplican por clase."}>
        <div className="fields">
          <div className="fields-3">
            <Field label="Entrenamiento" help="Fotos con las que aprenden el LoRA y el detector."><input type="number" min={0} value={train} disabled={locked} onChange={(e) => setTrain(+e.target.value)} /></Field>
            <Field label="Validación" help="Para elegir el mejor modelo durante el entrenamiento del detector."><input type="number" min={0} value={val} disabled={locked} onChange={(e) => setVal(+e.target.value)} /></Field>
            <Field label="Prueba" help="Se aparta y no se toca hasta el final: mide el resultado sin trampa."><input type="number" min={0} value={test} disabled={locked} onChange={(e) => setTest(+e.target.value)} /></Field>
          </div>
          <Field label="Semilla (seed)" hint="La misma semilla repite exactamente el mismo reparto."><input type="number" min={0} value={seed} disabled={locked} onChange={(e) => setSeed(+e.target.value)} /></Field>
          <button className="btn" disabled={locked || busy || n === 0} onClick={() => run(() => post(`/projects/${project.id}/split`, { train, val, test, seed }), "Partición aplicada").then(() => { reload(); refresh(); })}>Aplicar partición</button>
          <button className="btn btn-primary" disabled={locked || busy || !label.train || !label.test} onClick={() => setConfirm(true)}>Bloquear partición</button>
          <p className="field-hint">Al bloquear se fija la prueba: el LoRA no verá val ni test, y no se podrán importar más imágenes reales.</p>
        </div>
      </Card>
      <div className="stack">
        <Card title="Reparto actual">
          <div className="stat-row">
            <Stat value={label.train} label="Entrenamiento" /><Stat value={label.val} label="Validación" />
            <Stat value={label.test} label="Prueba" /><Stat value={label.none} label="Sin asignar" />
          </div>
        </Card>
        <Card title="Fuga de datos" note="Compara huellas perceptuales entre particiones y los datos usados por cada LoRA.">
          {status && status.cross_split_near_duplicates.length === 0 && status.lora_leaks.length === 0 && <Notice kind="good" title="Sin fuga detectada" />}
          {status?.cross_split_near_duplicates.length > 0 && (
            <Notice kind="bad" title={`${status.cross_split_near_duplicates.length} casi duplicados entre particiones`}>
              {status.cross_split_near_duplicates.slice(0, 5).map((d: any) => `${d.a} (${d.a_split}) y ${d.b} (${d.b_split})`).join("\n")}
            </Notice>
          )}
          {status?.lora_leaks.map((l: any) => <Notice key={l.lora} kind="bad" title={`El LoRA ${l.lora} vio ${l.images} imágenes de val o test`} />)}
        </Card>
      </div>
      {confirm && (
        <Confirm title="Bloquear partición" action="Bloquear"
          body="Después no podrás cambiar el reparto ni importar más imágenes reales en este proyecto. Es lo que permite defender la prueba como independiente."
          onClose={() => setConfirm(false)}
          onConfirm={async () => { setConfirm(false); await run(() => post(`/projects/${project.id}/split/lock`), "Partición bloqueada"); reload(); refresh(); }} />
      )}
    </div>
  );
}

function CaptionsTab({ project }: { project: Project }) {
  const { data, refresh } = usePoll(() => get<{ total: number; items: ImageRow[] }>(`/projects/${project.id}/images?origin=real&split=train&limit=300`), 6000, [project.id]);
  const [token, setToken] = useState("ohwxperson");
  const [overwrite, setOverwrite] = useState(false);
  const { busy, run } = useAction();
  return (
    <div className="stack">
      <Card title="Generar descripciones" note="BLIP describe solo el split de entrenamiento. Revisa el resultado: se equivoca con frecuencia y estas frases guían al LoRA.">
        <div className="row" style={{ alignItems: "flex-end" }}>
          <Field label="Token del sujeto"><input value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 200 }} /></Field>
          <label className="row muted"><input type="checkbox" checked={overwrite} onChange={(e) => setOverwrite(e.target.checked)} /> Reemplazar las existentes</label>
          <button className="btn btn-primary" disabled={busy || !project.pipeline.by_split.train} onClick={() => run(() => post(`/projects/${project.id}/captions`, { token, overwrite }), "Descripciones en cola")}>Generar con BLIP</button>
        </div>
      </Card>
      <Card title={`${data?.total ?? 0} imágenes de entrenamiento`}>
        {data?.items.length ? (
          <div className="stack">
            {data.items.map((image) => <CaptionRow key={image.id} image={image} onSaved={refresh} />)}
          </div>
        ) : <Empty title="Sin imágenes de entrenamiento">Define la partición primero.</Empty>}
      </Card>
    </div>
  );
}

function CaptionRow({ image, onSaved }: { image: ImageRow; onSaved: () => void }) {
  const [value, setValue] = useState(image.caption ?? "");
  const [dirty, setDirty] = useState(false);
  const { run } = useAction();
  useEffect(() => { if (!dirty) setValue(image.caption ?? ""); }, [image.caption, dirty]);
  return (
    <div style={{ display: "grid", gridTemplateColumns: "72px 1fr", gap: 12, alignItems: "center" }}>
      <img src={image.thumb_url} alt="" width={72} height={72} style={{ objectFit: "cover", borderRadius: 4 }} loading="lazy" />
      <textarea
        rows={2} aria-label={`Descripción de ${image.filename}`} value={value}
        onChange={(e) => { setValue(e.target.value); setDirty(true); }}
        onBlur={async () => {
          if (!dirty) return;
          await run(() => patch(`/images/${image.id}`, { caption: value }));
          setDirty(false);
          onSaved();
        }}
      />
    </div>
  );
}

function AnalysisTab({ projectId }: { projectId: string }) {
  const { data: stats } = usePoll(() => get<any>(`/projects/${projectId}/stats`), null, [projectId]);
  const { data: features, refresh } = usePoll(() => get<any>(`/projects/${projectId}/features`), 8000, [projectId]);
  const { busy, run } = useAction();
  if (!stats) return null;
  if (!stats.count) return <Empty title="Sin datos para analizar">Importa imágenes primero.</Empty>;
  const result = features?.result;
  return (
    <div className="stack">
      <Card title="Resumen">
        <div className="stat-row">
          <Stat value={stats.count} label="Imágenes" />
          <Stat value={formatBytes(stats.bytes)} label="En disco" />
          <Stat value={stats.imbalance_ratio ?? "—"} label="Desbalance entre clases" />
          <Stat value={stats.sharpness?.p50 ?? "—"} label="Nitidez mediana" />
        </div>
      </Card>
      {stats.imbalance_ratio > 3 && <Notice kind="warn" title="Clases desbalanceadas">La clase mayoritaria tiene {stats.imbalance_ratio} veces más imágenes que la minoritaria. Considera generar más de las clases escasas.</Notice>}
      <div className="grid grid-2">
        <Card title="Resoluciones"><BarList rows={stats.resolutions} /></Card>
        <Card title="Clases"><BarList rows={stats.classes} color="var(--viz-real)" /></Card>
        <Card title="Formatos"><BarList rows={stats.formats} /></Card>
        <Card title="Posibles artefactos" note="Calculado de la imagen: borrosa, subexpuesta, sobreexpuesta o sin contraste.">
          {stats.flags.length ? <BarList rows={stats.flags} /> : <p className="muted">No se detectaron.</p>}
        </Card>
      </div>
      <Card
        title="Mapa de similitud"
        note="Embeddings CLIP proyectados a 2D con PCA. Sirve para ver si lo generado cubre la zona de lo real o queda aparte. Las distancias son orientativas."
        actions={<button className="btn" disabled={busy} onClick={() => run(() => post(`/projects/${projectId}/features`), "Cálculo en cola").then(refresh)}>Calcular</button>}
      >
        {result ? (
          <div className="stack">
            <FeatureMap points={result.points} />
            {result.semantic_near_duplicates?.length > 0 && (
              <Notice kind="warn" title={`${result.semantic_near_duplicates.length} pares casi idénticos según CLIP`}>
                {result.semantic_near_duplicates.slice(0, 5).map((d: any) => `${d.a} y ${d.b}: ${d.similarity}`).join("\n")}
              </Notice>
            )}
          </div>
        ) : <p className="muted">Aún no se ha calculado.</p>}
      </Card>
    </div>
  );
}
