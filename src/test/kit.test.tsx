import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, ImageRow, Project, get, post } from "../lib/api";
import { BarList, ConditionDots, LossChart } from "../ui/charts";
import { Confirm, Progress, StatusPill, Thumb, ToastProvider, useAction } from "../ui/kit";
import ProjectsPage from "../pages/ProjectsPage";
import QualityPage from "../pages/QualityPage";
import { STAGES, flowProgress } from "../ui/flow";

afterEach(() => vi.restoreAllMocks());

function mockFetch(routes: Record<string, unknown>) {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const key = `${init?.method ?? "GET"} ${url}`;
    if (!(key in routes)) return new Response(JSON.stringify({ detail: `sin ruta ${key}` }), { status: 404, headers: { "content-type": "application/json" } });
    return new Response(JSON.stringify(routes[key]), { status: 200, headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

const image: ImageRow = {
  id: "1", origin: "synthetic", filename: "a.png", width: 512, height: 512, format: "PNG", split: "train", status: "needs_review",
  class_id: 1, class_name: "persona", class_source: null, caption: null, sha256: "x", generation: null, qc: null,
  annotation_count: 2, top_confidence: 0.91, thumb_url: "/t", file_url: "/f",
};

const pipeline = {
  real_images: 3, synthetic_images: 1, by_status: { real: {}, synthetic: {} }, by_split: { train: 0, val: 0, test: 0, none: 3 },
  split_locked: false, train_captions: 0, classes: 0, loras_ready: 0, quality_reports: 0, builds_ready: 0, experiments_ready: 0,
};

describe("api", () => {
  it("expone el mensaje del backend como ApiError", async () => {
    mockFetch({});
    await expect(get("/x")).rejects.toThrow("sin ruta GET /api/x");
    await expect(get("/x")).rejects.toBeInstanceOf(ApiError);
  });
  it("traduce los errores de validacion de FastAPI", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: [{ loc: ["body", "name"], msg: "requerido" }] }), { status: 422, headers: { "content-type": "application/json" } })));
    await expect(post("/x", {})).rejects.toThrow("name: requerido");
  });
});

describe("piezas", () => {
  it("la miniatura muestra clase y confianza del etiquetador", () => {
    render(<Thumb image={image} />);
    expect(screen.getByText("persona 0.91")).toBeInTheDocument();
    expect(screen.getByRole("button").className).toContain("origin-synthetic");
  });
  it("sin anotaciones no muestra etiqueta", () => {
    render(<Thumb image={{ ...image, annotation_count: 0, top_confidence: null }} />);
    expect(screen.queryByText(/persona/)).toBeNull();
  });
  it("la barra sin total es indeterminada y no inventa porcentaje", () => {
    const { container } = render(<Progress value={3} total={0} />);
    expect(container.querySelector(".indeterminate")).toBeTruthy();
    render(<Progress value={3} total={10} />);
    expect(screen.getAllByRole("progressbar")[1]).toHaveAttribute("aria-valuenow", "30");
  });
  it("traduce el estado", () => {
    render(<StatusPill status="no_detection" />);
    expect(screen.getByText("Sin detección")).toBeInTheDocument();
  });
  it("confirmar pide accion explicita y Escape cancela", async () => {
    const yes = vi.fn(), no = vi.fn();
    render(<Confirm title="Borrar" body="seguro" action="Eliminar" danger onConfirm={yes} onClose={no} />);
    await userEvent.keyboard("{Escape}");
    expect(no).toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Eliminar" }));
    expect(yes).toHaveBeenCalled();
  });
  it("useAction muestra el error sin romper la pantalla", async () => {
    function Sample() {
      const { run } = useAction();
      return <button onClick={() => run(async () => { throw new Error("fallo claro"); })}>ir</button>;
    }
    render(<ToastProvider><Sample /></ToastProvider>);
    await userEvent.click(screen.getByText("ir"));
    expect(await screen.findByText("fallo claro")).toBeInTheDocument();
  });
});

describe("graficas", () => {
  it("BarList pinta filas reales", () => {
    render(<BarList rows={[["512x512", 4], ["768x768", 2]]} />);
    expect(screen.getByText("512x512")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });
  it("LossChart no dibuja curva sin datos", () => {
    render(<LossChart points={[]} />);
    expect(screen.getByText(/primeros pasos/)).toBeInTheDocument();
  });
  it("LossChart resume el ultimo valor para lectores de pantalla", () => {
    render(<LossChart points={[{ step: 1, loss: 0.2 }, { step: 2, loss: 0.1 }]} />);
    expect(screen.getByRole("img")).toHaveAttribute("aria-label", expect.stringContaining("0.1000"));
  });
  it("ConditionDots muestra la media por condicion", () => {
    render(<ConditionDots label="x" rows={[{ name: "A", values: [0.5, 0.6], mean: 0.55 }]} />);
    expect(screen.getByText("0.550")).toBeInTheDocument();
  });
});

describe("riel de etapas", () => {
  it("cada etapa se calcula de los datos", () => {
    const p = { ...pipeline, split_locked: true, loras_ready: 1 };
    const done = STAGES.filter((s) => s.done(p)).map((s) => s.path);
    expect(done).toEqual(["dataset", "lora", "generate"]);
  });
});

describe("paginas", () => {
  it("proyectos: estado vacio y creacion por API", async () => {
    const project: Partial<Project> = { id: "abc", name: "Nuevo", description: "", pipeline, updated_at: 1 };
    const fetchMock = mockFetch({ "GET /api/projects": [], "POST /api/projects": project });
    render(<ToastProvider><MemoryRouter><ProjectsPage /></MemoryRouter></ToastProvider>);
    expect(await screen.findByText("Crea tu primer proyecto")).toBeInTheDocument();
    const create = screen.getByRole("button", { name: "Crear proyecto" });
    expect(create).toBeDisabled();
    await userEvent.type(screen.getByPlaceholderText("Detección de personas"), "Nuevo");
    await userEvent.click(create);
    await waitFor(() => expect(fetchMock.mock.calls.some(([, i]) => i?.method === "POST")).toBe(true));
    const [, init] = fetchMock.mock.calls.find(([, i]) => i?.method === "POST")!;
    expect(JSON.parse(init!.body as string).name).toBe("Nuevo");
  });
  it("proyectos: muestra el error de carga", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("", { status: 502 })));
    render(<ToastProvider><MemoryRouter><ProjectsPage /></MemoryRouter></ToastProvider>);
    expect(await screen.findByText("No se pudo cargar la lista")).toBeInTheDocument();
  });
  it("calidad: explica limites y avisos, no solo un numero", async () => {
    mockFetch({ "GET /api/projects/p/quality": [{
      id: "r", status: "ready", job_status: "succeeded", job_error: null, created_at: 1,
      report: {
        samples: { real_train: 20, synthetic: 30 }, warnings: ["FID no es fiable"],
        inception: { fid: 90.5, kid_mean: 0.1, kid_std: 0.01, kid_subset_size: 20 },
        precision_recall: { precision: 0.5, recall: 0.4 }, diversity: { synthetic: 0.2, real: 0.3 },
        similarity_to_real: { p50: 0.8, memorized: 0 }, duplicates: { near_duplicates: 0, rate: 0 }, per_class: {},
        visual: { sharpness_synthetic: { p50: 1 }, sharpness_real: { p50: 1 }, brightness_synthetic: { p50: 1 }, brightness_real: { p50: 1 }, artifact_flags: {} },
        labeling: { statuses: { approved: 3 }, no_detection_rate: 0, box_area_synthetic: {}, box_area_real: {} }, omitted: { inception_score: "No aplica." },
      },
    }] });
    render(<ToastProvider><QualityPage projectId="p" /></ToastProvider>);
    expect(await screen.findByText("90.5")).toBeInTheDocument();
    expect(screen.getByText("FID no es fiable")).toBeInTheDocument();
    expect(screen.getByText(/muy sesgado/)).toBeInTheDocument();
    expect(screen.getByText(/Inception Score omitido/)).toBeInTheDocument();
  });
});
