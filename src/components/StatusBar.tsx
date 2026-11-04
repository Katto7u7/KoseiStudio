type BackendState = "checking" | "online" | "offline";
type Section = "studio" | "training" | "dataset";

interface Props {
  backendState: BackendState;
  theme: "light" | "dark";
  onToggleTheme: () => void;
  activeSection: Section;
  onSectionChange: (s: Section) => void;
}

export default function StatusBar({ backendState, theme, onToggleTheme, activeSection, onSectionChange }: Props) {
  const statusLabel =
    backendState === "online"   ? "Backend en línea" :
    backendState === "offline"  ? "Backend fuera de línea" :
    "Verificando";

  return (
    <header className="topbar">
      <div className="topbar-row">
        <div className="topbar-left">
          <div className="brand">
            <span className="brand-title">KoseiStudio</span>
            <span className="brand-badge">beta</span>
          </div>
        </div>
        <div className="topbar-right">
          <div className={`status-pill ${backendState}`}>
            <span className="status-dot" />
            {statusLabel}
          </div>
          <button
            className="theme-toggle"
            onClick={onToggleTheme}
            title={theme === "dark" ? "Cambiar a tema claro" : "Cambiar a tema oscuro"}
            type="button"
            aria-label="Alternar tema"
          >
            {theme === "dark" ? "☀" : "☽"}
          </button>
        </div>
      </div>

      <nav className="tabs">
        <button
          className={`tab${activeSection === "studio" ? " active" : ""}`}
          onClick={() => onSectionChange("studio")}
          type="button"
        >
          ◈ Studio
        </button>
        <button
          className={`tab${activeSection === "training" ? " active" : ""}`}
          onClick={() => onSectionChange("training")}
          type="button"
        >
          ⬡ Training
        </button>
        <button
          className={`tab${activeSection === "dataset" ? " active" : ""}`}
          onClick={() => onSectionChange("dataset")}
          type="button"
        >
          ◫ Dataset
        </button>
      </nav>
    </header>
  );
}
