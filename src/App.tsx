import { useEffect, useState } from "react";
import { fetchHealth } from "./api";
import StatusBar from "./components/StatusBar";
import StudioSection from "./components/StudioSection";
import { DatasetSection } from "./components/Dataset";
import { TrainingSection } from "./components/Training";

type BackendState = "checking" | "online" | "offline";
type Section = "studio" | "training" | "dataset";

function detectTheme(): "light" | "dark" {
  const saved = localStorage.getItem("af_theme");
  if (saved === "light" || saved === "dark") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [theme, setTheme] = useState<"light" | "dark">(detectTheme);
  const [activeSection, setActiveSection] = useState<Section>("studio");
  const [backendState, setBackendState] = useState<BackendState>("checking");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("af_theme", theme);
  }, [theme]);

  useEffect(() => {
    fetchHealth()
      .then(() => setBackendState("online"))
      .catch(() => setBackendState("offline"));
  }, []);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  return (
    <div className="app">
      <StatusBar
        backendState={backendState}
        theme={theme}
        onToggleTheme={toggleTheme}
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      {activeSection === "studio"   && <StudioSection />}
      {activeSection === "training" && <TrainingSection />}
      {activeSection === "dataset"  && <DatasetSection />}
    </div>
  );
}
