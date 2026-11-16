import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

// Auto-recover from dynamic module preload errors caused by new deployments
window.addEventListener("vite:preloadError", (event) => {
  console.warn("Vite dynamic chunk preload failed (new deployment detected). Reloading page...", event);
  const reloadKey = "chunk_preload_reload_attempted";
  if (!sessionStorage.getItem(reloadKey)) {
    sessionStorage.setItem(reloadKey, "true");
    window.location.reload();
  }
});

// Clear reload attempt flag on clean load
window.addEventListener("load", () => {
  sessionStorage.removeItem("chunk_preload_reload_attempted");
});

createRoot(document.getElementById("root")!).render(<App />);
