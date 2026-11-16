import { lazy, ComponentType } from "react";

/**
 * Enhanced React.lazy wrapper that automatically catches chunk loading errors
 * (e.g., when a new deployment on Vercel invalidates old asset hashes)
 * and forces a one-time clean page reload to fetch the latest assets.
 */
export function lazyWithRetry<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T } | Record<string, any>>,
  exportName = "default"
) {
  return lazy(async () => {
    const reloadKey = `chunk_reload_${exportName}`;
    const hasReloaded = sessionStorage.getItem(reloadKey);

    try {
      const module = await factory();
      sessionStorage.removeItem(reloadKey);

      if (exportName in module && exportName !== "default") {
        return { default: module[exportName] as T };
      }
      if ("default" in module) {
        return { default: module.default as T };
      }
      return module as { default: T };
    } catch (error: any) {
      const message = String(error?.message || error || "");
      const isChunkError =
        message.includes("dynamically imported module") ||
        message.includes("Failed to fetch") ||
        message.includes("error loading dynamically imported module") ||
        message.includes("Importing a module script failed");

      if (isChunkError && !hasReloaded) {
        console.warn(`[ChunkRecovery] Auto-reloading to fetch newest version for ${exportName}...`);
        sessionStorage.setItem(reloadKey, "true");
        window.location.reload();
        // Return a pending promise to avoid unmounting error state before reload occurs
        return new Promise<{ default: T }>(() => {});
      }

      throw error;
    }
  });
}
