import { useState, useEffect, useCallback } from "react";

export type Theme = "light" | "dark";

const THEME_STORAGE_KEY = "theme";
const THEME_CHANGE_EVENT = "getreport-theme-change";

export function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "dark";
  } catch {
    return "dark";
  }
}

export function applyTheme(theme: Theme): void {
  try {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add(theme);
    root.style.colorScheme = theme;

    const metaColorScheme = document.querySelector('meta[name="color-scheme"]');
    if (metaColorScheme) {
      metaColorScheme.setAttribute("content", theme);
    }

    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.setAttribute(
        "content",
        theme === "light" ? "#F5F2EB" : "#07080a"
      );
    }

    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Ignore environments where DOM or storage is unavailable
  }
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);

    const handleStorage = (event: StorageEvent) => {
      if (event.key === THEME_STORAGE_KEY && event.newValue) {
        if (event.newValue === "light" || event.newValue === "dark") {
          setThemeState(event.newValue);
          applyTheme(event.newValue);
        }
      }
    };

    const handleCustomChange = (event: Event) => {
      // SAFETY: getreport-theme-change custom event is dispatched with CustomEvent<Theme> payload
      const customEvent = event as CustomEvent<Theme>;
      if (customEvent.detail === "light" || customEvent.detail === "dark") {
        setThemeState(customEvent.detail);
        applyTheme(customEvent.detail);
      }
    };

    window.addEventListener("storage", handleStorage);
    window.addEventListener(THEME_CHANGE_EVENT, handleCustomChange);

    return () => {
      window.removeEventListener("storage", handleStorage);
      window.removeEventListener(THEME_CHANGE_EVENT, handleCustomChange);
    };
  }, [theme]);

  const setTheme = useCallback((nextTheme: Theme) => {
    setThemeState(nextTheme);
    applyTheme(nextTheme);
    try {
      window.dispatchEvent(
        new CustomEvent<Theme>(THEME_CHANGE_EVENT, { detail: nextTheme })
      );
    } catch {
      // Ignore in headless environments
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState((current) => {
      const nextTheme: Theme = current === "light" ? "dark" : "light";
      applyTheme(nextTheme);
      try {
        window.dispatchEvent(
          new CustomEvent<Theme>(THEME_CHANGE_EVENT, { detail: nextTheme })
        );
      } catch {
        // Ignore in headless environments
      }
      return nextTheme;
    });
  }, []);

  return { theme, toggleTheme, setTheme };
}
