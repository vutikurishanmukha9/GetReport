import { useState, useEffect } from "react";

type Theme = "light";

export const useTheme = () => {
  const [theme] = useState<Theme>("light");

  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove("light", "dark");
    root.classList.add("light");
    localStorage.setItem("theme", "light");
  }, []);

  const toggleTheme = () => {
    // No-op
  };

  return { theme, setTheme: () => {}, toggleTheme };
};
