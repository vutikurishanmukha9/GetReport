import { Sun, Moon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";

interface ThemeToggleProps {
  className?: string;
  variant?: "ghost" | "outline";
}

export const ThemeToggle = ({ className = "", variant = "outline" }: ThemeToggleProps) => {
  const { theme, toggleTheme } = useTheme();
  const isLight = theme === "light";

  return (
    <Button
      type="button"
      variant={variant}
      size="icon"
      onClick={toggleTheme}
      className={`relative h-9 w-9 rounded-xl transition-all duration-200 cursor-pointer border border-stone-300 dark:border-white/10 bg-[#FAF8F5] dark:bg-[#16171b] hover:bg-stone-200/80 dark:hover:bg-[#1f2230] text-stone-800 dark:text-stone-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#800000] dark:focus-visible:ring-white/40 focus-visible:ring-offset-1 ${className}`}
      aria-label={isLight ? "Switch to dark mode" : "Switch to light mode"}
      aria-pressed={isLight}
      title={isLight ? "Switch to dark mode" : "Switch to light mode"}
    >
      <Sun className="h-4 w-4 rotate-0 scale-100 transition-transform duration-300 dark:-rotate-90 dark:scale-0 text-amber-700 dark:text-amber-400" />
      <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-transform duration-300 dark:rotate-0 dark:scale-100 text-stone-200" />
      <span className="sr-only">{isLight ? "Switch to dark mode" : "Switch to light mode"}</span>
    </Button>
  );
};
