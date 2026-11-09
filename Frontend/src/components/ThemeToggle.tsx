import React from "react";
import { Sun, Moon, Laptop } from "lucide-react";
import { useTheme } from "@/components/theme-provider";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface ThemeToggleProps {
  variant?: "pill" | "icon" | "dropdown";
  className?: string;
}

export function ThemeToggle({ variant = "pill", className = "" }: ThemeToggleProps) {
  const { theme, resolvedTheme, setTheme, toggleTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  if (variant === "dropdown") {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className={`inline-flex items-center justify-center h-9 w-9 rounded-xl border border-border/80 bg-background/80 hover:bg-muted text-foreground transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 select-none cursor-pointer ${className}`}
            aria-label="Select color theme"
          >
            {isDark ? (
              <Moon className="h-4 w-4 text-zinc-200 transition-transform duration-200 hover:rotate-12" />
            ) : (
              <Sun className="h-4 w-4 text-amber-500 transition-transform duration-200 hover:rotate-45" />
            )}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-36 rounded-xl p-1 shadow-lg border border-border bg-popover text-popover-foreground">
          <DropdownMenuItem
            onClick={() => setTheme("light")}
            className={`flex items-center gap-2 px-2.5 py-1.5 text-xs rounded-lg cursor-pointer ${
              theme === "light" ? "bg-accent font-semibold" : ""
            }`}
          >
            <Sun className="h-3.5 w-3.5 text-amber-500" />
            <span>Light</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setTheme("dark")}
            className={`flex items-center gap-2 px-2.5 py-1.5 text-xs rounded-lg cursor-pointer ${
              theme === "dark" ? "bg-accent font-semibold" : ""
            }`}
          >
            <Moon className="h-3.5 w-3.5 text-zinc-300" />
            <span>Dark</span>
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => setTheme("system")}
            className={`flex items-center gap-2 px-2.5 py-1.5 text-xs rounded-lg cursor-pointer ${
              theme === "system" ? "bg-accent font-semibold" : ""
            }`}
          >
            <Laptop className="h-3.5 w-3.5 text-muted-foreground" />
            <span>System</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  // Fast 1-click toggle with tooltip & subtle kinetic icon rotation
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={toggleTheme}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              toggleTheme();
            }
          }}
          className={`relative inline-flex items-center justify-center h-8.5 w-8.5 rounded-xl border border-zinc-200/90 dark:border-white/10 bg-white/90 dark:bg-[#16171b] hover:bg-zinc-100 dark:hover:bg-[#1a1d29] text-zinc-700 dark:text-zinc-200 shadow-2xs transition-all duration-200 hover:scale-105 active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 select-none cursor-pointer ${className}`}
          aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          <span className="sr-only">Toggle theme</span>
          <Sun className="h-4 w-4 rotate-0 scale-100 transition-all duration-300 dark:-rotate-90 dark:scale-0 text-amber-500" />
          <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all duration-300 dark:rotate-0 dark:scale-100 text-zinc-200" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="text-[11px] font-sans px-2.5 py-1 rounded-md shadow-md">
        {isDark ? "Switch to Light Mode" : "Switch to Dark Mode"}
      </TooltipContent>
    </Tooltip>
  );
}
export default ThemeToggle;

