import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { getInitialTheme, applyTheme, useTheme } from "@/hooks/useTheme";
import { ThemeToggle } from "@/components/ThemeToggle";
import { buttonVariants } from "@/components/ui/button";

describe("Theme System & useTheme", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = "";
    document.documentElement.style.colorScheme = "";

    // Ensure meta elements exist
    let metaColorScheme = document.querySelector('meta[name="color-scheme"]');
    if (!metaColorScheme) {
      metaColorScheme = document.createElement("meta");
      metaColorScheme.setAttribute("name", "color-scheme");
      document.head.appendChild(metaColorScheme);
    }
    metaColorScheme.setAttribute("content", "dark");

    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement("meta");
      metaThemeColor.setAttribute("name", "theme-color");
      document.head.appendChild(metaThemeColor);
    }
    metaThemeColor.setAttribute("content", "#07080a");
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("defaults to dark when localStorage is empty", () => {
    expect(getInitialTheme()).toBe("dark");
  });

  it("reads stored light theme from localStorage", () => {
    localStorage.setItem("theme", "light");
    expect(getInitialTheme()).toBe("light");
  });

  it("reads stored dark theme from localStorage", () => {
    localStorage.setItem("theme", "dark");
    expect(getInitialTheme()).toBe("dark");
  });

  it("applies light theme correctly to DOM and meta tags", () => {
    applyTheme("light");

    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(localStorage.getItem("theme")).toBe("light");

    const metaColorScheme = document.querySelector('meta[name="color-scheme"]');
    expect(metaColorScheme?.getAttribute("content")).toBe("light");

    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    expect(metaThemeColor?.getAttribute("content")).toBe("#F5F2EB");
  });

  it("applies dark theme correctly to DOM and meta tags", () => {
    applyTheme("dark");

    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.classList.contains("light")).toBe(false);
    expect(document.documentElement.style.colorScheme).toBe("dark");
    expect(localStorage.getItem("theme")).toBe("dark");

    const metaColorScheme = document.querySelector('meta[name="color-scheme"]');
    expect(metaColorScheme?.getAttribute("content")).toBe("dark");

    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    expect(metaThemeColor?.getAttribute("content")).toBe("#07080a");
  });

  it("renders ThemeToggle with accessible attributes and toggles theme on click", () => {
    localStorage.setItem("theme", "dark");
    applyTheme("dark");

    const TestHarness = () => {
      return <ThemeToggle />;
    };

    render(<TestHarness />);

    const toggleButton = screen.getByRole("button", { name: /switch to light mode/i });
    expect(toggleButton).toBeInTheDocument();
    expect(toggleButton).toHaveAttribute("type", "button");
    expect(toggleButton).toHaveAttribute("aria-pressed", "false");

    // Click toggle to switch to light mode
    act(() => {
      fireEvent.click(toggleButton);
    });

    expect(localStorage.getItem("theme")).toBe("light");
    expect(document.documentElement.classList.contains("light")).toBe(true);
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    // After toggle, aria-label should update
    expect(screen.getByRole("button", { name: /switch to dark mode/i })).toBeInTheDocument();
    expect(toggleButton).toHaveAttribute("aria-pressed", "true");
  });

  it("generates maroon button classes for primary, launch, and hero variants in light mode", () => {
    const defaultClasses = buttonVariants({ variant: "default" });
    expect(defaultClasses).toContain("bg-[#800000]");
    expect(defaultClasses).toContain("text-white");
    expect(defaultClasses).toContain("dark:bg-white");

    const launchClasses = buttonVariants({ variant: "launch" });
    expect(launchClasses).toContain("bg-[#800000]");
    expect(launchClasses).toContain("text-white");
    expect(launchClasses).toContain("dark:bg-white");

    const heroClasses = buttonVariants({ variant: "hero" });
    expect(heroClasses).toContain("bg-[#800000]");
    expect(heroClasses).toContain("text-white");
    expect(heroClasses).toContain("dark:bg-white");

    const primaryClasses = buttonVariants({ variant: "primary" });
    expect(primaryClasses).toContain("bg-[#800000]");
    expect(primaryClasses).toContain("text-white");
  });
});
