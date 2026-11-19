import { describe, it, expect, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { Toast, ToastProvider, ToastViewport, ToastTitle, ToastDescription } from "@/components/ui/toast";
import { applyTheme } from "@/hooks/useTheme";

describe("Toast Component Solid Opacity & Theme Architecture", () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = "";
  });

  it("renders light mode toast with 100% solid white background and opaque plate", () => {
    applyTheme("light");

    render(
      <ToastProvider>
        <Toast data-testid="toast-root" variant="default">
          <ToastTitle>File Staged</ToastTitle>
          <ToastDescription>1 file(s) added to staging.</ToastDescription>
        </Toast>
        <ToastViewport />
      </ToastProvider>
    );

    const toastElement = screen.getByTestId("toast-root");
    expect(toastElement).toBeInTheDocument();
    expect(toastElement.getAttribute("data-getreport-toast")).toBe("true");
    expect(toastElement.style.backgroundColor).toBe("rgb(255, 255, 255)");
    expect(toastElement.style.opacity).toBe("1");

    // Verify solid backing plate
    const backingPlate = toastElement.querySelector('[aria-hidden="true"]');
    expect(backingPlate).toBeInTheDocument();
    expect((backingPlate as HTMLElement).style.backgroundColor).toBe("rgb(255, 255, 255)");
    expect((backingPlate as HTMLElement).style.opacity).toBe("1");
  });

  it("renders dark mode toast with solid zinc obsidian background and opaque plate", () => {
    applyTheme("dark");

    render(
      <ToastProvider>
        <Toast data-testid="dark-toast-root" variant="default">
          <ToastTitle>File Staged</ToastTitle>
          <ToastDescription>1 file(s) added to staging.</ToastDescription>
        </Toast>
        <ToastViewport />
      </ToastProvider>
    );

    const toastElement = screen.getByTestId("dark-toast-root");
    expect(toastElement).toBeInTheDocument();
    expect(toastElement.getAttribute("data-getreport-toast")).toBe("true");
    // #18181b in rgb is rgb(24, 24, 27)
    expect(toastElement.style.backgroundColor).toBe("rgb(24, 24, 27)");
    expect(toastElement.style.opacity).toBe("1");

    const backingPlate = toastElement.querySelector('[aria-hidden="true"]');
    expect(backingPlate).toBeInTheDocument();
    expect((backingPlate as HTMLElement).style.backgroundColor).toBe("rgb(24, 24, 27)");
  });

  it("renders dark mode destructive toast with solid rose obsidian background", () => {
    applyTheme("dark");

    render(
      <ToastProvider>
        <Toast data-testid="destructive-dark-toast" variant="destructive">
          <ToastTitle>Error</ToastTitle>
          <ToastDescription>Dataset processing failed.</ToastDescription>
        </Toast>
        <ToastViewport />
      </ToastProvider>
    );

    const toastElement = screen.getByTestId("destructive-dark-toast");
    expect(toastElement).toBeInTheDocument();
    // #1c1316 in rgb is rgb(28, 19, 22)
    expect(toastElement.style.backgroundColor).toBe("rgb(28, 19, 22)");
    expect(toastElement.style.opacity).toBe("1");
  });
});
