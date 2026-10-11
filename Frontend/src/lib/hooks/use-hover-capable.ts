import { useEffect, useState } from "react";

/**
 * Returns true only on devices that have fine pointer hover (mouse / trackpad).
 * Touch devices fire phantom `:hover` on tap that sticks until tapped elsewhere.
 * Gating hover-only effects (scale lifts, glows) behind this prevents sticky states on mobile.
 */
export function useHoverCapable(): boolean {
  const [canHover, setCanHover] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(hover: hover) and (pointer: fine)");
    const update = () => setCanHover(mq.matches);
    update();
    mq.addEventListener?.("change", update);
    return () => mq.removeEventListener?.("change", update);
  }, []);

  return canHover;
}
