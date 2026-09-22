import { useEffect, useLayoutEffect } from "react";
import { useLocation } from "react-router-dom";

export const ScrollToTop = () => {
  const { pathname, search, hash } = useLocation();

  // Prevent browser from automatically restoring bottom scroll positions on SPA route changes
  useEffect(() => {
    if ("scrollRestoration" in window.history) {
      window.history.scrollRestoration = "manual";
    }
  }, []);

  // Handle route change: scroll to top immediately and guarantee top after Suspense resolves
  useLayoutEffect(() => {
    if (!hash) {
      const resetScroll = () => {
        window.scrollTo({ top: 0, left: 0, behavior: "instant" });
        document.documentElement.scrollTop = 0;
        document.body.scrollTop = 0;
      };

      // 1. Immediate reset
      resetScroll();

      // 2. Next animation frame (ensures DOM transition tick)
      const rafId = requestAnimationFrame(resetScroll);

      // 3. Short fallback timer to handle async Suspense lazy-loaded page mounting
      const timer = setTimeout(resetScroll, 50);

      return () => {
        cancelAnimationFrame(rafId);
        clearTimeout(timer);
      };
    } else {
      // Hash-based target (e.g. /documentation#security)
      const scrollToHash = () => {
        const id = hash.replace("#", "");
        const element = document.getElementById(id) || document.querySelector(hash);
        if (element) {
          element.scrollIntoView({ behavior: "smooth" });
          return true;
        }
        return false;
      };

      if (!scrollToHash()) {
        const timer = setTimeout(scrollToHash, 80);
        return () => clearTimeout(timer);
      }
    }
  }, [pathname, search, hash]);

  return null;
};

