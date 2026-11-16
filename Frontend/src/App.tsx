import { Suspense, useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation, Navigate } from "react-router-dom";
import { ScrollToTop } from "./components/ScrollToTop";
import { Loader2 } from "lucide-react";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { lazyWithRetry } from "@/lib/lazyWithRetry";

// Lazy Loaded Pages with deployment auto-recovery
const Index = lazyWithRetry(() => import("./pages/Index"), "Index");
const Workspace = lazyWithRetry(() => import("./pages/Workspace"), "Workspace");
const NotFound = lazyWithRetry(() => import("./pages/NotFound"), "NotFound");
const Pricing = lazyWithRetry(() => import("./pages/Pricing"), "Pricing");
const Features = lazyWithRetry(() => import("./pages/Features"), "Features");
const HowItWorks = lazyWithRetry(() => import("./pages/HowItWorks"), "HowItWorks");
const Documentation = lazyWithRetry(() => import("./pages/Documentation"), "Documentation");

const Examples = lazyWithRetry(() => import("./pages/Examples"), "Examples");
const PrivacyPolicy = lazyWithRetry(() => import("./pages/PrivacyPolicy"), "PrivacyPolicy");
const TermsOfService = lazyWithRetry(() => import("./pages/TermsOfService"), "TermsOfService");
const Contact = lazyWithRetry(() => import("./pages/Contact"), "Contact");
const Dashboard = lazyWithRetry(() => import("./pages/Dashboard"), "Dashboard");
const Analytics = lazyWithRetry(() => import("./pages/Analytics"), "Analytics");

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000,
      gcTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

const PageLoader = () => (
  <div className="min-h-screen flex flex-col items-center justify-center space-y-3 bg-background animate-in fade-in duration-300" role="status" aria-live="polite">
    <Loader2 className="h-8 w-8 text-primary animate-spin" />
    <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">Loading application...</span>
  </div>
);

const DocumentMetadata = () => {
  const location = useLocation();

  useEffect(() => {
    const labels = {
      "/": "GetReport | Data quality reports you can trust",
      "/workspace": "Workspace | GetReport",
      "/dashboard": "Audit history | GetReport",
      "/analytics": "Interactive BI Analytics | GetReport",
      "/documentation": "Documentation | GetReport",
      "/features": "Features | GetReport",
      "/pricing": "Pricing | GetReport",
      "/examples": "Examples | GetReport",
    } as const;
    const path = location.pathname;
    if (path in labels) {
      // SAFETY: 'in' operator check strictly validates that path exists in labels map keys
      document.title = labels[path as keyof typeof labels];
    } else {
      document.title = "GetReport";
    }
  }, [location.pathname]);

  return null;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <ScrollToTop />
        <DocumentMetadata />
        <ErrorBoundary fallbackTitle="Application Interface">
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/workspace" element={<Workspace />} />
              <Route path="/pricing" element={<Pricing />} />
              <Route path="/features" element={<Features />} />
              <Route path="/how-it-works" element={<HowItWorks />} />
              <Route path="/documentation" element={<Documentation />} />
              <Route path="/docs" element={<Navigate to="/documentation" replace />} />
              <Route path="/app" element={<Navigate to="/workspace" replace />} />

              <Route path="/examples" element={<Examples />} />
              <Route path="/privacy-policy" element={<PrivacyPolicy />} />
              <Route path="/terms-of-service" element={<TermsOfService />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/bi" element={<Navigate to="/analytics" replace />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
