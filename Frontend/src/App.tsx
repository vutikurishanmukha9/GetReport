import { lazy, Suspense, useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import { ScrollToTop } from "./components/ScrollToTop";
import { Loader2 } from "lucide-react";

// Lazy Loaded Pages
const Index = lazy(() => import("./pages/Index"));
const Workspace = lazy(() => import("./pages/Workspace"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Pricing = lazy(() => import("./pages/Pricing"));
const Features = lazy(() => import("./pages/Features"));
const HowItWorks = lazy(() => import("./pages/HowItWorks"));
const Documentation = lazy(() => import("./pages/Documentation"));

const Examples = lazy(() => import("./pages/Examples"));
const PrivacyPolicy = lazy(() => import("./pages/PrivacyPolicy"));
const TermsOfService = lazy(() => import("./pages/TermsOfService"));
const Contact = lazy(() => import("./pages/Contact"));
const Dashboard = lazy(() => import("./pages/Dashboard"));

const queryClient = new QueryClient();

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
        <Suspense fallback={<PageLoader />}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/workspace" element={<Workspace />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/features" element={<Features />} />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/documentation" element={<Documentation />} />

            <Route path="/examples" element={<Examples />} />
            <Route path="/privacy-policy" element={<PrivacyPolicy />} />
            <Route path="/terms-of-service" element={<TermsOfService />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/dashboard" element={<Dashboard />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
