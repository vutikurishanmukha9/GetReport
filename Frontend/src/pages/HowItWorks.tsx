import { UploadCloud, Search, FileDown, ArrowRight, Gauge, Brain, FileCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const steps = [
    {
        id: "01",
        icon: UploadCloud,
        title: "Upload and Ingest",
        description: "Drop your CSV or Excel file. Our Polars-powered engine streams and validates data types, detecting encoding, delimiters, and schema automatically.",
        details: ["Supports CSV, XLSX, XLS", "Auto-detects encoding (UTF-8, Latin-1)", "Handles 100K+ rows efficiently"],
    },
    {
        id: "02",
        icon: Search,
        title: "Clean and Score",
        description: "Data hygiene runs automatically: duplicates removed, missing values identified, types corrected. Every column gets a confidence grade (A-F).",
        details: ["Completeness, Consistency, Validity, Stability", "Cleaning audit trail", "Two-stage pipeline: Inspect then Execute"],
    },
    {
        id: "03",
        icon: Brain,
        title: "Analyze with Transparency",
        description: "Full statistical analysis with 'Why I Did X' explanations. Domain detection, correlation analysis, outlier detection, time-series analysis—all documented.",
        details: ["10+ analysis types evaluated", "Decision log with evidence", "Semantic domain detection"],
    },
    {
        id: "04",
        icon: Gauge,
        title: "Generate Insights",
        description: "AI-powered narrative insights, ML-ready feature recommendations, and actionable next steps. Smart schema corrections and relationship discovery.",
        details: ["Feature engineering suggestions", "Encoding & scaling recommendations", "Domain-specific insights"],
    },
    {
        id: "05",
        icon: FileCheck,
        title: "Export Report",
        description: "Professional PDF with executive summary, methodology section, confidence grades, charts, and full analysis. Ready for stakeholders.",
        details: ["15+ report sections", "High-DPI visualizations", "Compliance-ready documentation"],
    },
];

const HowItWorks = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-20">
                <div className="text-center max-w-3xl mx-auto mb-20">
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-6">How It Works</h1>
                    <p className="text-xl text-muted-foreground">
                        Five transparent steps from raw data to auditable insights. Every decision explained.
                    </p>
                </div>

                <div className="max-w-4xl mx-auto">
                    {steps.map((step, index) => (
                        <div key={step.id} className="relative flex gap-8 mb-12 last:mb-0">
                            {/* Vertical Line */}
                            {index < steps.length - 1 && (
                                <div className="absolute left-8 top-20 w-0.5 h-full bg-gradient-to-b from-primary/30 to-transparent" />
                            )}

                            {/* Icon */}
                            <div className="flex-shrink-0 w-16 h-16 rounded-full bg-primary/10 border-2 border-primary/30 flex items-center justify-center z-10">
                                <step.icon className="h-8 w-8 text-primary" />
                            </div>

                            {/* Content */}
                            <div className="flex-1 pb-8">
                                <div className="flex items-baseline gap-3 mb-3">
                                    <span className="text-4xl font-bold text-muted-foreground/30 font-mono">{step.id}</span>
                                    <h3 className="text-2xl font-bold">{step.title}</h3>
                                </div>
                                <p className="text-muted-foreground leading-relaxed mb-4">{step.description}</p>
                                <ul className="flex flex-wrap gap-2">
                                    {step.details.map((detail, i) => (
                                        <li key={i} className="px-3 py-1 bg-muted rounded-full text-sm text-muted-foreground">
                                            {detail}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    ))}
                </div>

            </div>

            {/* Architecture Diagram (Conceptual) */}
            <div className="mt-24 pt-24 border-t">
                <div className="text-center mb-16">
                    <h2 className="text-3xl font-bold mb-4">Under the Hood</h2>
                    <p className="text-muted-foreground">How GetReport processes your data securely and efficiently</p>
                </div>

                <div className="max-w-5xl mx-auto grid grid-cols-1 md:grid-cols-5 gap-4 items-center text-center">
                    {/* Client */}
                    <div className="p-6 bg-card border rounded-xl relative">
                        <div className="text-sm font-semibold mb-2">Frontend</div>
                        <div className="h-10 w-10 bg-blue-100 rounded-lg mx-auto flex items-center justify-center mb-2">
                            <Search className="h-5 w-5 text-blue-600" />
                        </div>
                        <div className="text-xs text-muted-foreground">React/Vite App</div>
                    </div>

                    <ArrowRight className="hidden md:block h-6 w-6 text-muted-foreground mx-auto" />

                    {/* API */}
                    <div className="p-6 bg-card border rounded-xl relative">
                        <div className="text-sm font-semibold mb-2">API Gateway</div>
                        <div className="h-10 w-10 bg-purple-100 rounded-lg mx-auto flex items-center justify-center mb-2">
                            <Gauge className="h-5 w-5 text-purple-600" />
                        </div>
                        <div className="text-xs text-muted-foreground">FastAPI Server</div>
                    </div>

                    <ArrowRight className="hidden md:block h-6 w-6 text-muted-foreground mx-auto" />

                    {/* Worker */}
                    <div className="p-6 bg-card border rounded-xl relative">
                        <div className="text-sm font-semibold mb-2">Analysis Engine</div>
                        <div className="h-10 w-10 bg-orange-100 rounded-lg mx-auto flex items-center justify-center mb-2">
                            <Brain className="h-5 w-5 text-orange-600" />
                        </div>
                        <div className="text-xs text-muted-foreground">Celery + Polars</div>
                    </div>
                </div>
            </div>

            <div className="mt-24 text-center">
                <Link to="/">
                    <Button size="lg" className="gap-2 h-12 px-8 text-lg">
                        Try It Yourself <ArrowRight className="h-5 w-5" />
                    </Button>
                </Link>
            </div>
        </div>
        </div >
    );
};

export default HowItWorks;
