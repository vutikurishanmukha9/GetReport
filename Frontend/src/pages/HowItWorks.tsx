import { UploadCloud, Search, FileDown, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const steps = [
    {
        id: "01",
        icon: UploadCloud,
        title: "Upload Your Data",
        description: "Drag and drop your CSV or Excel file. We support datasets of all sizes and automatically detect format and structure.",
    },
    {
        id: "02",
        icon: Search,
        title: "Instant Analysis",
        description: "Our engine cleans your data, runs statistical tests, detects patterns, and generates visualizations in seconds.",
    },
    {
        id: "03",
        icon: FileDown,
        title: "Get Your Report",
        description: "Download a professional PDF report containing all findings, charts, and AI-driven recommendations ready for presentation.",
    },
];

const HowItWorks = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-20">
                <div className="text-center max-w-3xl mx-auto mb-20">
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-6">How It Works</h1>
                    <p className="text-xl text-muted-foreground">
                        Three simple steps to go from raw data to comprehensive insights.
                    </p>
                </div>

                <div className="relative max-w-5xl mx-auto">
                    {/* Connecting Line (Desktop) */}
                    <div className="hidden md:block absolute top-24 left-0 w-full h-0.5 bg-gradient-to-r from-muted via-primary/20 to-muted" />

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative z-10">
                        {steps.map((step) => (
                            <div key={step.id} className="flex flex-col items-center text-center">
                                <div className="w-16 h-16 rounded-full bg-background border-4 border-primary/10 flex items-center justify-center mb-6 shadow-sm">
                                    <step.icon className="h-8 w-8 text-primary" />
                                </div>
                                <span className="text-6xl font-bold text-muted/20 mb-4 font-mono">{step.id}</span>
                                <h3 className="text-2xl font-bold mb-4">{step.title}</h3>
                                <p className="text-muted-foreground leading-relaxed">
                                    {step.description}
                                </p>
                            </div>
                        ))}
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
        </div>
    );
};

export default HowItWorks;
