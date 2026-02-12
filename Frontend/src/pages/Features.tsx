import { BarChart3, Brain, FileText, Layout, Wand2, ShieldCheck, Gauge, HelpCircle, Zap, Target } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const features = [
    {
        icon: Gauge,
        title: "Column Confidence Scores",
        description: "Every column gets an A-F grade across four dimensions: Completeness, Consistency, Validity, and Stability. Know exactly which columns you can trust.",
        badge: "NEW",
    },
    {
        icon: HelpCircle,
        title: "Why I Did X Explanations",
        description: "Full transparency into every analysis decision. See exactly why each analysis was run or skipped, with evidence and reasoning.",
        badge: "NEW",
    },
    {
        icon: Target,
        title: "Semantic Intelligence",
        description: "Automatic domain detection (Education, Sales, Healthcare, HR, Finance) with confidence scores and column role classification.",
        badge: "NEW",
    },
    {
        icon: Wand2,
        title: "Intelligent Hygiene",
        description: "Our Polars-based engine automatically identifies and rectifies common data quality issues—missing values, duplicates, and type mismatches—at lightning speed.",
    },
    {
        icon: Brain,
        title: "Context-Aware AI",
        description: "Leveraging RAG architecture, we analyze statistical summaries to provide narrative context and strategic recommendations without exposing raw confidential rows.",
    },
    {
        icon: BarChart3,
        title: "Auto-Visualization",
        description: "Automatically selects the most effective chart types for your data distributions and correlations—heatmaps, histograms, bar charts, and boxplots.",
    },
    {
        icon: FileText,
        title: "Board-Ready Reports",
        description: "Generate comprehensive PDF reports complete with Executive Summaries, Confidence Grades, Analysis Decisions, and high-DPI visualizations.",
    },
    {
        icon: Layout,
        title: "Interactive Inspection",
        description: "Audit your data health in real-time. Preview, sort, and verify cleaning rules before committing to analysis with our two-stage pipeline.",
    },
    {
        icon: Zap,
        title: "ML-Ready Recommendations",
        description: "Get encoding suggestions (One-Hot, Label, Target), scaling recommendations (StandardScaler, MinMax, RobustScaler), and feature engineering ideas.",
    },
    {
        icon: ShieldCheck,
        title: "Privacy by Design",
        description: "Your data is processed ephemerally in volatile memory. We mask PII before any AI processing occurs. No data is stored after analysis.",
    },
];

const Features = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            {/* Hero */}
            <div className="bg-muted/30 border-b">
                <div className="container mx-auto px-4 py-20 text-center">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
                        <Zap className="h-4 w-4" />
                        Tier 1 Trust Foundation Complete
                    </div>
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-6">
                        Powerful Features for Data Pros
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
                        From raw numbers to actionable intelligence in minutes. Every decision is explainable. Every column is graded. Every action is auditable.
                    </p>
                    <Link to="/">
                        <Button size="lg" className="h-12 px-8">Start Analyzing Now</Button>
                    </Link>
                </div>
            </div>

            {/* Grid */}
            <div className="container mx-auto px-4 py-24">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className="group p-8 rounded-2xl border bg-card hover:shadow-lg transition-all duration-300 hover:-translate-y-1 relative"
                        >
                            {"badge" in feature && feature.badge && (
                                <span className="absolute top-4 right-4 px-2 py-0.5 text-xs font-bold bg-primary text-primary-foreground rounded">
                                    {feature.badge}
                                </span>
                            )}
                            <div className="h-12 w-12 rounded-lg bg-primary/10 flex items-center justify-center mb-6 group-hover:bg-primary/20 transition-colors">
                                <feature.icon className="h-6 w-6 text-primary" />
                            </div>
                            <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                            <p className="text-muted-foreground leading-relaxed">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </div>

            {/* Technical Specs */ }
    <div className="bg-muted/30 border-y py-24">
        <div className="container mx-auto px-4">
            <div className="text-center mb-16">
                <h2 className="text-3xl font-bold mb-4">built for Scale & Security</h2>
                <p className="text-muted-foreground max-w-2xl mx-auto">
                    Enterprise-grade infrastructure designed for sensitive data workloads.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 max-w-5xl mx-auto">
                <div>
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                        <ShieldCheck className="h-5 w-5 text-primary" /> Security First
                    </h3>
                    <ul className="space-y-4">
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-green-600" />
                            </div>
                            <div>
                                <p className="font-semibold">Ephemeral Processing</p>
                                <p className="text-sm text-muted-foreground">Data is processed in-memory and discarded immediately after analysis.</p>
                            </div>
                        </li>
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-green-600" />
                            </div>
                            <div>
                                <p className="font-semibold">PII Redaction</p>
                                <p className="text-sm text-muted-foreground">Automatic detection and masking of sensitive PII before AI processing.</p>
                            </div>
                        </li>
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-green-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-green-600" />
                            </div>
                            <div>
                                <p className="font-semibold">Encryption</p>
                                <p className="text-sm text-muted-foreground">TLS 1.3 encryption in transit. No data at rest persistence.</p>
                            </div>
                        </li>
                    </ul>
                </div>

                <div>
                    <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                        <Gauge className="h-5 w-5 text-primary" /> Performance
                    </h3>
                    <ul className="space-y-4">
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-blue-600" />
                            </div>
                            <div>
                                <p className="font-semibold">Polars Engine</p>
                                <p className="text-sm text-muted-foreground">Multi-threaded data processing for 100k+ row datasets in seconds.</p>
                            </div>
                        </li>
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-blue-600" />
                            </div>
                            <div>
                                <p className="font-semibold">Async Architecture</p>
                                <p className="text-sm text-muted-foreground">Non-blocking Celery workers handle heavy compute tasks.</p>
                            </div>
                        </li>
                        <li className="flex gap-3">
                            <div className="h-6 w-6 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                <div className="h-2 w-2 rounded-full bg-blue-600" />
                            </div>
                            <div>
                                <p className="font-semibold">Smart Sampling</p>
                                <p className="text-sm text-muted-foreground">Representative sampling for large-scale AI insights without context overflow.</p>
                            </div>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    {/* CTA */ }
    <div className="container mx-auto px-4 py-24 text-center">
        <h2 className="text-3xl font-bold mb-6">Ready to transform your data?</h2>
        <Link to="/">
            <Button size="lg" className="h-12 px-8">Start Your Analysis</Button>
        </Link>
    </div>

    export default Features;
