import { BarChart3, Brain, FileText, Layout, Wand2, ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const features = [
    {
        icon: Wand2,
        title: "Automated Cleaning",
        description: "Instantly detect and fix missing values, duplicates, and inconsistent data types without writing a single line of code.",
    },
    {
        icon: Brain,
        title: "AI Insights",
        description: "Leverage advanced LLMs to generate narrative insights that explain the 'why' behind your data trends.",
    },
    {
        icon: BarChart3,
        title: "Smart Visualizations",
        description: "Automatically generate relevance-ranked charts including distributions, correlations, and trend analysis.",
    },
    {
        icon: FileText,
        title: "PDF Reporting",
        description: "Download board-ready PDF reports complete with executive summaries, methodology, and detailed findings.",
    },
    {
        icon: Layout,
        title: "Interactive Preview",
        description: "Explore your dataset in real-time with our interactive data grid and quality dashboard before generating reports.",
    },
    {
        icon: ShieldCheck,
        title: "Secure Processing",
        description: "Your data is processed securely and is never used to train our models. We prioritize your data privacy.",
    },
];

const Features = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            {/* Hero */}
            <div className="bg-muted/30 border-b">
                <div className="container mx-auto px-4 py-20 text-center">
                    <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-6">
                        Powerful Features for Data Pros
                    </h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10">
                        From raw numbers to actionable intelligence in minutes. See what makes GetReport the fastest way to understand your data.
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
                            className="group p-8 rounded-2xl border bg-card hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
                        >
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
    );
};

export default Features;
