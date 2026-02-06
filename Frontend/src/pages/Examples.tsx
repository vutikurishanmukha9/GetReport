import { BarChart, FileSpreadsheet, GraduationCap, Users, DollarSign, Heart, Gauge, Brain } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";

const examples = [
    {
        icon: BarChart,
        title: "Sales Analysis",
        description: "Retail sales performance Q1-Q4",
        domain: "Sales",
        features: ["Trend detection", "Seasonality analysis", "Revenue correlation"],
    },
    {
        icon: GraduationCap,
        title: "Student Performance",
        description: "Academic grades and attendance data",
        domain: "Education",
        features: ["Score distributions", "Attendance impact", "Grade predictions"],
    },
    {
        icon: Users,
        title: "Customer Churn",
        description: "SaaS metrics and user retention",
        domain: "Sales",
        features: ["Churn indicators", "Engagement scores", "Cohort analysis"],
    },
    {
        icon: DollarSign,
        title: "Financial Report",
        description: "Expense tracking and budget analysis",
        domain: "Finance",
        features: ["Budget variance", "Category breakdown", "Trend forecasting"],
    },
    {
        icon: Heart,
        title: "Health Metrics",
        description: "Patient vitals and treatment outcomes",
        domain: "Healthcare",
        features: ["Vital correlations", "Outcome analysis", "Risk stratification"],
    },
    {
        icon: Users,
        title: "HR Analytics",
        description: "Employee performance and satisfaction",
        domain: "HR",
        features: ["Tenure analysis", "Satisfaction drivers", "Turnover risk"],
    },
];

const Examples = () => {
    return (
        <div className="min-h-screen bg-background animate-in fade-in duration-500">
            <div className="container mx-auto px-4 py-16">
                <div className="text-center mb-16">
                    <h1 className="text-4xl font-bold mb-4">Example Reports</h1>
                    <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                        See what GetReport can do across different industries and use cases.
                        Our semantic intelligence automatically detects your data domain.
                    </p>
                </div>

                {/* Features highlight */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto mb-16">
                    <div className="p-4 border rounded-lg flex items-center gap-3">
                        <Gauge className="h-8 w-8 text-primary" />
                        <div>
                            <h4 className="font-semibold">Confidence Grades</h4>
                            <p className="text-sm text-muted-foreground">A-F scores for every column</p>
                        </div>
                    </div>
                    <div className="p-4 border rounded-lg flex items-center gap-3">
                        <Brain className="h-8 w-8 text-primary" />
                        <div>
                            <h4 className="font-semibold">Why I Did X</h4>
                            <p className="text-sm text-muted-foreground">Full decision transparency</p>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
                    {examples.map((example, index) => (
                        <Card key={index} className="hover:shadow-md transition-shadow group">
                            <CardHeader>
                                <div className="h-32 bg-muted rounded-md mb-4 flex items-center justify-center group-hover:bg-primary/5 transition-colors">
                                    <example.icon className="h-12 w-12 text-muted-foreground group-hover:text-primary transition-colors" />
                                </div>
                                <div className="flex items-center gap-2 mb-2">
                                    <span className="text-xs bg-primary/10 text-primary px-2 py-0.5 rounded">
                                        {example.domain}
                                    </span>
                                </div>
                                <CardTitle>{example.title}</CardTitle>
                                <CardDescription>{example.description}</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="flex flex-wrap gap-1">
                                    {example.features.map((feature, i) => (
                                        <span key={i} className="text-xs bg-muted px-2 py-1 rounded">
                                            {feature}
                                        </span>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    ))}
                </div>

                <div className="mt-16 text-center">
                    <p className="text-muted-foreground mb-4">Ready to analyze your own data?</p>
                    <Link to="/">
                        <Button size="lg">Create Your Own Report</Button>
                    </Link>
                </div>
            </div>
        </div>
    );
};

export default Examples;
