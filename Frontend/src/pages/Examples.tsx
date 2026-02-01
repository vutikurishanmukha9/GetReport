import { BarChart, FileSpreadsheet } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

const Examples = () => {
    return (
        <div className="container mx-auto px-4 py-16 animate-in fade-in duration-500">
            <div className="text-center mb-16">
                <h1 className="text-4xl font-bold mb-4">Example Reports</h1>
                <p className="text-xl text-muted-foreground">See what's possible with GetReport</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-5xl mx-auto">
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                    <CardHeader>
                        <div className="h-40 bg-muted rounded-md mb-4 flex items-center justify-center">
                            <BarChart className="h-10 w-10 text-muted-foreground" />
                        </div>
                        <CardTitle>Sales Analysis</CardTitle>
                        <CardDescription>Retail sales performance Q1-Q4</CardDescription>
                    </CardHeader>
                </Card>
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                    <CardHeader>
                        <div className="h-40 bg-muted rounded-md mb-4 flex items-center justify-center">
                            <FileSpreadsheet className="h-10 w-10 text-muted-foreground" />
                        </div>
                        <CardTitle>Customer Churn</CardTitle>
                        <CardDescription>SaaS metrics and user retention</CardDescription>
                    </CardHeader>
                </Card>
                <Card className="hover:shadow-md transition-shadow cursor-pointer">
                    <CardHeader>
                        <div className="h-40 bg-muted rounded-md mb-4 flex items-center justify-center">
                            <BarChart className="h-10 w-10 text-muted-foreground" />
                        </div>
                        <CardTitle>Marketing ROI</CardTitle>
                        <CardDescription>Campaign performance analysis</CardDescription>
                    </CardHeader>
                </Card>
            </div>

            <div className="mt-16 text-center">
                <Link to="/">
                    <Button size="lg">Create Your Own Report</Button>
                </Link>
            </div>
        </div>
    );
};

export default Examples;
