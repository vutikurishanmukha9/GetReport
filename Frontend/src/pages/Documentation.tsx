import { BookOpen, Construction } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Documentation = () => {
    return (
        <div className="min-h-[80vh] flex flex-col items-center justify-center text-center p-4 animate-in fade-in duration-500">
            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center mb-8">
                <BookOpen className="h-10 w-10 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight mb-4">Documentation</h1>
            <p className="text-xl text-muted-foreground max-w-md mb-8">
                We are crafting comprehensive guides to help you get the most out of GetReport. Check back soon!
            </p>
            <div className="flex items-center gap-2 text-sm text-amber-600 bg-amber-50 dark:bg-amber-900/20 px-4 py-2 rounded-full mb-8">
                <Construction className="h-4 w-4" />
                <span>Work In Progress</span>
            </div>
            <Link to="/">
                <Button variant="outline">Return to Home</Button>
            </Link>
        </div>
    );
};

export default Documentation;
