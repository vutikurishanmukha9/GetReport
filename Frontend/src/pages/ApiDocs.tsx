import { Server, Terminal } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const ApiDocs = () => {
    return (
        <div className="min-h-[80vh] flex flex-col items-center justify-center text-center p-4 animate-in fade-in duration-500">
            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center mb-8">
                <Terminal className="h-10 w-10 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight mb-4">API Reference</h1>
            <p className="text-xl text-muted-foreground max-w-md mb-8">
                Integrate GetReport directly into your own applications. Full REST API documentation is coming soon.
            </p>

            <div className="text-left bg-muted p-4 rounded-lg font-mono text-sm mb-8 w-full max-w-md">
                <div className="flex items-center gap-2 text-muted-foreground mb-2 border-b pb-2">
                    <Server className="h-3 w-3" /> api.getreport.com
                </div>
                <div className="space-y-1">
                    <p><span className="text-green-600">GET</span> /v1/status</p>
                    <p><span className="text-blue-600">POST</span> /v1/upload</p>
                    <p><span className="text-blue-600">POST</span> /v1/generate</p>
                </div>
            </div>

            <Link to="/">
                <Button variant="outline">Return to Home</Button>
            </Link>
        </div>
    );
};

export default ApiDocs;
