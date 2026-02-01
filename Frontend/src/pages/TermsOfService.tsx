import { Scale } from "lucide-react";

const TermsOfService = () => {
    return (
        <div className="container mx-auto px-4 py-16 max-w-4xl animate-in fade-in duration-500">
            <div className="mb-12 border-b pb-8">
                <div className="flex items-center gap-4 mb-4">
                    <Scale className="h-8 w-8 text-primary" />
                    <h1 className="text-4xl font-bold">Terms of Service</h1>
                </div>
                <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
            </div>

            <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
                <section>
                    <h2 className="text-2xl font-semibold mb-4">1. Acceptance of Terms</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        By accessing and using GetReport, you accept and agree to be bound by the terms and provision of this agreement.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">2. Use License</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        Permission is granted to temporarily download one copy of the materials (information or software) on GetReport's website for personal, non-commercial transitory viewing only.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">3. Disclaimer</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        The materials on GetReport's website are provided "as is". GetReport makes no warranties, expressed or implied, and hereby disclaims and negates all other warranties, including without limitation, implied warranties or conditions of merchantability, fitness for a particular purpose, or non-infringement of intellectual property or other violation of rights.
                    </p>
                </section>
            </div>
        </div>
    );
};

export default TermsOfService;
