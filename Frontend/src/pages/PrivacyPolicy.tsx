import { Shield } from "lucide-react";

const PrivacyPolicy = () => {
    return (
        <div className="container mx-auto px-4 py-16 max-w-4xl animate-in fade-in duration-500">
            <div className="mb-12 border-b pb-8">
                <div className="flex items-center gap-4 mb-4">
                    <Shield className="h-8 w-8 text-primary" />
                    <h1 className="text-4xl font-bold">Privacy Policy</h1>
                </div>
                <p className="text-muted-foreground">Last updated: {new Date().toLocaleDateString()}</p>
            </div>

            <div className="prose prose-slate dark:prose-invert max-w-none space-y-8">
                <section>
                    <h2 className="text-2xl font-semibold mb-4">1. Data Collection</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        We take your data privacy seriously. When you upload a file to GetReport, it is processed ephemerally on our secure servers. We do not store your raw datasets permanently. Once the analysis is complete and your report is generated, the original data is discarded.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">2. Usage of Information</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        The information you provide is used solely for the purpose of generating statistical reports and insights. We do not sell, trade, or otherwise transfer to outside parties your personally identifiable information or proprietary data.
                    </p>
                </section>

                <section>
                    <h2 className="text-2xl font-semibold mb-4">3. Security</h2>
                    <p className="text-muted-foreground leading-relaxed">
                        We implement a variety of security measures to maintain the safety of your personal information when you enter, submit, or access your personal information.
                    </p>
                </section>
            </div>
        </div>
    );
};

export default PrivacyPolicy;
