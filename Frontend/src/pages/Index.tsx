import { useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { DataPreview } from "@/components/DataPreview";
import { ReportGeneration } from "@/components/ReportGeneration";
import { Header } from "@/components/Header";
import { HeroSection } from "@/components/HeroSection";
import { FeaturesSection } from "@/components/FeaturesSection";
import { Footer } from "@/components/Footer";
import type { ApiResponse } from "@/types/api";

export type AppStep = "upload" | "preview" | "generating" | "complete";

const Index = () => {
  const [step, setStep] = useState<AppStep>("upload");
  const [apiData, setApiData] = useState<ApiResponse | null>(null);

  const handleFileUploaded = (data: ApiResponse) => {
    setApiData(data);
    setStep("preview");
  };

  const handleGenerateReport = () => {
    setStep("generating");
    // Actual generation logic moved to ReportGeneration component
    // But we need to transition to 'complete' when done?
    // Or ReportGeneration handles the transition?
    // Let's keep it simple: ReportGeneration takes the data and calls the API
    // We just handle the step transition here or pass a callback.
  };

  const handleReportComplete = () => {
    setStep("complete");
  };

  const handleReset = () => {
    setStep("upload");
    setApiData(null);
  };

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header onReset={handleReset} showReset={step !== "upload"} />

      <main className="flex-1">
        {step === "upload" && (
          <>
            <HeroSection />
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 md:py-12 lg:py-16">
              <FileUpload onFileUploaded={handleFileUploaded} />
            </div>
            <FeaturesSection />
          </>
        )}

        {step === "preview" && apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <DataPreview
              info={apiData.info}
              cleaningReport={apiData.cleaning_report}
              analysis={apiData.analysis}
              onGenerateReport={handleGenerateReport}
              onBack={handleReset}
            />
          </div>
        )}

        {(step === "generating" || step === "complete") && apiData && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <ReportGeneration
              step={step}
              filename={apiData.filename}
              analysis={apiData.analysis}
              charts={apiData.charts}
              insights={apiData.insights}
              onComplete={handleReportComplete}
              onReset={handleReset}
            />
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default Index;
