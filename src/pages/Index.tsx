import { useState } from "react";
import { FileUpload } from "@/components/FileUpload";
import { DataPreview } from "@/components/DataPreview";
import { ReportGeneration } from "@/components/ReportGeneration";
import { Header } from "@/components/Header";
import { HeroSection } from "@/components/HeroSection";
import { FeaturesSection } from "@/components/FeaturesSection";
import { Footer } from "@/components/Footer";

export type AppStep = "upload" | "preview" | "generating" | "complete";

export interface DatasetInfo {
  fileName: string;
  rows: number;
  columns: string[];
  preview: Record<string, unknown>[];
  dataTypes: Record<string, string>;
}

const Index = () => {
  const [step, setStep] = useState<AppStep>("upload");
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);

  const handleFileUploaded = (info: DatasetInfo) => {
    setDatasetInfo(info);
    setStep("preview");
  };

  const handleGenerateReport = () => {
    setStep("generating");
    // Simulate report generation
    setTimeout(() => {
      setStep("complete");
    }, 3000);
  };

  const handleReset = () => {
    setStep("upload");
    setDatasetInfo(null);
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

        {step === "preview" && datasetInfo && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <DataPreview 
              datasetInfo={datasetInfo} 
              onGenerateReport={handleGenerateReport}
              onBack={handleReset}
            />
          </div>
        )}

        {(step === "generating" || step === "complete") && datasetInfo && (
          <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8 lg:py-12">
            <ReportGeneration 
              step={step}
              datasetInfo={datasetInfo}
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
