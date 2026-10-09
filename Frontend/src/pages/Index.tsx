import { Header } from "@/components/Header";
import { HeroSection } from "@/components/HeroSection";
import { EngineTrustBar } from "@/components/EngineTrustBar";
import { ComparisonSection } from "@/components/ComparisonSection";
import { ArchitectureSection } from "@/components/ArchitectureSection";
import { QuickLaunchDock } from "@/components/QuickLaunchDock";
import { MobileFloatingBar } from "@/components/MobileFloatingBar";
import { Footer } from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen flex flex-col relative">
      {/* Living Ambient Animated Mesh Background Underlay */}
      <div className="fixed inset-0 -z-10 bg-mesh-home pointer-events-none" />
      <Header onReset={() => {}} showReset={false} />
      <main className="flex-1">
        <HeroSection />
        <EngineTrustBar />
        <ComparisonSection />
        <ArchitectureSection />
        <QuickLaunchDock />
      </main>
      <Footer />
      <MobileFloatingBar />
    </div>
  );
};

export default Index;
