import { FileText, Github, Linkedin, Mail } from "lucide-react";
import { Link } from "react-router-dom";

export const Footer = () => {
  return (
    <footer className="border-t bg-muted/30">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10 md:py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          {/* Brand */}
          <div className="sm:col-span-2 lg:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <Link to="/" className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                  <FileText className="h-4 w-4 text-primary-foreground" />
                </div>
                <span className="text-lg font-bold">GetReport</span>
              </Link>
            </div>
            <p className="text-sm text-muted-foreground max-w-xs">
              Transform your datasets into professional analytical reports instantly.
              AI-powered analysis with zero configuration.
            </p>
          </div>

          {/* How It Works */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">How It Works</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold shrink-0">1</span>
                Upload CSV or Excel
              </li>
              <li className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold shrink-0">2</span>
                AI analyzes your data
              </li>
              <li className="flex items-center gap-2">
                <span className="w-5 h-5 rounded-full bg-primary/10 text-primary text-xs flex items-center justify-center font-bold shrink-0">3</span>
                Download PDF report
              </li>
            </ul>
          </div>

          {/* Features */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">Features</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li>📊 Statistical Analysis</li>
              <li>📈 Auto-generated Charts</li>
              <li>🧠 AI Insights</li>
              <li>💬 Chat with Your Data</li>
            </ul>
          </div>

          {/* Connect */}
          <div>
            <h4 className="font-semibold mb-3 sm:mb-4">Connect</h4>
            <ul className="space-y-2 sm:space-y-3 text-sm text-muted-foreground">
              <li>
                <a
                  href="https://github.com/vutikurishanmukha9/GetReport"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors inline-flex items-center gap-2"
                >
                  <Github className="h-4 w-4" />
                  GitHub
                </a>
              </li>
              <li>
                <a
                  href="https://linkedin.com/in/vutikurishanmukha9"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-foreground transition-colors inline-flex items-center gap-2"
                >
                  <Linkedin className="h-4 w-4" />
                  LinkedIn
                </a>
              </li>
              <li>
                <a
                  href="mailto:vutikurishanmukha9@gmail.com"
                  className="hover:text-foreground transition-colors inline-flex items-center gap-2"
                >
                  <Mail className="h-4 w-4" />
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="mt-8 sm:mt-10 md:mt-12 pt-6 sm:pt-8 border-t flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs sm:text-sm text-muted-foreground">
            © {new Date().getFullYear()} GetReport. All rights reserved.
          </p>
          <p className="text-xs text-muted-foreground/60">
            Built with React, FastAPI & AI
          </p>
        </div>
      </div>
    </footer>
  );
};
