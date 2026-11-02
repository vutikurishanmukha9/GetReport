import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error caught by ErrorBoundary:", error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-full rounded-2xl border border-destructive/30 bg-destructive/5 p-6 sm:p-8 space-y-4 my-4 animate-in fade-in duration-200">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-destructive/10 text-destructive flex items-center justify-center border border-destructive/20 shrink-0">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-foreground">
                {this.props.fallbackTitle || "Something went wrong in this module"}
              </h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                An unexpected interface error occurred. You can safely retry or switch to another tab.
              </p>
            </div>
          </div>

          {this.state.error && (
            <div className="p-3 rounded-xl bg-background/80 border border-destructive/20 font-mono text-xs text-destructive overflow-x-auto max-h-32">
              {this.state.error.message || String(this.state.error)}
            </div>
          )}

          <div className="flex items-center gap-2 pt-1">
            <Button
              size="sm"
              variant="outline"
              onClick={this.handleReset}
              className="gap-1.5 text-xs h-8 cursor-pointer"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>Retry Module</span>
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

