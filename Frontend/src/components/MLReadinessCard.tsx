import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { AlertTriangle, CheckCircle, HelpCircle, XCircle } from "lucide-react";
import { MLReadiness } from "@/types/api";

interface MLReadinessCardProps {
  mlReadiness?: MLReadiness;
}

export const MLReadinessCard: React.FC<MLReadinessCardProps> = ({ mlReadiness }) => {
  if (!mlReadiness) return null;

  const { score, status, reasons, recommendation, column_context } = mlReadiness;

  // Status-specific configuration
  const statusConfig = {
    Ready: {
      colorClass: "bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-950/20 dark:text-emerald-400 dark:border-emerald-900/30",
      badgeColorClass: "bg-emerald-500 text-white",
      icon: <CheckCircle className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />,
    },
    "Needs Cleaning": {
      colorClass: "bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-950/20 dark:text-amber-400 dark:border-amber-900/30",
      badgeColorClass: "bg-amber-500 text-white",
      icon: <AlertTriangle className="h-5 w-5 text-amber-600 dark:text-amber-400" />,
    },
    "Not Ready": {
      colorClass: "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950/20 dark:text-rose-400 dark:border-rose-900/30",
      badgeColorClass: "bg-rose-500 text-white",
      icon: <XCircle className="h-5 w-5 text-rose-600 dark:text-rose-400" />,
    },
  }[status] || {
    colorClass: "bg-slate-50 text-slate-700 border-slate-200",
    badgeColorClass: "bg-slate-500 text-white",
    icon: <HelpCircle className="h-5 w-5 text-slate-600" />,
  };

  return (
    <Card className="border border-border bg-card shadow-premium rounded-2xl overflow-hidden mt-6">
      <CardHeader className="border-b border-border bg-muted/20 px-6 py-4">
        <CardTitle className="text-lg font-semibold flex items-center gap-2">
          Machine Learning Readiness Assessment
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6">
        <div className="flex flex-col md:flex-row items-start md:items-center gap-6 pb-6 border-b border-border">
          <div className="flex items-center gap-4">
            <div className={`flex flex-col items-center justify-center w-20 h-20 rounded-2xl ${statusConfig.badgeColorClass} font-bold shadow-lg shadow-black/5`}>
              <span className="text-2xl leading-none">{Math.round(score)}%</span>
              <span className="text-[9px] uppercase tracking-wider font-semibold opacity-90 mt-1">Score</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xl font-bold tracking-tight text-foreground">{status}</span>
                {statusConfig.icon}
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">{column_context}</p>
            </div>
          </div>
          
          <div className="flex-1">
            <div className={`p-4 rounded-xl border ${statusConfig.colorClass}`}>
              <h4 className="text-xs font-semibold uppercase tracking-wider mb-1">Expert Recommendation</h4>
              <p className="text-sm font-medium leading-relaxed">{recommendation}</p>
            </div>
          </div>
        </div>

        {reasons && reasons.length > 0 && (
          <div className="pt-6">
            <h4 className="text-sm font-semibold text-foreground mb-3">Detected Issues & Constraints</h4>
            <ul className="space-y-2.5">
              {reasons.map((reason, idx) => (
                <li key={idx} className="flex items-start gap-2.5 text-sm text-muted-foreground">
                  <span className="w-1.5 h-1.5 rounded-full bg-rose-500 dark:bg-rose-400 mt-2 shrink-0" />
                  <span className="leading-relaxed">{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
