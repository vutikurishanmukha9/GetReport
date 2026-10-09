import React, { useState } from 'react';
import { Check, X, AlertTriangle, AlertCircle, Info, Lock, CheckCheck, XCircle } from 'lucide-react';
import { api } from '@/services/api';

// Types
interface Issue {
    id: string;
    issue_type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    column: string | null;
    affected_rows: number;
    affected_pct: number;
    description: string;
    suggested_fix: string;
    fix_code: string;
    status: 'pending' | 'approved' | 'rejected' | 'modified';
    user_note: string;
}

interface IssueLedgerData {
    issues: Issue[];
    summary: {
        pending: number;
        approved: number;
        rejected: number;
        modified: number;
        total: number;
    };
    locked: boolean;
    locked_at: string | null;
}

interface IssueLedgerProps {
    taskId: string;
    data: IssueLedgerData;
    onRefresh: () => void;
    onProceed: () => void;
}

// API client is imported from services/api.ts

// Severity badge colors
const severityColors = {
    critical: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    high: 'bg-orange-500/15 text-orange-300 border-orange-500/30',
    medium: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
    low: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
};

const statusColors = {
    pending: 'bg-muted/50 text-muted-foreground border-border/80',
    approved: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
    rejected: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
    modified: 'bg-primary/10 text-primary border-primary/20',
};

const issueTypeLabels = {
    missing_values: 'Missing Values',
    duplicates: 'Duplicates',
    type_mismatch: 'Type Mismatch',
    outliers: 'Outliers',
    multivariate_anomaly: 'Multivariate Anomaly',
    format_issue: 'Format Issue',
    high_cardinality: 'High Cardinality',
    empty_column: 'Empty Column',
    constant_column: 'Constant Column',
    encoding_issue: 'Encoding Issue',
} as const;

function formatIssueType(type: string): string {
    if (type in issueTypeLabels) {
        // SAFETY: 'in' operator check narrows type to valid keyof issueTypeLabels
        return issueTypeLabels[type as keyof typeof issueTypeLabels];
    }
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

export const IssueLedger: React.FC<IssueLedgerProps> = ({
    taskId,
    data,
    onRefresh,
    onProceed
}) => {
    const [loading, setLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [optimisticIssues, setOptimisticIssues] = useState<Issue[] | null>(null);

    // Sync optimistic state when parent data changes
    const issuesList = optimisticIssues || data.issues;

    // Derived summary counts for instant UI badge updates
    const summary = React.useMemo(() => {
        const approved = issuesList.filter(i => i.status === 'approved').length;
        const rejected = issuesList.filter(i => i.status === 'rejected').length;
        const modified = issuesList.filter(i => i.status === 'modified').length;
        const pending = issuesList.filter(i => i.status === 'pending').length;
        return {
            approved,
            rejected,
            modified,
            pending,
            total: issuesList.length
        };
    }, [issuesList]);

    const handleApprove = async (issueId: string) => {
        setLoading(issueId);
        setError(null);

        // Optimistic mutation (0ms feedback)
        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.id === issueId ? { ...i, status: 'approved' as const } : i))
        );

        try {
            await api.approveIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous); // Rollback on failure
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    const handleReject = async (issueId: string) => {
        setLoading(issueId);
        setError(null);

        // Optimistic mutation (0ms feedback)
        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.id === issueId ? { ...i, status: 'rejected' as const } : i))
        );

        try {
            await api.rejectIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous); // Rollback on failure
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    const handleApproveAll = async () => {
        setLoading('all');
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.status === 'pending' ? { ...i, status: 'approved' as const } : i))
        );

        try {
            await api.approveAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    const handleRejectAll = async () => {
        setLoading('all');
        setError(null);

        const previous = issuesList;
        setOptimisticIssues(prev =>
            (prev || data.issues).map(i => (i.status === 'pending' ? { ...i, status: 'rejected' as const } : i))
        );

        try {
            await api.rejectAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setOptimisticIssues(previous);
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    const handleLockAndProceed = async () => {
        setLoading('lock');
        setError(null);
        try {
            await api.lockIssues(taskId);
            onProceed();
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(null);
        }
    };

    const issues = issuesList;
    const locked = data.locked;
    const hasPending = summary.pending > 0;

    if (issues.length === 0) {
        return (
            <div className="bg-card border border-border/80 shadow-premium rounded-2xl p-8 max-w-2xl mx-auto animate-in fade-in duration-300">
                <div className="flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-emerald-500/15 rounded-full flex items-center justify-center mb-4 border border-emerald-500/30 shadow-2xs">
                        <Check className="w-8 h-8 text-emerald-400" />
                    </div>
                    <h3 className="text-2xl font-display font-bold text-foreground">
                        Data Quality: Excellent
                    </h3>
                    <p className="text-muted-foreground mt-2 max-w-md mx-auto">
                        We have thoroughly scanned your dataset and found no structural anomalies, missing values, or formatting issues. Your data is perfectly clean and ready for analysis.
                    </p>
                </div>
                
                <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Completeness</h4>
                            <p className="text-xs text-muted-foreground mt-1">No missing or null values detected across all columns.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Consistency</h4>
                            <p className="text-xs text-muted-foreground mt-1">Data types are uniform and formats are consistent.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Uniqueness</h4>
                            <p className="text-xs text-muted-foreground mt-1">No duplicate rows or redundant identifiers found.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-400 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Validity</h4>
                            <p className="text-xs text-muted-foreground mt-1">No extreme outliers or invalid entries identified.</p>
                        </div>
                    </div>
                </div>

                <div className="mt-8 flex justify-center">
                    <button
                        onClick={onProceed}
                        className="px-8 py-3 bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 hover:from-violet-500 hover:via-purple-500 hover:to-indigo-500 text-white font-display font-semibold rounded-xl border border-violet-400/40 ring-1 ring-white/20 shadow-[0_4px_14px_-2px_rgba(124,58,237,0.38),inset_0_1px_1px_rgba(255,255,255,0.3)] transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center gap-2 cursor-pointer"
                    >
                        <Lock className="w-4 h-4" />
                        Proceed to Analysis
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-card border border-border/80 shadow-premium rounded-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-400">
            {/* Header */}
            <div className="border-b border-border/60 px-4 sm:px-6 py-4 sm:py-5 bg-gradient-to-r from-muted/20 via-card to-card">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-start gap-3.5">
                        <div className="w-10 h-10 bg-primary/10 border border-primary/20 rounded-xl flex items-center justify-center shrink-0 shadow-2xs">
                            <AlertTriangle className="w-5 h-5 text-primary" />
                        </div>
                        <div>
                            <h3 className="text-lg sm:text-xl font-display font-bold text-foreground tracking-tight flex items-center gap-2">
                                Issue Ledger
                            </h3>
                            <p className="text-muted-foreground text-xs sm:text-sm mt-0.5 font-sans">
                                Review and approve or reject data fixes before commencing analysis
                            </p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2 self-start sm:self-center">
                        <span className="bg-muted/60 text-foreground px-3 py-1 rounded-full border border-border/60 text-xs font-medium shadow-2xs">
                            {summary.total} {summary.total === 1 ? 'issue' : 'issues'}
                        </span>
                        {locked && (
                            <span className="bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 px-3 py-1 rounded-full flex items-center gap-1.5 text-xs font-semibold shadow-2xs">
                                <Lock className="w-3.5 h-3.5 text-emerald-400" /> Locked
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Summary Bar */}
            <div className="bg-muted/10 border-b border-border/60 px-4 sm:px-6 py-3 sm:py-3.5">
                <div className="flex flex-wrap gap-3 sm:gap-4 items-center justify-between">
                    <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-xs">
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-300 font-medium">
                            <span className="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></span>
                            Pending: {summary.pending}
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-medium">
                            <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
                            Approved: {summary.approved}
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 font-medium">
                            <span className="w-2 h-2 bg-rose-500 rounded-full"></span>
                            Rejected: {summary.rejected}
                        </span>
                    </div>

                    {!locked && (
                        <div className="flex flex-wrap items-center gap-2">
                            <button
                                onClick={handleApproveAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3.5 py-1.5 border border-emerald-500/30 bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 rounded-xl text-xs font-semibold disabled:opacity-40 flex items-center gap-1.5 shadow-2xs hover:shadow-xs cursor-pointer t-spring-press"
                            >
                                <CheckCheck className="w-4 h-4 text-emerald-400" /> Approve All
                            </button>
                            <button
                                onClick={handleRejectAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3.5 py-1.5 border border-rose-500/30 bg-rose-500/15 hover:bg-rose-500/25 text-rose-300 rounded-xl text-xs font-semibold disabled:opacity-40 flex items-center gap-1.5 shadow-2xs hover:shadow-xs cursor-pointer t-spring-press"
                            >
                                <XCircle className="w-4 h-4 text-rose-400" /> Reject All
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-rose-500/15 border-b border-rose-500/30 px-6 py-3 text-xs font-semibold text-rose-300 flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                    <span>{error}</span>
                </div>
            )}

            {/* Issues Table */}
            <div className="max-h-[440px] overflow-x-auto overflow-y-auto custom-scrollbar">
                <table className="w-full text-left border-collapse min-w-[700px]">
                    <thead className="bg-muted/40 border-b border-border/80 sticky top-0 z-10 backdrop-blur-md">
                        <tr className="text-[11px] font-sans font-bold text-muted-foreground uppercase tracking-wider">
                            <th className="px-6 py-3.5">Issue Details</th>
                            <th className="px-4 py-3.5">Column</th>
                            <th className="px-4 py-3.5">Severity</th>
                            <th className="px-4 py-3.5">Impact</th>
                            <th className="px-4 py-3.5">Suggested Fix</th>
                            <th className="px-4 py-3.5">Status</th>
                            <th className="px-6 py-3.5 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border/60 bg-card">
                        {issues.map((issue) => (
                            <tr
                                key={issue.id}
                                className={`hover:bg-primary/[0.02] transition-colors duration-150 ${
                                    issue.status === 'rejected' ? 'opacity-50 bg-muted/10' : ''
                                }`}
                            >
                                <td className="px-6 py-4">
                                    <div className="flex items-start gap-2.5">
                                        {issue.severity === 'critical' && (
                                            <div className="p-1 rounded-md bg-rose-500/15 text-rose-400 shrink-0 mt-0.5 border border-rose-500/30">
                                                <AlertCircle className="w-4 h-4" />
                                            </div>
                                        )}
                                        {issue.severity === 'high' && (
                                            <div className="p-1 rounded-md bg-orange-500/15 text-orange-400 shrink-0 mt-0.5 border border-orange-500/30">
                                                <AlertTriangle className="w-4 h-4" />
                                            </div>
                                        )}
                                        {issue.severity === 'medium' && (
                                            <div className="p-1 rounded-md bg-amber-500/15 text-amber-400 shrink-0 mt-0.5 border border-amber-500/30">
                                                <Info className="w-4 h-4" />
                                            </div>
                                        )}
                                        {issue.severity === 'low' && (
                                            <div className="p-1 rounded-md bg-emerald-500/15 text-emerald-400 shrink-0 mt-0.5 border border-emerald-500/30">
                                                <Check className="w-4 h-4" />
                                            </div>
                                        )}
                                        <div>
                                            <span className="font-display font-semibold text-sm text-foreground tracking-tight block">
                                                {formatIssueType(issue.issue_type)}
                                            </span>
                                            <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed max-w-sm font-sans">
                                                {issue.description}
                                            </p>
                                        </div>
                                    </div>
                                </td>
                                <td className="px-4 py-4">
                                    <code className="text-xs bg-muted/40 text-foreground border border-border/80 px-2.5 py-1 rounded-lg font-mono shadow-2xs">
                                        {issue.column || 'all_rows'}
                                    </code>
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-[10px] font-sans font-bold uppercase tracking-wider px-2.5 py-1 rounded-full border shadow-2xs ${severityColors[issue.severity]}`}>
                                        {issue.severity}
                                    </span>
                                </td>
                                <td className="px-4 py-4 text-xs font-sans text-muted-foreground">
                                    <span className="font-semibold text-foreground">{issue.affected_rows.toLocaleString()}</span> rows
                                    <span className="text-muted-foreground/70 ml-1 font-mono">({issue.affected_pct}%)</span>
                                </td>
                                <td className="px-4 py-4">
                                    <span className="text-xs font-sans text-muted-foreground max-w-xs block truncate" title={issue.suggested_fix}>
                                        {issue.suggested_fix}
                                    </span>
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border inline-flex items-center gap-1.5 shadow-2xs ${statusColors[issue.status]}`}>
                                        {issue.status === 'pending' && <span className="w-1.5 h-1.5 bg-amber-400 rounded-full animate-pulse"></span>}
                                        {issue.status === 'approved' && <Check className="w-3 h-3 text-emerald-400" />}
                                        {issue.status === 'rejected' && <X className="w-3 h-3 text-rose-400" />}
                                        <span className="capitalize">{issue.status}</span>
                                    </span>
                                </td>
                                <td className="px-6 py-4 text-right">
                                    {!locked && issue.status === 'pending' && (
                                        <div className="flex justify-end items-center gap-2">
                                            <button
                                                onClick={() => handleApprove(issue.id)}
                                                disabled={loading !== null}
                                                className="w-8 h-8 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/30 rounded-full border border-emerald-500/30 flex items-center justify-center shadow-2xs hover:shadow-xs cursor-pointer disabled:opacity-50 t-spring-press"
                                                title="Approve Fix"
                                            >
                                                <Check className="w-4 h-4 text-emerald-400" />
                                            </button>
                                            <button
                                                onClick={() => handleReject(issue.id)}
                                                disabled={loading !== null}
                                                className="w-8 h-8 bg-rose-500/15 text-rose-300 hover:bg-rose-500/30 rounded-full border border-rose-500/30 flex items-center justify-center shadow-2xs hover:shadow-xs cursor-pointer disabled:opacity-50 t-spring-press"
                                                title="Reject Fix"
                                            >
                                                <X className="w-4 h-4 text-rose-400" />
                                            </button>
                                        </div>
                                    )}
                                    {issue.status !== 'pending' && (
                                        <span className={`text-xs font-semibold inline-flex items-center gap-1 ${
                                            issue.status === 'approved' ? 'text-emerald-400' : 'text-rose-400'
                                        }`}>
                                            {issue.status === 'approved' ? '✓ Approved' : '✕ Rejected'}
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer Actions */}
            <div className="bg-muted/10 border-t border-border/60 px-6 py-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div className="flex items-center gap-2 text-xs sm:text-sm text-muted-foreground font-sans">
                        {hasPending ? (
                            <>
                                <div className="p-1 rounded-md bg-amber-100 text-amber-700">
                                    <AlertTriangle className="w-4 h-4" />
                                </div>
                                <span><strong className="text-foreground">{summary.pending}</strong> issue(s) still pending review before locking dataset.</span>
                            </>
                        ) : (
                            <>
                                <div className="p-1 rounded-md bg-emerald-100 text-emerald-700">
                                    <Check className="w-4 h-4" />
                                </div>
                                <span className="text-emerald-800 font-medium">All issues successfully reviewed and ready for analysis.</span>
                            </>
                        )}
                    </div>

                    <button
                        onClick={handleLockAndProceed}
                        disabled={loading !== null || hasPending || locked}
                        className="px-6 py-2.5 bg-primary text-primary-foreground rounded-xl font-semibold shadow-premium hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all duration-150 hover:-translate-y-0.5 active:scale-95 cursor-pointer"
                    >
                        <Lock className="w-4 h-4" />
                        Lock & Proceed to Analysis
                    </button>
                </div>
            </div>
        </div>
    );
};

export default IssueLedger;
