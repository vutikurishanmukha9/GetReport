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
    critical: 'bg-red-50 text-red-700 border-red-200',
    high: 'bg-orange-50 text-orange-700 border-orange-200',
    medium: 'bg-amber-50 text-amber-700 border-amber-200',
    low: 'bg-emerald-50 text-emerald-700 border-emerald-200',
};

const statusColors = {
    pending: 'bg-muted text-muted-foreground border-border/80',
    approved: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    rejected: 'bg-red-50 text-red-700 border-red-200',
    modified: 'bg-primary/5 text-primary border-primary/20',
};

const issueTypeLabels: Record<string, string> = {
    missing_values: 'Missing Values',
    duplicates: 'Duplicates',
    type_mismatch: 'Type Mismatch',
    outliers: 'Outliers',
    format_issue: 'Format Issue',
    high_cardinality: 'High Cardinality',
    empty_column: 'Empty Column',
    constant_column: 'Constant Column',
    encoding_issue: 'Encoding Issue',
};

export const IssueLedger: React.FC<IssueLedgerProps> = ({
    taskId,
    data,
    onRefresh,
    onProceed
}) => {
    const [loading, setLoading] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleApprove = async (issueId: string) => {
        setLoading(issueId);
        setError(null);
        try {
            await api.approveIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setError((e as Error).message);
        } finally {
            setLoading(null);
        }
    };

    const handleReject = async (issueId: string) => {
        setLoading(issueId);
        setError(null);
        try {
            await api.rejectIssue(taskId, issueId);
            onRefresh();
        } catch (e) {
            setError((e as Error).message);
        } finally {
            setLoading(null);
        }
    };

    const handleApproveAll = async () => {
        setLoading('all');
        setError(null);
        try {
            await api.approveAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setError((e as Error).message);
        } finally {
            setLoading(null);
        }
    };

    const handleRejectAll = async () => {
        setLoading('all');
        setError(null);
        try {
            await api.rejectAllIssues(taskId);
            onRefresh();
        } catch (e) {
            setError((e as Error).message);
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
            setError((e as Error).message);
        } finally {
            setLoading(null);
        }
    };

    const { issues, summary, locked } = data;
    const hasPending = summary.pending > 0;

    if (issues.length === 0) {
        return (
            <div className="bg-card border border-border shadow-premium rounded-2xl p-8 max-w-2xl mx-auto animate-in fade-in duration-300">
                <div className="flex flex-col items-center text-center">
                    <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mb-4 border border-emerald-200">
                        <Check className="w-8 h-8 text-emerald-600" />
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
                        <CheckCheck className="w-5 h-5 text-emerald-500 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Completeness</h4>
                            <p className="text-xs text-muted-foreground mt-1">No missing or null values detected across all columns.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-500 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Consistency</h4>
                            <p className="text-xs text-muted-foreground mt-1">Data types are uniform and formats are consistent.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-500 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Uniqueness</h4>
                            <p className="text-xs text-muted-foreground mt-1">No duplicate rows or redundant identifiers found.</p>
                        </div>
                    </div>
                    <div className="bg-muted/20 border border-border rounded-xl p-4 flex items-start gap-3">
                        <CheckCheck className="w-5 h-5 text-emerald-500 shrink-0" />
                        <div className="text-left">
                            <h4 className="text-sm font-semibold text-foreground">Validity</h4>
                            <p className="text-xs text-muted-foreground mt-1">No extreme outliers or invalid entries identified.</p>
                        </div>
                    </div>
                </div>

                <div className="mt-8 flex justify-center">
                    <button
                        onClick={onProceed}
                        className="px-8 py-3 bg-primary text-primary-foreground font-medium rounded-xl shadow-premium transition-all duration-150 hover:-translate-y-0.5 active:scale-95 flex items-center gap-2"
                    >
                        <Lock className="w-4 h-4" />
                        Proceed to Analysis
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-card border border-border shadow-premium rounded-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-4 duration-400">
            {/* Header */}
            <div className="border-b border-border px-6 py-5 bg-muted/10">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                        <h3 className="text-xl font-display font-bold text-foreground flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5 text-primary" />
                            Issue Ledger
                        </h3>
                        <p className="text-muted-foreground text-xs sm:text-sm mt-1">
                            Review and approve or reject data fixes before commencing analysis
                        </p>
                    </div>
                    <div className="flex items-center gap-2 self-start sm:self-center font-mono text-xs">
                        <span className="bg-muted px-3 py-1 rounded-md border border-border text-muted-foreground">
                          {summary.total} issues
                        </span>
                        {locked && (
                            <span className="bg-emerald-50 text-emerald-700 border border-emerald-200 px-3 py-1 rounded-md flex items-center gap-1.5 font-bold">
                                <Lock className="w-3.5 h-3.5" /> locked
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Summary Bar */}
            <div className="bg-muted/5 border-b border-border px-6 py-3">
                <div className="flex flex-wrap gap-4 items-center justify-between">
                    <div className="flex gap-4 text-xs font-mono">
                        <span className="flex items-center gap-1.5 text-muted-foreground">
                            <span className="w-2 h-2 bg-muted-foreground/45 rounded-full"></span>
                            pending: {summary.pending}
                        </span>
                        <span className="flex items-center gap-1.5 text-emerald-750">
                            <span className="w-2 h-2 bg-emerald-500 rounded-full"></span>
                            approved: {summary.approved}
                        </span>
                        <span className="flex items-center gap-1.5 text-destructive">
                            <span className="w-2 h-2 bg-destructive rounded-full"></span>
                            rejected: {summary.rejected}
                        </span>
                    </div>

                    {!locked && (
                        <div className="flex gap-2">
                            <button
                                onClick={handleApproveAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3 py-1.5 border border-border bg-white hover:bg-muted/50 rounded-lg text-xs font-mono font-medium hover:text-foreground disabled:opacity-50 flex items-center gap-1 transition-colors active:scale-95"
                            >
                                <CheckCheck className="w-4 h-4 text-emerald-600" /> approve all
                            </button>
                            <button
                                onClick={handleRejectAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3 py-1.5 border border-border bg-white hover:bg-muted/50 rounded-lg text-xs font-mono font-medium hover:text-foreground disabled:opacity-50 flex items-center gap-1 transition-colors active:scale-95"
                            >
                                <XCircle className="w-4 h-4 text-destructive" /> reject all
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 border-b border-red-200 px-6 py-3 text-sm font-mono text-destructive">
                    {error}
                </div>
            )}

            {/* Issues Table */}
            <div className="max-h-[420px] overflow-auto">
                <table className="w-full whitespace-nowrap md:whitespace-normal">
                    <thead className="bg-muted/30 border-b border-border sticky top-0 z-10">
                        <tr className="text-left text-[10px] font-mono font-semibold text-muted-foreground uppercase tracking-wider">
                            <th className="px-6 py-3">Issue Details</th>
                            <th className="px-4 py-3">Column</th>
                            <th className="px-4 py-3">Severity</th>
                            <th className="px-4 py-3">Impact</th>
                            <th className="px-4 py-3">Suggested Fix</th>
                            <th className="px-4 py-3">Status</th>
                            <th className="px-4 py-3 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                        {issues.map((issue) => (
                            <tr
                                key={issue.id}
                                className={`hover:bg-muted/10 transition-colors ${issue.status === 'rejected' ? 'opacity-55' : ''
                                    }`}
                            >
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-2">
                                        {issue.severity === 'critical' && <AlertCircle className="w-4 h-4 text-red-500 shrink-0" />}
                                        {issue.severity === 'high' && <AlertTriangle className="w-4 h-4 text-orange-500 shrink-0" />}
                                        {issue.severity === 'medium' && <Info className="w-4 h-4 text-yellow-500 shrink-0" />}
                                        <span className="font-display font-semibold text-sm text-foreground">
                                            {issueTypeLabels[issue.issue_type] || issue.issue_type}
                                        </span>
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed max-w-sm font-sans">
                                        {issue.description}
                                    </p>
                                </td>
                                <td className="px-4 py-4">
                                    <code className="text-xs bg-muted/30 text-foreground border border-border px-2 py-1 rounded-md font-mono">
                                        {issue.column || 'all_rows'}
                                    </code>
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full border ${severityColors[issue.severity]}`}>
                                        {issue.severity}
                                    </span>
                                </td>
                                <td className="px-4 py-4 text-xs font-mono text-muted-foreground">
                                    {issue.affected_rows.toLocaleString()} rows
                                    <span className="text-muted-foreground/60 ml-1">({issue.affected_pct}%)</span>
                                </td>
                                <td className="px-4 py-4 text-xs text-muted-foreground max-w-xs truncate font-mono">
                                    {issue.suggested_fix.toLowerCase()}
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-[10px] font-mono font-semibold px-2.5 py-0.5 rounded-full border ${statusColors[issue.status]}`}>
                                        {issue.status}
                                    </span>
                                </td>
                                <td className="px-4 py-4 text-right">
                                    {!locked && issue.status === 'pending' && (
                                        <div className="flex justify-end gap-1.5">
                                            <button
                                                onClick={() => handleApprove(issue.id)}
                                                disabled={loading !== null}
                                                className="p-1.5 bg-emerald-50 text-emerald-700 rounded-full border border-emerald-200 hover:bg-emerald-100 disabled:opacity-50 transition-all duration-150 active:scale-90"
                                                title="Approve"
                                            >
                                                <Check className="w-3.5 h-3.5" />
                                            </button>
                                            <button
                                                onClick={() => handleReject(issue.id)}
                                                disabled={loading !== null}
                                                className="p-1.5 bg-red-50 text-red-700 rounded-full border border-red-200 hover:bg-red-100 disabled:opacity-50 transition-all duration-150 active:scale-90"
                                                title="Reject"
                                            >
                                                <X className="w-3.5 h-3.5" />
                                            </button>
                                        </div>
                                    )}
                                    {issue.status !== 'pending' && (
                                        <span className="text-xs font-mono font-bold text-muted-foreground mr-2">
                                            {issue.status === 'approved' ? '✓ approved' : '✗ rejected'}
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer Actions */}
            <div className="bg-muted/5 border-t border-border px-6 py-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <p className="text-xs sm:text-sm text-muted-foreground leading-normal font-sans">
                        {hasPending ? (
                            <>
                                <AlertTriangle className="w-4 h-4 inline mr-1.5 text-primary align-text-bottom" />
                                <strong>{summary.pending}</strong> issue(s) still pending review
                            </>
                        ) : (
                            <>
                                <Check className="w-4 h-4 inline mr-1.5 text-emerald-600 align-text-bottom" />
                                All issues successfully reviewed.
                            </>
                        )}
                    </p>

                    <button
                        onClick={handleLockAndProceed}
                        disabled={loading !== null || hasPending || locked}
                        className="px-6 py-2.5 bg-primary text-primary-foreground rounded-xl font-medium shadow-premium hover:shadow-lg disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all duration-150 hover:-translate-y-0.5 active:scale-95"
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
