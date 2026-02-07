import React, { useState } from 'react';
import { Check, X, AlertTriangle, AlertCircle, Info, Lock, CheckCheck, XCircle } from 'lucide-react';

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

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Severity badge colors
const severityColors = {
    critical: 'bg-red-100 text-red-800 border-red-200 dark:bg-red-900/30 dark:text-red-400',
    high: 'bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-900/30 dark:text-orange-400',
    medium: 'bg-yellow-100 text-yellow-800 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-400',
    low: 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400',
};

const statusColors = {
    pending: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300',
    approved: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    rejected: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    modified: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
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
            const res = await fetch(`${API_BASE}/jobs/${taskId}/issues/${issueId}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            if (!res.ok) throw new Error('Failed to approve issue');
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
            const res = await fetch(`${API_BASE}/jobs/${taskId}/issues/${issueId}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            if (!res.ok) throw new Error('Failed to reject issue');
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
            const res = await fetch(`${API_BASE}/jobs/${taskId}/issues/approve-all`, {
                method: 'POST',
            });
            if (!res.ok) throw new Error('Failed to approve all issues');
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
            const res = await fetch(`${API_BASE}/jobs/${taskId}/issues/reject-all`, {
                method: 'POST',
            });
            if (!res.ok) throw new Error('Failed to reject all issues');
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
            const res = await fetch(`${API_BASE}/jobs/${taskId}/issues/lock`, {
                method: 'POST',
            });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Failed to lock issues');
            }
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
            <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6 text-center">
                <Check className="w-12 h-12 mx-auto text-green-500 mb-3" />
                <h3 className="text-lg font-semibold text-green-800 dark:text-green-300">
                    No Issues Detected
                </h3>
                <p className="text-green-600 dark:text-green-400 mt-2">
                    Your data looks clean! You can proceed directly to analysis.
                </p>
                <button
                    onClick={onProceed}
                    className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                    Proceed to Analysis
                </button>
            </div>
        );
    }

    return (
        <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-sm overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-xl font-bold text-white flex items-center gap-2">
                            <AlertTriangle className="w-5 h-5" />
                            Issue Ledger
                        </h3>
                        <p className="text-blue-100 text-sm mt-1">
                            Review and approve/reject data fixes before cleaning
                        </p>
                    </div>
                    <div className="flex items-center gap-4 text-white text-sm">
                        <span className="bg-white/20 px-3 py-1 rounded-full">
                            {summary.total} issues
                        </span>
                        {locked && (
                            <span className="bg-green-500 px-3 py-1 rounded-full flex items-center gap-1">
                                <Lock className="w-4 h-4" /> Locked
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Summary Bar */}
            <div className="bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-3">
                <div className="flex flex-wrap gap-4 items-center justify-between">
                    <div className="flex gap-4 text-sm">
                        <span className="flex items-center gap-1 text-gray-500 dark:text-gray-400">
                            <span className="w-2 h-2 bg-gray-400 rounded-full"></span>
                            Pending: {summary.pending}
                        </span>
                        <span className="flex items-center gap-1 text-green-600 dark:text-green-400">
                            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                            Approved: {summary.approved}
                        </span>
                        <span className="flex items-center gap-1 text-red-600 dark:text-red-400">
                            <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                            Rejected: {summary.rejected}
                        </span>
                    </div>

                    {!locked && (
                        <div className="flex gap-2">
                            <button
                                onClick={handleApproveAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3 py-1.5 bg-green-100 text-green-700 rounded-lg text-sm font-medium hover:bg-green-200 disabled:opacity-50 flex items-center gap-1 transition-colors"
                            >
                                <CheckCheck className="w-4 h-4" /> Approve All
                            </button>
                            <button
                                onClick={handleRejectAll}
                                disabled={loading !== null || !hasPending}
                                className="px-3 py-1.5 bg-red-100 text-red-700 rounded-lg text-sm font-medium hover:bg-red-200 disabled:opacity-50 flex items-center gap-1 transition-colors"
                            >
                                <XCircle className="w-4 h-4" /> Reject All
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800 px-6 py-3 text-red-700 dark:text-red-400">
                    {error}
                </div>
            )}

            {/* Issues Table */}
            <div className="max-h-96 overflow-y-auto">
                <table className="w-full">
                    <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
                        <tr className="text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                            <th className="px-6 py-3">Issue</th>
                            <th className="px-4 py-3">Column</th>
                            <th className="px-4 py-3">Severity</th>
                            <th className="px-4 py-3">Impact</th>
                            <th className="px-4 py-3">Suggested Fix</th>
                            <th className="px-4 py-3">Status</th>
                            <th className="px-4 py-3 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                        {issues.map((issue) => (
                            <tr
                                key={issue.id}
                                className={`hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors ${issue.status === 'rejected' ? 'opacity-50' : ''
                                    }`}
                            >
                                <td className="px-6 py-4">
                                    <div className="flex items-center gap-2">
                                        {issue.severity === 'critical' && <AlertCircle className="w-4 h-4 text-red-500" />}
                                        {issue.severity === 'high' && <AlertTriangle className="w-4 h-4 text-orange-500" />}
                                        {issue.severity === 'medium' && <Info className="w-4 h-4 text-yellow-500" />}
                                        <span className="font-medium text-gray-900 dark:text-gray-100">
                                            {issueTypeLabels[issue.issue_type] || issue.issue_type}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                                        {issue.description}
                                    </p>
                                </td>
                                <td className="px-4 py-4">
                                    <code className="text-sm bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                                        {issue.column || 'All rows'}
                                    </code>
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-xs font-medium px-2 py-1 rounded-full border ${severityColors[issue.severity]}`}>
                                        {issue.severity.toUpperCase()}
                                    </span>
                                </td>
                                <td className="px-4 py-4 text-sm text-gray-600 dark:text-gray-400">
                                    {issue.affected_rows.toLocaleString()} rows
                                    <span className="text-gray-400 ml-1">({issue.affected_pct}%)</span>
                                </td>
                                <td className="px-4 py-4 text-sm text-gray-600 dark:text-gray-400 max-w-xs truncate">
                                    {issue.suggested_fix}
                                </td>
                                <td className="px-4 py-4">
                                    <span className={`text-xs font-medium px-2 py-1 rounded-full ${statusColors[issue.status]}`}>
                                        {issue.status.toUpperCase()}
                                    </span>
                                </td>
                                <td className="px-4 py-4 text-right">
                                    {!locked && issue.status === 'pending' && (
                                        <div className="flex justify-end gap-2">
                                            <button
                                                onClick={() => handleApprove(issue.id)}
                                                disabled={loading !== null}
                                                className="p-1.5 bg-green-100 text-green-700 rounded hover:bg-green-200 disabled:opacity-50 transition-colors"
                                                title="Approve"
                                            >
                                                <Check className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => handleReject(issue.id)}
                                                disabled={loading !== null}
                                                className="p-1.5 bg-red-100 text-red-700 rounded hover:bg-red-200 disabled:opacity-50 transition-colors"
                                                title="Reject"
                                            >
                                                <X className="w-4 h-4" />
                                            </button>
                                        </div>
                                    )}
                                    {issue.status !== 'pending' && (
                                        <span className="text-xs text-gray-400">
                                            {issue.status === 'approved' ? '✓' : '✗'}
                                        </span>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Footer Actions */}
            <div className="bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                        {hasPending ? (
                            <>
                                <AlertTriangle className="w-4 h-4 inline mr-1 text-yellow-500" />
                                {summary.pending} issue(s) still pending review
                            </>
                        ) : (
                            <>
                                <Check className="w-4 h-4 inline mr-1 text-green-500" />
                                All issues reviewed. Ready to proceed.
                            </>
                        )}
                    </p>

                    <button
                        onClick={handleLockAndProceed}
                        disabled={loading !== null || hasPending || locked}
                        className="px-6 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg font-medium hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all shadow-md hover:shadow-lg"
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
