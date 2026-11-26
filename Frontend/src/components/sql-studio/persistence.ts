import type { QueryHistoryItem, SqlWorksheetTab } from "./types";
import { DEFAULT_STARTER_SQL } from "./types";

const WORKSHEETS_KEY_PREFIX = "getreport_sql_worksheets_v1:";
const HISTORY_KEY_PREFIX = "getreport_sql_history_v1:";
const MAX_PERSISTED_HISTORY = 30;
const MAX_PERSISTED_TABS = 12;

export function createDefaultWorksheet(sql: string = DEFAULT_STARTER_SQL, index: number = 1): SqlWorksheetTab {
  return {
    id: `tab-${index}-${Date.now()}`,
    title: `Query ${index}`,
    sql,
    isSavedSql: false,
  };
}

export function loadPersistedWorksheets(taskId: string): SqlWorksheetTab[] | null {
  if (!taskId) return null;
  try {
    const raw = window.localStorage.getItem(`${WORKSHEETS_KEY_PREFIX}${taskId}`);
    if (!raw) return null;
    const parsed: SqlWorksheetTab[] = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    const valid = parsed.filter(
      (item) =>
        Boolean(item) &&
        Boolean(item.id) &&
        Boolean(item.title) &&
        item.sql !== null &&
        item.sql !== undefined &&
        item.sql.constructor === String,
    );
    return valid.length > 0 ? valid.slice(0, MAX_PERSISTED_TABS) : null;
  } catch {
    return null;
  }
}

export function savePersistedWorksheets(taskId: string, tabs: SqlWorksheetTab[]): void {
  if (!taskId) return;
  try {
    window.localStorage.setItem(`${WORKSHEETS_KEY_PREFIX}${taskId}`, JSON.stringify(tabs.slice(0, MAX_PERSISTED_TABS)));
  } catch {
    // Ignore storage quota errors
  }
}

export function loadPersistedHistory(taskId: string): QueryHistoryItem[] | null {
  if (!taskId) return null;
  try {
    const raw = window.localStorage.getItem(`${HISTORY_KEY_PREFIX}${taskId}`);
    if (!raw) return null;
    const parsed: QueryHistoryItem[] = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return null;
    const valid = parsed.filter(
      (item) =>
        Boolean(item) &&
        Boolean(item.id) &&
        Boolean(item.sql) &&
        (item.status === "success" || item.status === "error"),
    );
    return valid.length > 0 ? valid.slice(0, MAX_PERSISTED_HISTORY) : null;
  } catch {
    return null;
  }
}

export function savePersistedHistory(taskId: string, history: QueryHistoryItem[]): void {
  if (!taskId) return;
  try {
    window.localStorage.setItem(
      `${HISTORY_KEY_PREFIX}${taskId}`,
      JSON.stringify(history.slice(0, MAX_PERSISTED_HISTORY)),
    );
  } catch {
    // Ignore storage quota errors
  }
}
