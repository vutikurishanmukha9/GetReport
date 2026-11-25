import type { CleaningRule, CleaningRulesMap, InspectionReport } from "@/types/api";

const ALLOWED_RULE_ACTIONS: ReadonlySet<string> = new Set([
  "drop_rows",
  "fill_mean",
  "fill_median",
  "fill_mode",
  "fill_value",
  "fill_group_median",
  "replace_outliers_median",
  "cap_outliers",
  "flag_outliers",
  "coerce_numeric",
  "clean_masked_nulls",
  "drop_column",
  "clean_text",
  "fix_mojibake",
  "standardize_categories",
  "cast_boolean",
  "parse_date",
  "nullify_sentinel",
  "nullify_out_of_range",
  "nullify_invalid_regex",
  "add_missing_indicator",
  "log1p_transform",
  "hash_pii",
  "keep_as_text",
  "winsorize",
  "default",
  "none",
]);

export function normalizeRuleAction(rawAction: string): CleaningRule["action"] {
  if (rawAction === "replace_outliers_median" || rawAction === "winsorize") {
    return "cap_outliers";
  }
  if (ALLOWED_RULE_ACTIONS.has(rawAction)) {
    // SAFETY: membership in ALLOWED_RULE_ACTIONS guarantees union match
    return rawAction as CleaningRule["action"];
  }
  return "default";
}

export function buildSuggestedRules(report?: InspectionReport | null): CleaningRulesMap {
  const initial: CleaningRulesMap = {};
  (report?.issues || []).forEach((issue) => {
    if (issue.column && issue.column !== "Multiple" && issue.action && issue.action !== "default") {
      const normalizedAction = normalizeRuleAction(issue.action);
      initial[issue.column] = {
        action: normalizedAction,
        value: normalizedAction === "fill_value" ? "Unknown" : undefined,
      };
    }
  });
  return initial;
}
