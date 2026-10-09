from typing import Dict, Any, Optional, List
from .schemas import ExecutiveMLSection

class ExecutiveMLReportSynthesizer:
    """
    Synthesizes Unsupervised clustering personas and Supervised AutoML leaderboards
    into cohesive, executive-ready narrative blocks and tables for PDF generation.
    """

    def synthesize_section(
        self,
        unsupervised_data: Optional[Dict[str, Any]] = None,
        supervised_data: Optional[Dict[str, Any]] = None,
    ) -> ExecutiveMLSection:
        unsupervised_ran = bool(unsupervised_data and unsupervised_data.get("ran"))
        supervised_ran = bool(supervised_data and supervised_data.get("ran"))

        if not unsupervised_ran and not supervised_ran:
            return ExecutiveMLSection(has_ml_content=False)

        takeaways: List[str] = []

        # 1. Unsupervised Segmentation Summary
        segmentation_summary: Optional[Dict[str, Any]] = None
        if unsupervised_ran:
            k = unsupervised_data.get("optimal_k", 0)
            var_pct = unsupervised_data.get("explained_variance_pct", 0.0)
            sil_score = unsupervised_data.get("silhouette_score", 0.0)
            personas = unsupervised_data.get("personas", [])

            takeaways.append(
                f"Statistical profiling identified {k} natural operational cohorts, "
                f"capturing {var_pct:.0f}% of total population variance."
            )

            if personas:
                top_persona = personas[0]
                takeaways.append(
                    f"Leading Cohort ({top_persona.get('name', 'Tier 1')}) represents "
                    f"{top_persona.get('share_pct', 0.0):.1f}% of records: {top_persona.get('summary', '')}"
                )

            formatted_personas = []
            for p in personas:
                traits = []
                for feat in p.get("distinguishing_features", []):
                    traits.append({
                        "feature": feat.get("feature", ""),
                        "direction": feat.get("direction", ""),
                        "index_ratio": f"{feat.get('index_ratio', 1.0):.2f}x",
                    })

                formatted_personas.append({
                    "cluster_id": p.get("cluster_id", 0),
                    "name": p.get("name", ""),
                    "size": p.get("size", 0),
                    "share_pct": f"{p.get('share_pct', 0.0):.1f}%",
                    "traits": traits,
                    "summary": p.get("summary", ""),
                })

            segmentation_summary = {
                "optimal_k": k,
                "silhouette_score": round(sil_score, 3),
                "explained_variance_pct": round(var_pct, 1),
                "personas": formatted_personas,
            }

        # 2. Supervised Predictive Attribution Summary
        driver_attribution: Optional[Dict[str, Any]] = None
        model_governance: Optional[Dict[str, Any]] = None

        if supervised_ran:
            target_col = supervised_data.get("target_column", "Unknown")
            task_type = supervised_data.get("task_type", "unknown").replace("_", " ").title()
            best_model = supervised_data.get("best_model_name", "Baseline Model")
            primary_metric = supervised_data.get("primary_metric_name", "score").upper()
            primary_val = supervised_data.get("primary_metric_value", 0.0)
            drivers = supervised_data.get("key_drivers", [])
            leaderboard = supervised_data.get("leaderboard", [])

            takeaways.append(
                f"Outcome modeling for '{target_col}' confirmed strong predictive signal "
                f"({primary_metric}: {primary_val:.2f})."
            )

            if drivers:
                top_driver = drivers[0]
                takeaways.append(
                    f"'{target_col}' is predominantly influenced by '{top_driver.get('feature')}' "
                    f"({top_driver.get('importance_pct', 0.0):.1f}% relative impact: {top_driver.get('impact_description', '')})."
                )

            formatted_drivers = []
            for d in drivers:
                formatted_drivers.append({
                    "feature": d.get("feature", ""),
                    "importance": f"{d.get('importance_pct', 0.0):.1f}%",
                    "direction": d.get("direction", "neutral"),
                    "impact": d.get("impact_description", ""),
                })

            driver_attribution = {
                "target_column": target_col,
                "task_type": task_type,
                "best_model_name": best_model,
                "primary_metric": primary_metric,
                "primary_value": round(primary_val, 4),
                "drivers": formatted_drivers,
            }

            model_governance = {
                "selected_model": best_model,
                "leaderboard": [
                    {
                        "model": e.get("model_name"),
                        "score": f"{e.get('primary_metric_value', 0.0):.4f}",
                        "latency_sec": f"{e.get('train_time_sec', 0.0):.3f}s",
                    }
                    for e in leaderboard
                ],
                "all_metrics": supervised_data.get("metrics", {}),
                "execution_time_sec": round(supervised_data.get("total_execution_time_sec", 0.0), 3),
            }

        return ExecutiveMLSection(
            has_ml_content=True,
            title="Key Performance Drivers & Operational Cohorts",
            executive_takeaways=takeaways,
            segmentation_summary=segmentation_summary,
            driver_attribution=driver_attribution,
            model_governance=model_governance,
        )
