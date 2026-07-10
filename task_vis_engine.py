"""
task_vis_engine.py
==================
TaskVis Inspired Goal-Aware Optimization Module for Analytical Intent Alignment.
"""

from typing import Dict, List, Any

class TaskVisEngine:
    def __init__(self):
        # Maps user-driven analytical tasks to optimal chart families and feature weights
        self.goal_matrix = {
            "TREND_ANALYSIS": {
                "preferred_charts": ["line", "area"],
                "boost_factor": 0.25,
                "description": "Optimized to expose temporal progressions, continuous data adjustments, and direction vectors."
            },
            "COMPARISON": {
                "preferred_charts": ["bar", "pie"],
                "boost_factor": 0.20,
                "description": "Optimized to expose magnitude variations, rank benchmarks, and part-to-whole segment ratios."
            },
            "CORRELATION": {
                "preferred_charts": ["scatter", "bubble"],
                "boost_factor": 0.30,
                "description": "Optimized to evaluate dual-axis numerical covariances, clusters, and pattern correlations."
            },
            "DISTRIBUTION": {
                "preferred_charts": ["histogram", "box"],
                "boost_factor": 0.25,
                "description": "Optimized to visualize dataset dispersion, density splits, quartile structures, and skewness."
            },
            "ANOMALY_DETECTION": {
                "preferred_charts": ["scatter", "box", "line"],
                "boost_factor": 0.35,
                "description": "Optimized to detect structural deviations, data outliers, and volatile spikes."
            },
            "FORECASTING": {
                "preferred_charts": ["line"],
                "boost_factor": 0.30,
                "description": "Optimized to project historical series sequences into future timeline estimation intervals."
            }
        }

    def align_recommendations(self, recommendations: List[Dict[str, Any]], target_goal: str) -> List[Dict[str, Any]]:
        """
        Dynamically adjusts XGBoost output weights based on TaskVis goal-alignment heuristics.
        """
        goal_key = str(target_goal).upper().replace(" ", "_")
        if goal_key not in self.goal_matrix:
            return recommendations # Pass through unchanged if no explicit task match occurs

        meta = self.goal_matrix[goal_key]
        preferred = meta["preferred_charts"]
        boost = meta["boost_factor"]

        for rec in recommendations:
            vis_type = str(rec.get("vis_type", "bar")).lower()
            current_conf = float(rec.get("confidence_score", 0.85))

            # Apply proportional boost if the chart type matches the analytical task goal
            if vis_type in preferred:
                rec["confidence_score"] = min(1.0, current_conf + boost)
                rec["is_goal_optimized"] = True
                rec["goal_explanation"] = f"Task Vis Boost applied: {meta['description']}"
            else:
                rec["confidence_score"] = max(0.1, current_conf - (boost * 0.5))
                rec["is_goal_optimized"] = False
                rec["goal_explanation"] = f"De-prioritized. Chart type does not match optimal target parameters for {target_goal}."

        # Re-sort the recommendation list based on their new goal-aware confidence scores
        recommendations.sort(key=lambda x: x.get("confidence_score", 0.0), reverse=True)
        return recommendations