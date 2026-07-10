"""
critic_engine.py
================
RL-Text2Vis Inspired Visual Critic Engine for Rule-Driven Evaluation.
"""

from typing import Dict, List, Any

class VisualizationCritic:
    def __init__(self):
        # Setup analytical task map matching vis_type to expected column count profile distributions
        self.suitability_matrix = {
            "bar": {"min_cols": 1, "max_cols": 3, "optimal_types": ["categorical", "numeric"]},
            "line": {"min_cols": 2, "max_cols": 3, "optimal_types": ["temporal", "numeric"]},
            "scatter": {"min_cols": 2, "max_cols": 2, "optimal_types": ["numeric"]},
            "pie": {"min_cols": 1, "max_cols": 2, "optimal_types": ["categorical"]},
            "histogram": {"min_cols": 1, "max_cols": 1, "optimal_types": ["numeric"]},
            "box": {"min_cols": 1, "max_cols": 2, "optimal_types": ["categorical", "numeric"]}
        }

    def evaluate(self, chart: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a chart generation payload and dataset profile parameters to output an 
        RL-Text2Vis inspired holistic visualization score distribution.
        """
        vis_type = str(chart.get("vis_type", "bar")).lower()
        columns_used = chart.get("columns_used", [])
        num_cols = len(columns_used)
        
        # Pull XGBoost confidence directly 
        xgboost_conf = float(chart.get("confidence_score", 0.85))
        recommendation_confidence = int(xgboost_conf * 100)

        # 1. Evaluate Chart Suitability
        suitability_score = 85 # Baseline default metric
        if vis_type in self.suitability_matrix:
            rules = self.suitability_matrix[vis_type]
            if rules["min_cols"] <= num_cols <= rules["max_cols"]:
                suitability_score += 10
            else:
                suitability_score -= 15
                
            # Intercept specific profile matching indicators
            if vis_type == "scatter" and num_cols == 2:
                suitability_score = max(suitability_score, 95)
            elif vis_type == "line" and len(profile.get("temporal_cols", [])) > 0:
                suitability_score = max(suitability_score, 94)
        suitability_score = min(max(suitability_score, 10), 100)

        # 2. Evaluate Readability (Axis configuration and layout checks)
        readability_score = 90 # Default structural base score assuming pipeline template rendering rules pass
        if num_cols == 0:
            readability_score -= 40
        elif num_cols > 3:
            readability_score -= 15 # Heavy label truncation distortion penalty
        readability_score = min(max(readability_score, 10), 100)

        # 3. Evaluate Visual Clarity (Complexity & multi-variable alignment checks)
        clarity_score = 92
        if "id" in [c.lower() for c in columns_used]:
            clarity_score -= 20 # Structural penalty for plotting noise parameters
        clarity_score = min(max(clarity_score, 10), 100)

        # 4. Evaluate Information Density
        # Avoid clutter or over-aggregation sparsity checkpoints
        information_density_score = 88
        if num_cols == 2:
            information_density_score = 92
        elif num_cols >= 4:
            information_density_score = 70 # Overcrowded visualization frame boundary
        information_density_score = min(max(information_density_score, 10), 100)

        # 5. Derive Deterministic Overall Unified Score Balance Equation
        overall_score = int(
            (suitability_score * 0.30) + 
            (readability_score * 0.20) + 
            (clarity_score * 0.20) + 
            (information_density_score * 0.15) + 
            (recommendation_confidence * 0.15)
        )

        # 6. Generate Structural Semantic Summary Strings
        summary_type = vis_type.replace("_", " ").title()
        critic_summary = (
            f"The {summary_type} visualization is well-suited because the structure utilizes {num_cols} attribute maps "
            f"matching optimal presentation tasks with an overall execution balance score of {overall_score}%."
        )
        if overall_score < 75:
            critic_summary = f"The structural complexity of chosen columns causes clarity compression. Consider dimensionality filtering adjustments."

        return {
            "chart_type": summary_type,
            "suitability_score": suitability_score,
            "readability_score": readability_score,
            "clarity_score": clarity_score,
            "information_density_score": information_density_score,
            "recommendation_confidence": recommendation_confidence,
            "overall_score": overall_score,
            "critic_summary": critic_summary
        }