"""
shap_engine.py
==============
SHAP Explainability Engine for Tree-Based Visualization Recommendation Models.
"""

import numpy as np
import shap
from typing import Dict, List, Any

class ShapExplainerEngine:
    def __init__(self, trained_xgb_model: Any, feature_names: List[str] = None):
        """
        Initializes the SHAP TreeExplainer for the XGBoost model.
        """
        self.model = trained_xgb_model
        # Use TreeExplainer for exact, high-performance polynomial-time calculations
        self.explainer = shap.TreeExplainer(self.model)
        
        # Default fallback feature vocabulary mapping if not supplied by application layer
        self.feature_names = feature_names or [
            "Correlation Strength", "Numerical Columns Count", "Categorical Columns Count",
            "Temporal Column Presence", "Missing Value Ratio", "Outlier Ratio", "High Cardinality Flag"
        ]

    def explain_recommendation(self, feature_vector: np.ndarray, target_class_idx: int) -> Dict[str, Any]:
        """
        Computes local feature contributions (Shapley Values) for a single inference event.
        
        :param feature_vector: 1D or 2D numpy array representing the input metadata characteristics.
        :param target_class_idx: The integer class label index corresponding to the predicted chart type.
        """
        # Ensure correct array orientation shape [1, N]
        if len(feature_vector.shape) == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Generate SHAP values for all classes
        shap_values = self.explainer.shap_values(feature_vector)

        # Handle structural variance across shap output formats (list vs multi-dim array)
        if isinstance(shap_values, list):
            # Binary/Multi-class list format from specific xgboost builds
            local_shap = shap_values[target_class_idx][0]
        elif len(shap_values.shape) == 3:
            # Shape: [samples, features, classes]
            local_shap = shap_values[0, :, target_class_idx]
        else:
            # Multi-class spatial array shape format: [classes, features] or [samples, classes, features]
            if len(shap_values.shape) == 2 and shap_values.shape[0] == 1:
                local_shap = shap_values[0]
            else:
                local_shap = shap_values[0]

        contributions = []
        for idx, val in enumerate(local_shap):
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature Meta_{idx}"
            contributions.append({
                "feature_name": feat_name,
                "shap_value": float(round(val, 4)),
                "display_impact": f"{'+' if val >= 0 else ''}{round(val, 2)}"
            })

        # Sort features based on absolute magnitude of their contribution
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "contributions": contributions[:4], # Extract top 4 defining variables
            "base_value": float(self.explainer.expected_value[target_class_idx]) if hasattr(self.explainer.expected_value, "__len__") else float(self.explainer.expected_value)
        }