import os

class KG4VisEngine:
    """
    Knowledge Graph Reasoning Engine for VisRec (KG4Vis Architecture).
    Maps analytical intentions and chart structures to valid ontological reasoning paths.
    """
    def __init__(self):
        # Define the core ontological paths for the visualization domain
        self.ontology_graph = {
            "COMPARISON": ["COMPARISON", "Categorical Aggregation", "bar"],
            "DISTRIBUTION": ["DISTRIBUTION", "Distribution Analysis", "histogram"],
            "CORRELATION": ["CORRELATION", "Scatter Relationship", "scatter"],
            "TREND": ["TREND", "Temporal Evolution", "line"]
        }

    def generate_reasoning_trail(self, raw_input: str) -> list:
        """
        Parses raw recommender signals and charts them against verified knowledge graph nodes.
        Returns a 3-element list representing [Task Node -> Transformation Node -> Visual Node].
        """
        if not raw_input:
            return ["General Exploration", "Generic Analysis", "Fallback Display"]

        # Clean and uppercase the matching input string safely
        normalized_signal = str(raw_input).upper().strip()

        # ─── CRITIC LAYER SUBSTRING MATCHING GATEKEEPER ───
        # Intercepts chart/insight strings to ensure accurate ontological classification
        target_insight = "COMPARISON"  # Default global fallback node
        
        if "BOX" in normalized_signal or "DIST" in normalized_signal or "HIST" in normalized_signal:
            target_insight = "DISTRIBUTION"
        elif "CORR" in normalized_signal or "RELATION" in normalized_signal or "SCATTER" in normalized_signal:
            target_insight = "CORRELATION"
        elif "TREND" in normalized_signal or "TIME" in normalized_signal or "TEMP" in normalized_signal or "LINE" in normalized_signal:
            target_insight = "TREND"
        elif "COMP" in normalized_signal or "BAR" in normalized_signal or "CAT" in normalized_signal:
            target_insight = "COMPARISON"
        # ───────────────────────────────────────────────────

        # Fetch the matching verified node trail from our graph mapping dictionary
        return self.ontology_graph.get(target_insight, ["General Exploration", "Generic Analysis", "Fallback Display"])