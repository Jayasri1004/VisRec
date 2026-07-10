"""
explainer.py
============
Llama 3.1 Explainability & Critic Loop Engine for VisRec.
"""

import json
import re
import time
import warnings
from typing import Any, Dict, List, Optional

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CHART_PROMPT = """You are a data visualization expert. Explain in 2-3 clear sentences why the following chart is a good choice for this dataset. Be specific about the columns and what the analyst will learn. Do not use bullet points. Do not repeat the chart type name in the first word.

Dataset: {dataset_name} ({row_count} rows, {col_count} columns)
Chart type: {vis_type}
Columns used: {columns}
Analytical goal: {goal}
Confidence score: {confidence:.0%}

Write only the explanation, nothing else."""


_INSIGHT_PROMPT = """You are a data analyst. Explain in 1-2 plain-English sentences what the following statistical finding means for someone exploring this dataset, and what they should do about it.

Finding: {description}
Severity: {severity}
Suggested visualization: {recommended_vis}

Write only the explanation, nothing else."""


_DASHBOARD_PROMPT = """You are a data storytelling expert. Write a 2-3 sentence narrative that introduces the recommended dashboard below to an analyst seeing it for the first time. Describe what story the combination of charts tells and what questions it helps answer.

Dataset: {dataset_name} ({row_count} rows)
Charts included: {chart_summary}
Analytical goals covered: {goals}

Write only the narrative, nothing else."""


# CRITIC PROMPT: Rewrites the entire data story using ONLY the charts chosen by the human critic
_CRITIC_REVISION_PROMPT = """You are an expert data critic and story refiner. The user has explicitly curated their dashboard by ACCEPTING and DISMISSING specific charts. 
Your task is to write a fresh, comprehensive 3-sentence summary narrative of the dashboard focusing ONLY on the charts they ACCEPTED. Acknowledge what the main analytical focus is now based on their choices.

Dataset: {dataset_name}
ACCEPTED Visualizations to focus on: {accepted_charts}
DISMISSED/REJECTED Visualizations to ignore: {dismissed_charts}

Write only the updated narrative, nothing else."""


# ---------------------------------------------------------------------------
# Fallback template strings
# ---------------------------------------------------------------------------

def _fallback_chart_rationale(chart: Dict) -> str:
    return (
        f"{chart.get('description', 'Data visualization charting')} using {', '.join(chart['columns_used'][:3])}. "
        f"Analytical goal: {chart['insight_type'].replace('_', ' ')}. "
        f"Confidence: {chart['confidence_score']:.0%}."
    )

def _fallback_insight_narrative(insight: Dict) -> str:
    return insight.get("description", "")

def _fallback_dashboard_narrative(result: Dict) -> str:
    goals = result.get("dashboard", {}).get("goals_covered", [])
    n = result.get("dashboard", {}).get("n_charts", 0)
    return f"This dashboard presents {n} visualizations covering {', '.join(g.replace('_', ' ') for g in goals)}."


# ---------------------------------------------------------------------------
# Main class with Critic Layer
# ---------------------------------------------------------------------------

class VisExplainer:
    def __init__(self, model: str = "llama3.1:8b", ollama_url: str = "http://localhost:11434", timeout: int = 30, max_charts: int = 6):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.max_charts = max_charts
        self.available = self._check_ollama()

        if self.available:
            print(f" [Explainer] Ollama connected — model: {self.model}")
        else:
            warnings.warn(f"[VisExplainer] Ollama not reachable at {self.ollama_url}. Using fallbacks.", RuntimeWarning)

    def _check_ollama(self) -> bool:
        if not _REQUESTS_AVAILABLE:
            return False
        try:
            r = _requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _call(self, prompt: str) -> Optional[str]:
        if not self.available or not _REQUESTS_AVAILABLE:
            return None
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150, "top_p": 0.9},
            }
            r = _requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout)
            if r.status_code == 200:
                text = r.json().get("response", "").strip()
                return re.sub(r"\*\*|__|\*|_|`", "", text)
            return None
        except Exception as e:
            warnings.warn(f"[VisExplainer] LLM call failed: {e}", RuntimeWarning)
            return None

    def explain_chart(self, chart: Dict, dataset_name: str, row_count: int, col_count: int) -> str:
        if not self.available: return _fallback_chart_rationale(chart)
        prompt = _CHART_PROMPT.format(
            dataset_name=dataset_name, row_count=row_count, col_count=col_count,
            vis_type=chart["vis_type"], columns=", ".join(chart["columns_used"]),
            goal=chart["insight_type"].replace("_", " "), confidence=chart["confidence_score"]
        )
        result = self._call(prompt)
        return result if result else _fallback_chart_rationale(chart)

    def explain_insight(self, insight: Dict) -> str:
        if not self.available: return _fallback_insight_narrative(insight)
        prompt = _INSIGHT_PROMPT.format(
            description=insight.get("description", ""), severity=insight.get("severity", "medium"), recommended_vis=insight.get("recommended_vis", "chart")
        )
        result = self._call(prompt)
        return result if result else _fallback_insight_narrative(insight)

    def explain_dashboard(self, result: Dict) -> str:
        if not self.available: return _fallback_dashboard_narrative(result)
        charts = result.get("charts", [])
        chart_summary = "; ".join(f"{c['vis_type']} ({', '.join(c['columns_used'][:2])})" for c in charts[:6])
        goals = result.get("dashboard", {}).get("goals_covered", [])
        profile = result.get("profile", {})
        prompt = _DASHBOARD_PROMPT.format(dataset_name=result.get("dataset_name", "dataset"), row_count=profile.get("row_count", "?"), chart_summary=chart_summary, goals=", ".join(g.replace("_", " ") for g in goals))
        result_text = self._call(prompt)
        return result_text if result_text else _fallback_dashboard_narrative(result)

    # ─── ACTIVE CRITIC AGENTIC LOOP INTERFACE ───
    def re_evaluate_with_critic(self, result: Dict) -> Dict:
        """
        Takes human critic feedback, filters goals, and triggers DeepVis
        to completely re-evaluate and re-write the dashboard narrative summary.
        """
        charts = result.get("charts", [])
        dataset_name = result.get("dataset_name", "dataset")

        accepted = [c for c in charts if c.get("review_state") == "accepted"]
        dismissed = [c for c in charts if c.get("review_state") == "dismissed"]

        # Build detailed strings of what the user likes/dislikes for Llama context
        accepted_summary = "; ".join(f"{c['vis_type']} using {', '.join(c['columns_used'])}" for c in accepted) if accepted else "None"
        dismissed_summary = "; ".join(f"{c['vis_type']} using {', '.join(c['columns_used'])}" for c in dismissed) if dismissed else "None"

        print(f" [Critic Loop] Running Re-evaluation. Accepted: {len(accepted)}, Dismissed: {len(dismissed)}")

        if not self.available:
            # Fallback text summary update
            result["dashboard"]["narrative"] = f"Dashboard updated by critic. Active accepted charts: {len(accepted)}. Ignored charts: {len(dismissed)}."
            return result

        prompt = _CRITIC_REVISION_PROMPT.format(
            dataset_name=dataset_name,
            accepted_charts=accepted_summary,
            dismissed_charts=dismissed_summary
        )

        revised_narrative = self._call(prompt)
        if revised_narrative:
            result["dashboard"]["narrative"] = revised_narrative
        else:
            result["dashboard"]["narrative"] = "Dashboard summary refreshed based on curated charts."
        
        return result

    def enrich(self, result: Dict) -> Dict:
        profile = result.get("profile", {})
        dataset_name = result.get("dataset_name", "dataset")
        row_count = profile.get("row_count", 0)
        col_count = profile.get("col_count", 0)
        status = "llm" if self.available else "fallback"

        charts = result.get("charts", [])
        sorted_indices = sorted(range(len(charts)), key=lambda i: charts[i].get("confidence_score", 0), reverse=True)

        for rank, idx in enumerate(sorted_indices):
            chart = charts[idx]
            if rank < self.max_charts:
                explanation = self.explain_chart(chart, dataset_name, row_count, col_count)
            else:
                explanation = _fallback_chart_rationale(chart)
            charts[idx]["explanation"] = explanation

        dashboard = result.get("dashboard", {})
        dashboard["narrative"] = self.explain_dashboard(result)
        result["explainer_status"] = status
        return result