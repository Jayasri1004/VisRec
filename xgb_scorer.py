"""
xgb_scorer.py
=============

XGBoost-backed confidence scoring for VisRecommender, as a drop-in
replacement for the hand-tuned linear formula in
`VisRecommender._compute_confidence`.

Design goals
------------
1. Backward compatible: same inputs/outputs as the original formula
   (vis_type: str, columns: List[str]) -> float in [0, 1].
2. Safe by default: if xgboost is not installed, or no trained model
   is available yet, this transparently falls back to the *exact*
   original linear formula already in app.py. The pipeline never
   breaks because of this module.
3. Self-bootstrapping: on first run with no saved model, it trains a
   small XGBoost regressor on synthetic-but-principled training data
   derived from the same domain knowledge already encoded in
   PERCEPTUAL_SCORES and VIS_TYPE_RULES (so behavior starts close to
   the original formula, then can be improved later with real
   feedback data without changing any call site).
4. Retrainable: exposes `record_feedback` + `retrain` so that, once a
   feedback loop exists (e.g. accept/reject signals from
   `/api/result/{run_id}`), the model can be improved without
   touching VisRecommender at all.

This module intentionally has zero hard dependency on the rest of
app.py beyond the two static dicts it needs for feature construction
and synthetic bootstrap data, both of which are passed in by the
caller rather than imported, to avoid any circular-import risk.
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# =============================================================================
# Feature construction
# =============================================================================

@dataclass
class ScoringContext:
    """
    Everything the scorer needs about the current dataset/profile to
    build a feature vector. Mirrors exactly what
    VisRecommender._compute_confidence already has access to, so the
    integration patch only has to pass `self` data through, never
    recompute anything.
    """
    vis_type: str
    columns: List[str]
    perceptual_score: float          # PERCEPTUAL_SCORES.get(vis_type, 0.5)
    rule_satisfied: bool             # _check_vis_type_conditions(vis_type)
    row_count: int                   # self.profile['row_count']
    numeric_cols: int                # self.profile['numeric_cols']
    categorical_cols: int            # self.profile['categorical_cols']
    temporal_cols: int                # self.profile['temporal_cols']
    unique_cat_ratio: float          # self.profile['unique_cat_ratio']
    missing_ratio: float             # self.profile['missing_ratio']


_FEATURE_NAMES = [
    "perceptual_score",
    "rule_satisfied",
    "n_cols",
    "col_fit",
    "row_count_log",
    "row_bonus",
    "numeric_cols",
    "categorical_cols",
    "temporal_cols",
    "unique_cat_ratio",
    "missing_ratio",
]


def _col_fit(n_cols: int) -> float:
    """Same column-count-fit heuristic as the original formula."""
    return 1.0 if 2 <= n_cols <= 4 else max(0.3, 1.0 - 0.15 * abs(n_cols - 2))


def _row_bonus(row_count: int) -> float:
    """Same row-count bonus as the original formula."""
    return min(1.0, row_count / 100) * 0.1


def build_feature_vector(ctx: ScoringContext) -> np.ndarray:
    """Build the numeric feature vector XGBoost will score."""
    n_cols = len(ctx.columns)
    row_count_log = float(np.log1p(max(ctx.row_count, 0)))
    features = [
        ctx.perceptual_score,
        1.0 if ctx.rule_satisfied else 0.0,
        float(n_cols),
        _col_fit(n_cols),
        row_count_log,
        _row_bonus(ctx.row_count),
        float(ctx.numeric_cols),
        float(ctx.categorical_cols),
        float(ctx.temporal_cols),
        float(ctx.unique_cat_ratio),
        float(ctx.missing_ratio),
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)


# =============================================================================
# Original linear formula (fallback) -- copied verbatim in behavior from
# VisRecommender._compute_confidence so the fallback path is provably
# identical to current production behavior.
# =============================================================================

def original_linear_formula(ctx: ScoringContext) -> float:
    base_score = ctx.perceptual_score
    data_fit = 1.0 if ctx.rule_satisfied else 0.3
    col_fit = _col_fit(len(ctx.columns))
    row_bonus = _row_bonus(ctx.row_count)
    confidence = (0.5 * base_score + 0.3 * data_fit + 0.2 * col_fit) + row_bonus
    return min(1.0, max(0.0, confidence))


# =============================================================================
# Synthetic bootstrap training data
# =============================================================================

def _generate_bootstrap_dataset(
    perceptual_scores: Dict[str, float],
    n_samples_per_type: int = 60,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (features, target) pairs for the initial model,
    anchored to the same domain priors already in PERCEPTUAL_SCORES so
    that an untrained XGBoost model starts out close to today's
    behavior rather than randomly. The "target" is the original linear
    formula's output plus small structured noise representing the
    kinds of corrections we'd expect real feedback to teach (e.g.
    rewarding rule satisfaction more strongly, and tapering off the
    row-count bonus sooner than the original linear formula does).

    This is a bootstrap, not a claim of real learned signal -- it is
    designed to be overwritten by `retrain()` once real feedback
    exists.
    """
    rng = np.random.RandomState(seed)
    X, y = [], []

    for vis_type, base in perceptual_scores.items():
        for _ in range(n_samples_per_type):
            rule_satisfied = rng.random() > 0.35
            n_cols = rng.randint(1, 6)
            row_count = int(rng.choice([5, 20, 50, 100, 250, 500, 2000, 10000]))
            numeric_cols = rng.randint(0, 6)
            categorical_cols = rng.randint(0, 4)
            temporal_cols = rng.randint(0, 2)
            unique_cat_ratio = float(rng.uniform(0, 1))
            missing_ratio = float(rng.uniform(0, 0.3))

            ctx = ScoringContext(
                vis_type=vis_type,
                columns=["c"] * n_cols,
                perceptual_score=base,
                rule_satisfied=rule_satisfied,
                row_count=row_count,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                temporal_cols=temporal_cols,
                unique_cat_ratio=unique_cat_ratio,
                missing_ratio=missing_ratio,
            )

            baseline = original_linear_formula(ctx)

            # Structured corrections, representing the *direction* future
            # real feedback is expected to push the model:
            #  - reward rule satisfaction more sharply (current formula
            #    only weights it at 0.3 of the total score)
            #  - penalize high missingness a bit harder
            #  - slightly discount the unbounded row-count bonus for very
            #    large datasets (diminishing returns)
            target = baseline
            if rule_satisfied:
                target += 0.05
            else:
                target -= 0.10
            target -= 0.15 * missing_ratio
            if row_count > 5000:
                target -= 0.03
            target += rng.normal(0, 0.02)  # small noise so the tree doesn't overfit a formula
            target = float(np.clip(target, 0.0, 1.0))

            X.append(build_feature_vector(ctx)[0])
            y.append(target)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# =============================================================================
# Main scorer
# =============================================================================

class XGBConfidenceScorer:
    """
    Drop-in confidence scorer. Construct once per process (or once per
    VisRecommender, it's cheap to share) and call `.score(ctx)`.

    Usage from VisRecommender (see PATCH section in the accompanying
    instructions for the exact integration):

        self._xgb_scorer = XGBConfidenceScorer(
            perceptual_scores=self.PERCEPTUAL_SCORES,
            model_path=os.path.join(self.config.output_dir, "xgb_confidence_model.json"),
        )
        ...
        confidence = self._xgb_scorer.score(ctx)
    """

    def __init__(
        self,
        perceptual_scores: Dict[str, float],
        model_path: Optional[str] = None,
        auto_bootstrap: bool = True,
    ):
        self.perceptual_scores = perceptual_scores
        self.model_path = model_path
        self.available = _XGB_AVAILABLE
        self._model = None
        self._feedback_log: List[Dict] = []

        if not self.available:
            warnings.warn(
                "[XGBConfidenceScorer] xgboost is not installed; "
                "falling back to the original linear confidence formula. "
                "Install with `pip install xgboost` to enable learned scoring.",
                RuntimeWarning,
            )
            return

        loaded = False
        if self.model_path and os.path.exists(self.model_path):
            try:
                self._model = xgb.XGBRegressor()
                self._model.load_model(self.model_path)
                loaded = True
            except Exception as e:
                warnings.warn(
                    f"[XGBConfidenceScorer] Could not load model at "
                    f"{self.model_path}: {e}. Will re-bootstrap.",
                    RuntimeWarning,
                )

        if not loaded and auto_bootstrap:
            self._bootstrap_train()

    # -- training -------------------------------------------------------

    def _bootstrap_train(self):
        """Train an initial model on synthetic, formula-anchored data."""
        X, y = _generate_bootstrap_dataset(self.perceptual_scores)
        self._model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model.fit(X, y)
        if self.model_path:
            try:
                os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
                self._model.save_model(self.model_path)
            except Exception as e:
                warnings.warn(
                    f"[XGBConfidenceScorer] Could not save bootstrapped "
                    f"model to {self.model_path}: {e}",
                    RuntimeWarning,
                )

    def retrain(self, extra_X: Optional[np.ndarray] = None, extra_y: Optional[np.ndarray] = None):
        """
        Retrain on the synthetic bootstrap set plus any real feedback
        collected via `record_feedback`. Call this periodically (e.g.
        from a scheduled job or an admin endpoint) once real
        accept/reject/click feedback exists. Safe to call even with no
        extra data -- it just re-bootstraps.
        """
        if not self.available:
            return
        X, y = _generate_bootstrap_dataset(self.perceptual_scores)
        if self._feedback_log:
            fb_X = np.array([row["features"] for row in self._feedback_log], dtype=np.float32)
            fb_y = np.array([row["target"] for row in self._feedback_log], dtype=np.float32)
            X = np.vstack([X, fb_X])
            y = np.concatenate([y, fb_y])
        if extra_X is not None and extra_y is not None:
            X = np.vstack([X, extra_X])
            y = np.concatenate([y, extra_y])

        self._model = xgb.XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
        )
        self._model.fit(X, y)
        if self.model_path:
            try:
                os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
                self._model.save_model(self.model_path)
            except Exception as e:
                warnings.warn(
                    f"[XGBConfidenceScorer] Could not save retrained "
                    f"model to {self.model_path}: {e}",
                    RuntimeWarning,
                )

    def record_feedback(self, ctx: ScoringContext, accepted: bool):
        """
        Record a real-world feedback signal (e.g. user accepted/kept a
        recommended chart vs. dismissed it) for use in the next
        `retrain()` call. Does not retrain immediately -- call
        `retrain()` separately (e.g. on a schedule) since retraining on
        every single click would be wasteful.
        """
        target = 0.85 if accepted else 0.15
        self._feedback_log.append({
            "features": build_feature_vector(ctx)[0].tolist(),
            "target": target,
        })

    # -- scoring ----------------------------------------------------------

    def score(self, ctx: ScoringContext) -> float:
        """
        Return a confidence score in [0, 1] for the given context.
        Falls back to the original linear formula if xgboost is
        unavailable or the model failed to train/load for any reason.
        """
        if not self.available or self._model is None:
            return original_linear_formula(ctx)

        try:
            features = build_feature_vector(ctx)
            pred = float(self._model.predict(features)[0])
            return float(np.clip(pred, 0.0, 1.0))
        except Exception as e:
            warnings.warn(
                f"[XGBConfidenceScorer] Scoring failed ({e}); "
                "falling back to linear formula for this call.",
                RuntimeWarning,
            )
            return original_linear_formula(ctx)
