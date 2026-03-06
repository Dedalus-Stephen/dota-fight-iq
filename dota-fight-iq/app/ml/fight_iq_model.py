"""
Fight IQ Score Model — XGBoost Regression

Predicts a 0-100 score representing how closely a player's fight performance
matches 7000+ MMR patterns.

Training approach:
    - High-MMR samples (7k+, avg_rank_tier >= 80) are scored 85-100
    - Samples are weighted by rank tier to create a continuous target
    - Sub-scores computed for each category: ability, damage, item, survival, extraction

Inference:
    - Returns composite score + sub-scores for a single player-fight
    - Sub-scores decomposed using SHAP-like feature contribution groups
"""

import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from app.ml import SCORE_WEIGHTS, MODEL_NAMES
from app.ml.feature_engineering import FIGHT_IQ_FEATURE_COLS

logger = logging.getLogger(__name__)


# ── Score Label Generation ────────────────────────────────

def generate_fight_iq_labels(df: pd.DataFrame, rank_col: str = "avg_rank_tier") -> pd.Series:
    """
    Generate target labels (0-100) from rank tier.

    Rank tier encoding in Dota 2:
        - 80+ = Immortal (our 7k+ baseline = 85-100)
        - 70-79 = Divine
        - 60-69 = Ancient
        - 50-59 = Legend
        - etc.

    Since we're training on 7k+ data, all samples are high-rank.
    We create variation using performance metrics relative to peers.
    """
    if rank_col not in df.columns or df[rank_col].isna().all():
        # All data is high-MMR (from our collection pipeline) — use performance-based scoring
        return _performance_based_labels(df)

    # Rank-based baseline + performance adjustment
    rank = df[rank_col].fillna(80)
    base_score = rank.clip(10, 100)  # rough mapping: rank tier → base score
    return base_score


def _performance_based_labels(df: pd.DataFrame) -> pd.Series:
    """
    When all data is from same rank bracket, create labels from relative performance.
    Players who performed better than median in key metrics get higher scores.
    """
    score_components = pd.DataFrame(index=df.index)

    # Each metric contributes to the score based on percentile rank within the dataset
    for col, weight in [
        ("damage_per_sec", 0.25),
        ("ability_casts_per_sec", 0.20),
        ("gold_delta", 0.15),
        ("survived", 0.15),
        ("item_activations", 0.10),
        ("kills", 0.10),
        ("healing_per_sec", 0.05),
    ]:
        if col in df.columns:
            # Percentile rank: 0-1
            pct = df[col].rank(pct=True, method="average")
            score_components[col] = pct * weight
        else:
            score_components[col] = 0

    # Composite: weighted sum → scale to 40-100 range
    # (40 floor because even bad 7k+ plays are decent by average standards)
    raw = score_components.sum(axis=1)
    scaled = 40 + (raw * 60)
    return scaled.clip(0, 100).round(1)


# ── Model Training ────────────────────────────────────────

class FightIQModel:
    """XGBoost regression model for Fight IQ scoring."""

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.version: str = ""
        self.metrics: dict = {}
        self.feature_importances: dict = {}

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the Fight IQ model on a prepared DataFrame.

        Args:
            df: DataFrame with FIGHT_IQ_FEATURE_COLS + rank/performance data

        Returns:
            Training metrics dict
        """
        logger.info(f"Training Fight IQ model on {len(df)} samples")

        # Generate labels
        y = generate_fight_iq_labels(df)

        # Select features
        available_features = [c for c in FIGHT_IQ_FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0)

        if len(X) < 50:
            raise ValueError(f"Not enough training data: {len(X)} samples (need 50+)")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # XGBoost config — tuned for tabular data, prevents overfitting
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(available_features),
        }

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importances = {
            col: round(float(imp), 4)
            for col, imp in sorted(
                zip(available_features, importances),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        self.version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        logger.info(
            f"Fight IQ model trained: MAE={self.metrics['mae']}, "
            f"R²={self.metrics['r2']}, version={self.version}"
        )
        return self.metrics

    def predict(self, features: dict) -> dict:
        """
        Score a single player-fight.

        Args:
            features: dict from extract_player_fight_features()

        Returns:
            {fight_iq_score, ability_score, damage_score, item_score,
             survival_score, extraction_score, component_details}
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        available_features = [c for c in FIGHT_IQ_FEATURE_COLS if c in features]
        X = np.array([[features.get(c, 0) for c in available_features]])

        composite = float(self.model.predict(X)[0])
        composite = max(0, min(100, composite))

        # Decompose into sub-scores using feature group contributions
        sub_scores = self._decompose_score(features, composite)

        return {
            "fight_iq_score": round(composite, 1),
            **sub_scores,
            "model_version": self.version,
            "component_details": {
                "feature_importances": self.feature_importances,
                "raw_features": {k: features.get(k) for k in available_features},
            },
        }

    def _decompose_score(self, features: dict, composite: float) -> dict:
        """
        Break composite score into sub-scores by feature group.
        Uses weighted feature contribution approach.
        """
        # Group features into categories
        groups = {
            "ability": ["ability_casts_per_sec", "total_ability_casts"],
            "damage": ["damage_per_sec", "damage_per_nw", "damage_total"],
            "item": ["item_activations", "bkb_used", "blink_used"],
            "survival": ["deaths", "survived", "buybacks"],
            "extraction": ["gold_delta", "xp_delta"],
        }

        # Calculate each group's contribution based on feature importances
        total_importance = sum(self.feature_importances.values()) or 1
        group_scores = {}

        for group_name, group_features in groups.items():
            group_importance = sum(
                self.feature_importances.get(f, 0) for f in group_features
            )
            # Scale the composite score by this group's relative importance
            weight = SCORE_WEIGHTS[group_name]
            raw_contribution = (group_importance / total_importance) * composite
            # Blend with the weight-based target
            group_scores[f"{group_name}_score"] = round(
                0.5 * raw_contribution / weight + 0.5 * composite, 1
            )

        return group_scores

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score an entire DataFrame. Returns df with score columns added."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        available_features = [c for c in FIGHT_IQ_FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0)
        scores = self.model.predict(X)
        df = df.copy()
        df["fight_iq_score"] = np.clip(scores, 0, 100).round(1)
        return df

    # ── Serialization ─────────────────────────────────────

    def serialize(self) -> bytes:
        """Serialize model + metadata for storage."""
        payload = {
            "model": self.model,
            "version": self.version,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> "FightIQModel":
        """Load model from serialized bytes."""
        payload = pickle.loads(data)
        instance = cls()
        instance.model = payload["model"]
        instance.version = payload["version"]
        instance.metrics = payload.get("metrics", {})
        instance.feature_importances = payload.get("feature_importances", {})
        return instance
