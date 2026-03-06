"""
Fight Outcome Prediction — XGBoost Classification

Binary classifier: given the game state at the moment a fight starts,
predict which team wins the fight.

This tells users: "Your team had a 65% chance of winning this fight
based on the state going in — but you lost it."

Separates strategic advantage from execution quality.
"""

import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from app.ml import MODEL_NAMES
from app.ml.feature_engineering import (
    extract_fight_outcome_features,
    FIGHT_OUTCOME_FEATURE_COLS,
)

logger = logging.getLogger(__name__)


class FightOutcomeModel:
    """XGBoost binary classifier for fight outcome prediction."""

    def __init__(self):
        self.model: xgb.XGBClassifier | None = None
        self.version: str = ""
        self.metrics: dict = {}
        self.feature_importances: dict = {}

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train fight outcome classifier.

        Args:
            df: DataFrame with FIGHT_OUTCOME_FEATURE_COLS + 'radiant_won_fight' label

        Returns:
            Training metrics dict
        """
        logger.info(f"Training fight outcome model on {len(df)} samples")

        label_col = "radiant_won_fight"
        if label_col not in df.columns:
            raise ValueError(f"Missing label column: {label_col}")

        available_features = [c for c in FIGHT_OUTCOME_FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0)
        y = df[label_col].astype(int)

        if len(X) < 50:
            raise ValueError(f"Not enough training data: {len(X)} samples (need 50+)")

        # Check class balance
        class_counts = y.value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Handle class imbalance
        scale_pos = float(len(y_train[y_train == 0])) / max(len(y_train[y_train == 1]), 1)

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            scale_pos_weight=scale_pos,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.0  # single class in test set

        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "auc": round(float(auc), 4),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": len(available_features),
            "class_distribution": dict(class_counts),
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
            f"Fight outcome model trained: accuracy={self.metrics['accuracy']}, "
            f"AUC={self.metrics['auc']}, version={self.version}"
        )
        return self.metrics

    def predict(self, features: dict) -> dict:
        """
        Predict fight outcome for a single fight.

        Args:
            features: dict from extract_fight_outcome_features()

        Returns:
            {radiant_win_prob, prediction, confidence, model_version}
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        available_features = [c for c in FIGHT_OUTCOME_FEATURE_COLS if c in features]
        X = np.array([[features.get(c, 0) for c in available_features]])

        proba = float(self.model.predict_proba(X)[0][1])
        prediction = "radiant" if proba >= 0.5 else "dire"
        confidence = abs(proba - 0.5) * 2  # 0 = coin flip, 1 = certain

        return {
            "radiant_win_prob": round(proba, 3),
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "model_version": self.version,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict outcomes for a DataFrame of fights."""
        if self.model is None:
            raise RuntimeError("Model not trained or loaded")

        available_features = [c for c in FIGHT_OUTCOME_FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0)
        probas = self.model.predict_proba(X)[:, 1]

        df = df.copy()
        df["radiant_win_prob"] = np.round(probas, 3)
        df["predicted_winner"] = np.where(probas >= 0.5, "radiant", "dire")
        return df

    # ── Serialization ─────────────────────────────────────

    def serialize(self) -> bytes:
        payload = {
            "model": self.model,
            "version": self.version,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> "FightOutcomeModel":
        payload = pickle.loads(data)
        instance = cls()
        instance.model = payload["model"]
        instance.version = payload["version"]
        instance.metrics = payload.get("metrics", {})
        instance.feature_importances = payload.get("feature_importances", {})
        return instance
