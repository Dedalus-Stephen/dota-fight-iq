"""
Scoring Service

Loads trained models into memory and provides inference for API endpoints.
Models are loaded once at startup from storage (local disk or S3).
When a new model version is deployed, workers pick it up on restart
or via periodic version check.

Usage in FastAPI:
    from app.ml.scoring import get_scorer
    scorer = get_scorer()
    result = scorer.score_fight(player_features, fight_features, context)
"""

import logging

from app.core.storage import get_storage
from app.ml.fight_iq_model import FightIQModel
from app.ml.fight_outcome_model import FightOutcomeModel
from app.ml.clustering import FightClusteringModel
from app.ml.benchmarks import compute_deltas, generate_recommendations
from app.ml.feature_engineering import (
    extract_player_fight_features,
    extract_fight_outcome_features,
    extract_clustering_features,
    build_similarity_vector,
)
from app.core import database as db

logger = logging.getLogger(__name__)


class ScoringService:
    """
    In-memory model serving for Fight IQ scoring.
    Loads the latest model versions from storage on init.
    """

    def __init__(self):
        self.fight_iq: FightIQModel | None = None
        self.fight_outcome: FightOutcomeModel | None = None
        self.clustering: FightClusteringModel | None = None
        self._loaded = False

    def load_models(self):
        """Load latest model versions from storage."""
        storage = get_storage()

        # Fight IQ model
        try:
            versions = storage.list_model_versions("fight_iq_xgboost")
            if versions:
                latest = sorted(versions)[-1]
                data = storage.get_model("fight_iq_xgboost", latest)
                if data:
                    self.fight_iq = FightIQModel.deserialize(data)
                    logger.info(f"Loaded Fight IQ model: {latest}")
        except Exception as e:
            logger.warning(f"Could not load Fight IQ model: {e}")

        # Fight Outcome model
        try:
            versions = storage.list_model_versions("fight_outcome_xgboost")
            if versions:
                latest = sorted(versions)[-1]
                data = storage.get_model("fight_outcome_xgboost", latest)
                if data:
                    self.fight_outcome = FightOutcomeModel.deserialize(data)
                    logger.info(f"Loaded Fight Outcome model: {latest}")
        except Exception as e:
            logger.warning(f"Could not load Fight Outcome model: {e}")

        # Clustering model
        try:
            versions = storage.list_model_versions("fight_archetypes_dbscan")
            if versions:
                latest = sorted(versions)[-1]
                data = storage.get_model("fight_archetypes_dbscan", latest)
                if data:
                    self.clustering = FightClusteringModel.deserialize(data)
                    logger.info(f"Loaded Clustering model: {latest}")
        except Exception as e:
            logger.warning(f"Could not load Clustering model: {e}")

        self._loaded = True
        logger.info(
            f"Scoring service ready: "
            f"fight_iq={'✓' if self.fight_iq else '✗'}, "
            f"outcome={'✓' if self.fight_outcome else '✗'}, "
            f"clustering={'✓' if self.clustering else '✗'}"
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def score_player_fight(
        self,
        fight_stat: dict,
        teamfight: dict,
        match: dict,
        all_players: list[dict],
    ) -> dict:
        """
        Full scoring for a single player's fight performance.

        Returns:
            {
                fight_iq_score, sub_scores,
                percentile_deltas, recommendations,
                fight_archetype, outcome_prediction,
                similarity_vector
            }
        """
        result = {}

        # Extract features
        player_features = extract_player_fight_features(
            fight_stat, teamfight, match, all_players
        )

        # Fight IQ Score
        if self.fight_iq:
            try:
                score_result = self.fight_iq.predict(player_features)
                result["fight_iq"] = score_result
            except Exception as e:
                logger.warning(f"Fight IQ scoring failed: {e}")
                result["fight_iq"] = None
        else:
            result["fight_iq"] = None

        # Benchmark comparison
        try:
            benchmarks_for_context = self._get_benchmarks(player_features)
            deltas = compute_deltas(player_features, benchmarks_for_context)
            recommendations = generate_recommendations(deltas)
            result["deltas"] = deltas
            result["recommendations"] = recommendations
        except Exception as e:
            logger.warning(f"Benchmark comparison failed: {e}")
            result["deltas"] = []
            result["recommendations"] = []

        # Fight archetype
        if self.clustering:
            try:
                cluster_features = extract_clustering_features(
                    teamfight,
                    [fight_stat],  # just this player's stats for context
                )
                archetype = self.clustering.predict(cluster_features)
                result["fight_archetype"] = archetype
            except Exception as e:
                logger.warning(f"Clustering prediction failed: {e}")
                result["fight_archetype"] = None
        else:
            result["fight_archetype"] = teamfight.get("fight_archetype")

        # Similarity vector
        result["similarity_vector"] = build_similarity_vector(player_features)

        return result

    def predict_fight_outcome(
        self,
        teamfight: dict,
        fight_stats: list[dict],
        match: dict,
        all_players: list[dict],
    ) -> dict | None:
        """Predict which team should win a fight based on pre-fight state."""
        if not self.fight_outcome:
            return None

        features = extract_fight_outcome_features(
            teamfight, fight_stats, match, all_players
        )
        if not features:
            return None

        try:
            return self.fight_outcome.predict(features)
        except Exception as e:
            logger.warning(f"Outcome prediction failed: {e}")
            return None

    def _get_benchmarks(self, player_features: dict) -> dict[str, dict]:
        """Fetch benchmarks for this player's context from DB."""
        hero_id = player_features.get("hero_id")
        t_bucket = player_features.get("time_bucket")
        n_bucket = player_features.get("nw_bucket")

        if not hero_id:
            return {}

        sb = db.get_supabase()
        result = (
            sb.table("hero_benchmarks")
            .select("*")
            .eq("hero_id", hero_id)
            .eq("time_bucket", t_bucket)
            .eq("nw_bucket", n_bucket)
            .execute()
        )

        return {row["metric_name"]: row for row in (result.data or [])}

    def get_model_info(self) -> dict:
        def sanitize(data):
            """Recursively convert numpy types to native python types."""
            if isinstance(data, dict):
                return {k: sanitize(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize(i) for i in data]
            elif hasattr(data, "item"): # This catches numpy scalars
                return data.item()
            return data

        return {
            "fight_iq": {
                "loaded": self.fight_iq is not None,
                "version": self.fight_iq.version if self.fight_iq else None,
                "metrics": sanitize(self.fight_iq.metrics) if self.fight_iq else None,
            },
            "fight_outcome": {
                "loaded": self.fight_outcome is not None,
                "version": self.fight_outcome.version if self.fight_outcome else None,
                "metrics": sanitize(self.fight_outcome.metrics) if self.fight_outcome else None,
            },
            "clustering": {
                "loaded": self.clustering is not None,
                "version": self.clustering.version if self.clustering else None,
                "metrics": sanitize(self.clustering.metrics) if self.clustering else None,
                "archetypes": sanitize(self.clustering.get_archetype_summary()) if self.clustering else None,
            },
    }


# ── Singleton ─────────────────────────────────────────────

_scorer: ScoringService | None = None


def get_scorer() -> ScoringService:
    """Get or create the scoring service singleton."""
    global _scorer
    if _scorer is None:
        _scorer = ScoringService()
        _scorer.load_models()
    return _scorer


def reload_models():
    """Force reload models (e.g., after retraining)."""
    global _scorer
    _scorer = ScoringService()
    _scorer.load_models()
    return _scorer
