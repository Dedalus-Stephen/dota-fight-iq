"""
Fight Archetype Clustering — DBSCAN

Discovers natural fight types from structural features:
    - Smoke ganks (low player count, short duration, high kill concentration)
    - Highground sieges (late game, long duration, many deaths)
    - Roshan fights (specific timing patterns)
    - Open 5v5 (many active players, balanced kills)
    - Pickoffs (1-2 deaths, very short)
    etc.

DBSCAN is chosen because:
    - Doesn't require pre-specifying number of clusters
    - Handles noise (outlier fights that don't fit any type)
    - Density-based: finds clusters of arbitrary shape

After clustering, each cluster gets a human-readable label
based on its centroid characteristics.
"""

import logging
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from app.ml import MODEL_NAMES
from app.ml.feature_engineering import CLUSTERING_FEATURE_COLS

logger = logging.getLogger(__name__)


# ── Auto-labeling rules ──────────────────────────────────

def _label_cluster(centroid: dict) -> str:
    """
    Assign a human-readable archetype label based on cluster centroid features.
    These are heuristic rules — they'll be refined as we see real clusters.
    """
    deaths = centroid.get("deaths_count", 0)
    duration = centroid.get("duration_sec", 0)
    active = centroid.get("active_players", 0)
    time_min = centroid.get("fight_time_minutes", 0)
    dmg_share = centroid.get("max_damage_share", 0)
    kill_imb = centroid.get("kill_imbalance", 0)

    # Pickoff: very few deaths, short, few players
    if deaths <= 2 and active <= 4 and duration < 10:
        return "pickoff"

    # Smoke gank: moderate deaths, short, one side dominates
    if deaths <= 4 and kill_imb > 0.6 and duration < 15:
        return "smoke_gank"

    # Highground siege: late game, long, many deaths
    if time_min > 30 and duration > 15 and deaths >= 5:
        return "highground_siege"

    # Early skirmish: early game, short
    if time_min < 15 and deaths <= 4:
        return "early_skirmish"

    # Open 5v5: many active players, balanced kills
    if active >= 7 and kill_imb < 0.3:
        return "open_5v5"

    # Decisive wipe: many deaths, one-sided
    if deaths >= 6 and kill_imb > 0.5:
        return "decisive_wipe"

    # Roshan contest: mid-late game, moderate duration
    # (We don't have Rosh proximity data yet — this is a rough heuristic)
    if 20 < time_min < 40 and 4 <= deaths <= 7 and 10 < duration < 25:
        return "roshan_contest"

    # Default
    return "teamfight"


class FightClusteringModel:
    """DBSCAN-based fight archetype discovery."""

    def __init__(self):
        self.dbscan: DBSCAN | None = None
        self.scaler: StandardScaler | None = None
        self.cluster_labels: dict[int, str] = {}  # cluster_id → archetype name
        self.cluster_centroids: dict[int, dict] = {}
        self.version: str = ""
        self.metrics: dict = {}

    def train(
        self,
        df: pd.DataFrame,
        eps: float = 0.8,
        min_samples: int = 15,
    ) -> dict:
        """
        Run DBSCAN clustering on fight structural features.

        Args:
            df:          DataFrame with CLUSTERING_FEATURE_COLS
            eps:         DBSCAN neighborhood radius (tune this)
            min_samples: minimum cluster size

        Returns:
            Training metrics dict
        """
        logger.info(f"Running DBSCAN clustering on {len(df)} fights")

        available_features = [c for c in CLUSTERING_FEATURE_COLS if c in df.columns]
        X = df[available_features].fillna(0).values

        if len(X) < min_samples * 2:
            raise ValueError(
                f"Not enough data for clustering: {len(X)} fights "
                f"(need at least {min_samples * 2})"
            )

        # Standardize features (DBSCAN is distance-based)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Run DBSCAN
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = self.dbscan.fit_predict(X_scaled)

        df = df.copy()
        df["cluster_id"] = labels

        # Analyze clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_pct = n_noise / len(labels) * 100

        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points ({noise_pct:.1f}%)")

        # If too much noise or too few clusters, suggest tuning
        if noise_pct > 40:
            logger.warning(
                f"High noise rate ({noise_pct:.1f}%). Consider increasing eps or decreasing min_samples."
            )

        # Compute centroids and auto-label
        self.cluster_centroids = {}
        self.cluster_labels = {}

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                self.cluster_labels[-1] = "unclassified"
                continue

            cluster_df = df[df["cluster_id"] == cluster_id]
            centroid = {
                col: round(float(cluster_df[col].mean()), 2)
                for col in available_features
            }
            centroid["sample_count"] = len(cluster_df)
            self.cluster_centroids[cluster_id] = centroid
            self.cluster_labels[cluster_id] = _label_cluster(centroid)

        self.version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        self.metrics = {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_pct": round(noise_pct, 1),
            "total_samples": len(df),
            "eps": eps,
            "min_samples": min_samples,
            "cluster_sizes": {
                self.cluster_labels.get(cid, f"cluster_{cid}"): int(count)
                for cid, count in df["cluster_id"].value_counts().items()
            },
        }

        logger.info(f"Cluster archetypes: {self.cluster_labels}")
        return self.metrics

    def predict(self, features: dict) -> str:
        """
        Assign a fight archetype to a new fight.

        Since DBSCAN doesn't have a native predict method,
        we find the nearest cluster centroid.
        """
        if not self.cluster_centroids or self.scaler is None:
            raise RuntimeError("Model not trained or loaded")

        available_features = [c for c in CLUSTERING_FEATURE_COLS if c in features]
        x = np.array([[features.get(c, 0) for c in available_features]])
        x_scaled = self.scaler.transform(x)[0]

        # Find nearest centroid
        best_cluster = -1
        best_dist = float("inf")

        for cluster_id, centroid in self.cluster_centroids.items():
            centroid_vals = np.array([centroid.get(c, 0) for c in available_features])
            centroid_scaled = self.scaler.transform(centroid_vals.reshape(1, -1))[0]
            dist = np.linalg.norm(x_scaled - centroid_scaled)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster_id

        # If too far from any centroid, mark as unclassified
        if best_dist > 3.0:  # threshold in scaled space
            return "unclassified"

        return self.cluster_labels.get(best_cluster, "unclassified")

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        """Assign archetypes to a DataFrame of fights."""
        return pd.Series(
            [self.predict(row.to_dict()) for _, row in df.iterrows()],
            index=df.index,
        )

    def get_archetype_summary(self) -> list[dict]:
        """Return summary of discovered archetypes for documentation/API."""
        summary = []
        for cluster_id, label in self.cluster_labels.items():
            if cluster_id == -1:
                continue
            centroid = self.cluster_centroids.get(cluster_id, {})
            summary.append({
                "archetype": label,
                "cluster_id": cluster_id,
                "avg_deaths": centroid.get("deaths_count", 0),
                "avg_duration_sec": centroid.get("duration_sec", 0),
                "avg_active_players": centroid.get("active_players", 0),
                "avg_game_time_min": centroid.get("fight_time_minutes", 0),
                "sample_count": centroid.get("sample_count", 0),
            })
        return summary

    # ── Serialization ─────────────────────────────────────

    def serialize(self) -> bytes:
        payload = {
            "dbscan": self.dbscan,
            "scaler": self.scaler,
            "cluster_labels": self.cluster_labels,
            "cluster_centroids": self.cluster_centroids,
            "version": self.version,
            "metrics": self.metrics,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> "FightClusteringModel":
        payload = pickle.loads(data)
        instance = cls()
        instance.dbscan = payload["dbscan"]
        instance.scaler = payload["scaler"]
        instance.cluster_labels = payload["cluster_labels"]
        instance.cluster_centroids = payload["cluster_centroids"]
        instance.version = payload.get("version", "")
        instance.metrics = payload.get("metrics", {})
        return instance
