"""
ML Pipeline — Phase 2

Modules:
    feature_engineering  — Extract ML features from fight_player_stats rows
    benchmarks           — Compute percentile benchmarks per hero per context
    fight_iq_model       — XGBoost regression: Fight IQ Score 0-100
    fight_outcome_model  — XGBoost classification: fight win probability
    clustering           — DBSCAN fight archetype discovery
"""

# ── Shared Constants ──────────────────────────────────────

TIME_BUCKETS = ["0-15", "15-25", "25-35", "35-45", "45+"]
NW_BUCKETS = ["below_avg", "average", "above_avg", "far_ahead"]
DURATION_BUCKETS = ["short", "medium", "long"]       # <10s, 10-20s, 20s+
SIZE_BUCKETS = ["skirmish", "teamfight", "bloodbath"]  # 2-4, 5-7, 8+ deaths

# Fight IQ sub-score weights (must sum to 1.0)
SCORE_WEIGHTS = {
    "ability": 0.30,
    "damage": 0.25,
    "item": 0.20,
    "survival": 0.15,
    "extraction": 0.10,
}

# Minimum samples required for a benchmark bucket to be considered reliable
MIN_BENCHMARK_SAMPLES = 10

MODEL_NAMES = {
    "fight_iq": "fight_iq_xgboost",
    "fight_outcome": "fight_outcome_xgboost",
    "clustering": "fight_archetypes_dbscan",
}
