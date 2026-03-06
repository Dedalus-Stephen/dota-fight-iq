"""
ML Training Pipeline

CLI script to run the full Phase 2 training pipeline:
    1. Fetch fight data from Supabase
    2. Extract features
    3. Compute benchmarks → upsert to hero_benchmarks
    4. Train Fight IQ model → store in storage
    5. Train Fight Outcome model → store in storage
    6. Run DBSCAN clustering → update teamfights.fight_archetype
    7. Build similarity vectors → upsert to fight_vectors

Usage:
    # Full pipeline
    python -m scripts.train_models

    # Individual steps
    python -m scripts.train_models --benchmarks-only
    python -m scripts.train_models --fight-iq-only
    python -m scripts.train_models --outcome-only
    python -m scripts.train_models --clustering-only
    python -m scripts.train_models --vectors-only

    # Dry run (compute but don't write to DB)
    python -m scripts.train_models --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time

import pandas as pd

from app.core import database as db
from app.core.storage import get_storage
from app.ml.feature_engineering import (
    build_training_dataframe,
    extract_fight_outcome_features,
    extract_clustering_features,
    build_similarity_vector,
    CLUSTERING_FEATURE_COLS,
    FIGHT_OUTCOME_FEATURE_COLS,
)
from app.ml.benchmarks import compute_benchmarks
from app.ml.fight_iq_model import FightIQModel
from app.ml.fight_outcome_model import FightOutcomeModel
from app.ml.clustering import FightClusteringModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_models")


# ── Data Fetching ─────────────────────────────────────────

def fetch_training_data() -> tuple[list, list, list, list]:
    """
    Fetch all processed data from Supabase for training.
    Returns (fights, fight_stats, matches, match_players).
    """
    sb = db.get_supabase()

    logger.info("Fetching matches...")
    matches_result = sb.table("matches").select("*").eq("is_parsed", True).execute()
    matches = matches_result.data
    logger.info(f"  → {len(matches)} matches")

    logger.info("Fetching teamfights...")
    fights_result = sb.table("teamfights").select("*").execute()
    fights = fights_result.data
    logger.info(f"  → {len(fights)} teamfights")

    logger.info("Fetching fight_player_stats...")
    # Paginate — Supabase default limit is 1000
    all_stats = []
    page_size = 1000
    offset = 0
    while True:
        batch = (
            sb.table("fight_player_stats")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        ).data
        all_stats.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    logger.info(f"  → {len(all_stats)} fight_player_stats rows")

    logger.info("Fetching match_players...")
    all_players = []
    offset = 0
    while True:
        batch = (
            sb.table("match_players")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        ).data
        all_players.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    logger.info(f"  → {len(all_players)} match_players rows")

    return fights, all_stats, matches, all_players


# ── Pipeline Steps ────────────────────────────────────────

def step_benchmarks(df: pd.DataFrame, dry_run: bool = False) -> list[dict]:
    """Step 1: Compute and store hero benchmarks."""
    logger.info("=" * 60)
    logger.info("STEP 1: Computing benchmarks")
    logger.info("=" * 60)

    benchmarks = compute_benchmarks(df)

    if not dry_run and benchmarks:
        logger.info(f"Upserting {len(benchmarks)} benchmark rows to Supabase...")
        # Batch upsert in chunks
        chunk_size = 500
        for i in range(0, len(benchmarks), chunk_size):
            chunk = benchmarks[i : i + chunk_size]
            db.upsert_benchmarks(chunk)
        logger.info("  → Benchmarks stored.")
    else:
        logger.info(f"  → Dry run: would upsert {len(benchmarks)} benchmark rows")

    # Summary stats
    heroes = set(b["hero_id"] for b in benchmarks)
    logger.info(
        f"  Benchmarks: {len(benchmarks)} rows across {len(heroes)} heroes"
    )
    return benchmarks


def step_fight_iq(df: pd.DataFrame, dry_run: bool = False) -> FightIQModel:
    """Step 2: Train Fight IQ Score model."""
    logger.info("=" * 60)
    logger.info("STEP 2: Training Fight IQ model")
    logger.info("=" * 60)

    model = FightIQModel()
    metrics = model.train(df)

    logger.info(f"  Metrics: {metrics}")
    logger.info(f"  Top features: {dict(list(model.feature_importances.items())[:5])}")

    if not dry_run:
        storage = get_storage()
        model_bytes = model.serialize()
        key = storage.store_model("fight_iq_xgboost", model.version, model_bytes)
        logger.info(f"  → Model stored: {key} ({len(model_bytes) / 1024:.1f} KB)")
    else:
        logger.info(f"  → Dry run: model not stored")

    return model


def step_fight_outcome(
    fights: list[dict],
    fight_stats: list[dict],
    matches: list[dict],
    match_players: list[dict],
    dry_run: bool = False,
) -> FightOutcomeModel:
    """Step 3: Train Fight Outcome Prediction model."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training Fight Outcome model")
    logger.info("=" * 60)

    # Build outcome features (per-fight, not per-player)
    match_map = {m["match_id"]: m for m in matches}
    players_by_match = {}
    for p in match_players:
        mid = p["match_id"]
        if mid not in players_by_match:
            players_by_match[mid] = []
        players_by_match[mid].append(p)

    stats_by_fight = {}
    for s in fight_stats:
        tid = s["teamfight_id"]
        if tid not in stats_by_fight:
            stats_by_fight[tid] = []
        stats_by_fight[tid].append(s)

    outcome_rows = []
    for fight in fights:
        fight_id = fight["id"]
        match_id = fight["match_id"]
        match = match_map.get(match_id)
        if not match:
            continue
        stats = stats_by_fight.get(fight_id, [])
        players = players_by_match.get(match_id, [])

        features = extract_fight_outcome_features(fight, stats, match, players)
        if features:
            features["teamfight_id"] = fight_id
            outcome_rows.append(features)

    outcome_df = pd.DataFrame(outcome_rows)
    logger.info(f"  Built {len(outcome_df)} fight outcome samples")

    model = FightOutcomeModel()
    metrics = model.train(outcome_df)
    logger.info(f"  Metrics: {metrics}")

    if not dry_run:
        storage = get_storage()
        model_bytes = model.serialize()
        key = storage.store_model("fight_outcome_xgboost", model.version, model_bytes)
        logger.info(f"  → Model stored: {key} ({len(model_bytes) / 1024:.1f} KB)")

    return model


def step_clustering(
    fights: list[dict],
    fight_stats: list[dict],
    dry_run: bool = False,
) -> FightClusteringModel:
    """Step 4: Run DBSCAN fight archetype clustering."""
    logger.info("=" * 60)
    logger.info("STEP 4: DBSCAN fight archetype clustering")
    logger.info("=" * 60)

    stats_by_fight = {}
    for s in fight_stats:
        tid = s["teamfight_id"]
        if tid not in stats_by_fight:
            stats_by_fight[tid] = []
        stats_by_fight[tid].append(s)

    cluster_rows = []
    for fight in fights:
        fight_id = fight["id"]
        stats = stats_by_fight.get(fight_id, [])
        features = extract_clustering_features(fight, stats)
        cluster_rows.append(features)

    cluster_df = pd.DataFrame(cluster_rows)
    logger.info(f"  Built {len(cluster_df)} clustering samples")

    model = FightClusteringModel()
    metrics = model.train(cluster_df)
    logger.info(f"  Metrics: {metrics}")
    logger.info(f"  Archetypes: {model.get_archetype_summary()}")

    if not dry_run:
        # Store model
        storage = get_storage()
        model_bytes = model.serialize()
        key = storage.store_model("fight_archetypes_dbscan", model.version, model_bytes)
        logger.info(f"  → Model stored: {key}")

        # Update teamfights table with archetype labels
        logger.info("  Updating teamfight archetypes in DB...")
        sb = db.get_supabase()
        for _, row in cluster_df.iterrows():
            fight_id = row.get("teamfight_id")
            if fight_id is None:
                continue
            archetype = model.predict(row.to_dict())
            try:
                sb.table("teamfights").update(
                    {"fight_archetype": archetype}
                ).eq("id", int(fight_id)).execute()
            except Exception as e:
                logger.warning(f"  Failed to update fight {fight_id}: {e}")
        logger.info("  → Archetypes written to teamfights table")

    return model


def step_vectors(
    df: pd.DataFrame,
    dry_run: bool = False,
) -> int:
    """Step 5: Build and store pgvector similarity vectors."""
    logger.info("=" * 60)
    logger.info("STEP 5: Building similarity vectors")
    logger.info("=" * 60)

    vectors = []
    for _, row in df.iterrows():
        features = row.to_dict()
        vec = build_similarity_vector(features)

        vectors.append({
            "teamfight_id": int(features.get("teamfight_id", 0)),
            "match_id": int(features.get("match_id", 0)),
            "hero_id": int(features.get("hero_id", 0)),
            "embedding": vec,
            "metadata": {
                "game_time": features.get("game_time"),
                "time_bucket": features.get("time_bucket"),
                "nw_bucket": features.get("nw_bucket"),
                "fight_size": features.get("fight_size"),
                "fight_iq_score": features.get("fight_iq_score"),
            },
        })

    logger.info(f"  Built {len(vectors)} similarity vectors (32-dim)")

    if not dry_run and vectors:
        sb = db.get_supabase()
        # Format embedding as pgvector string
        for v in vectors:
            v["embedding"] = f"[{','.join(str(x) for x in v['embedding'])}]"

        # Batch insert
        chunk_size = 500
        inserted = 0
        for i in range(0, len(vectors), chunk_size):
            chunk = vectors[i : i + chunk_size]
            try:
                sb.table("fight_vectors").upsert(
                    chunk, on_conflict="teamfight_id,hero_id"
                ).execute()
                inserted += len(chunk)
            except Exception as e:
                logger.warning(f"  Vector insert batch failed: {e}")
        logger.info(f"  → Inserted {inserted} vectors into fight_vectors")
    else:
        logger.info(f"  → Dry run: would insert {len(vectors)} vectors")

    return len(vectors)


# ── Main Pipeline ─────────────────────────────────────────

def run_pipeline(args):
    """Execute the full training pipeline."""
    start = time.time()
    logger.info("🚀 Starting ML training pipeline")
    logger.info(f"   Dry run: {args.dry_run}")

    # Fetch data
    fights, fight_stats, matches, match_players = fetch_training_data()

    if not fights:
        logger.error("No teamfight data found. Run data collection first.")
        sys.exit(1)

    # Build training DataFrame (per-player features)
    logger.info("Building training DataFrame...")
    df = build_training_dataframe(fights, fight_stats, matches, match_players)

    if len(df) < 50:
        logger.error(f"Only {len(df)} training samples. Need at least 50.")
        sys.exit(1)

    run_all = not any([
        args.benchmarks_only,
        args.fight_iq_only,
        args.outcome_only,
        args.clustering_only,
        args.vectors_only,
    ])

    # Step 1: Benchmarks
    if run_all or args.benchmarks_only:
        step_benchmarks(df, dry_run=args.dry_run)

    # Step 2: Fight IQ model
    fight_iq_model = None
    if run_all or args.fight_iq_only:
        fight_iq_model = step_fight_iq(df, dry_run=args.dry_run)

    # Step 3: Fight Outcome model
    if run_all or args.outcome_only:
        step_fight_outcome(fights, fight_stats, matches, match_players, dry_run=args.dry_run)

    # Step 4: Clustering
    if run_all or args.clustering_only:
        step_clustering(fights, fight_stats, dry_run=args.dry_run)

    # Step 5: Similarity vectors
    if run_all or args.vectors_only:
        # Add scores to df if we have the model
        if fight_iq_model:
            df = fight_iq_model.predict_batch(df)
        step_vectors(df, dry_run=args.dry_run)

    elapsed = time.time() - start
    logger.info(f"✅ Pipeline complete in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Dota Fight IQ — ML Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write to DB/storage")
    parser.add_argument("--benchmarks-only", action="store_true", help="Only compute benchmarks")
    parser.add_argument("--fight-iq-only", action="store_true", help="Only train Fight IQ model")
    parser.add_argument("--outcome-only", action="store_true", help="Only train outcome model")
    parser.add_argument("--clustering-only", action="store_true", help="Only run DBSCAN clustering")
    parser.add_argument("--vectors-only", action="store_true", help="Only build similarity vectors")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
