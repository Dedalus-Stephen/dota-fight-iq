"""
Compute Extended Benchmarks

Runs the full benchmark aggregation pipeline across all dimensions:
    - Fight (existing, enhanced with targeting)
    - Laning
    - Farming
    - Itemization
    - Objectives
    - Support

Usage:
    python -m scripts.compute_extended_benchmarks
    python -m scripts.compute_extended_benchmarks --laning-only
    python -m scripts.compute_extended_benchmarks --farming-only
    python -m scripts.compute_extended_benchmarks --items-only
    python -m scripts.compute_extended_benchmarks --objectives-only
    python -m scripts.compute_extended_benchmarks --support-only
    python -m scripts.compute_extended_benchmarks --targeting-only
    python -m scripts.compute_extended_benchmarks --dry-run
"""

import argparse
import logging
import time

import pandas as pd

from app.core import database as db
from app.core.database import get_supabase
from app.ml.extended_benchmarks import (
    compute_laning_benchmarks,
    compute_farming_benchmarks,
    compute_item_timing_benchmarks,
    compute_objective_benchmarks,
    compute_support_benchmarks,
    compute_fight_targeting_benchmarks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("extended_benchmarks")


def _fetch_paginated(table: str, filters: dict | None = None) -> list[dict]:
    """Fetch all rows from a table with pagination."""
    sb = get_supabase()
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        query = sb.table(table).select("*").range(offset, offset + page_size - 1)
        if filters:
            for key, val in filters.items():
                query = query.eq(key, val)
        batch = query.execute().data
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    return all_rows


def step_laning(dry_run: bool = False):
    """Compute laning phase benchmarks."""
    logger.info("=" * 60)
    logger.info("LANING BENCHMARKS")
    logger.info("=" * 60)

    rows = _fetch_paginated("laning_analysis")
    if not rows:
        logger.warning("No laning_analysis data found. Run backfill_v2 first.")
        return

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} laning records across {df['hero_id'].nunique()} heroes")

    benchmarks = compute_laning_benchmarks(df)

    if not dry_run and benchmarks:
        for i in range(0, len(benchmarks), 500):
            db.upsert_benchmarks(benchmarks[i:i+500])
        logger.info(f"Upserted {len(benchmarks)} laning benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} laning benchmarks")


def step_farming(dry_run: bool = False):
    """Compute farming efficiency benchmarks."""
    logger.info("=" * 60)
    logger.info("FARMING BENCHMARKS")
    logger.info("=" * 60)

    rows = _fetch_paginated("farming_analysis")
    if not rows:
        logger.warning("No farming_analysis data found.")
        return

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} farming records across {df['hero_id'].nunique()} heroes")

    benchmarks = compute_farming_benchmarks(df)

    if not dry_run and benchmarks:
        for i in range(0, len(benchmarks), 500):
            db.upsert_benchmarks(benchmarks[i:i+500])
        logger.info(f"Upserted {len(benchmarks)} farming benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} farming benchmarks")


def step_items(dry_run: bool = False):
    """Compute item timing benchmarks."""
    logger.info("=" * 60)
    logger.info("ITEM TIMING BENCHMARKS")
    logger.info("=" * 60)

    rows = _fetch_paginated("itemization_analysis")
    if not rows:
        logger.warning("No itemization_analysis data found.")
        return

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} itemization records across {df['hero_id'].nunique()} heroes")

    benchmarks = compute_item_timing_benchmarks(df)

    if not dry_run and benchmarks:
        sb = get_supabase()
        for i in range(0, len(benchmarks), 500):
            sb.table("item_timing_benchmarks").upsert(
                benchmarks[i:i+500],
                on_conflict="hero_id,item_key,role,patch"
            ).execute()
        logger.info(f"Upserted {len(benchmarks)} item timing benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} item timing benchmarks")


def step_objectives(dry_run: bool = False):
    """Compute objective timing benchmarks."""
    logger.info("=" * 60)
    logger.info("OBJECTIVE BENCHMARKS")
    logger.info("=" * 60)

    rows = _fetch_paginated("match_objectives")
    if not rows:
        logger.warning("No match_objectives data found.")
        return

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} objective events")

    benchmarks = compute_objective_benchmarks(df)

    if not dry_run and benchmarks:
        sb = get_supabase()
        sb.table("objective_benchmarks").upsert(
            benchmarks,
            on_conflict="objective_type,patch"
        ).execute()
        logger.info(f"Upserted {len(benchmarks)} objective benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} objective benchmarks")


def step_support(dry_run: bool = False):
    """Compute support efficiency benchmarks."""
    logger.info("=" * 60)
    logger.info("SUPPORT BENCHMARKS")
    logger.info("=" * 60)

    players = _fetch_paginated("match_players")
    wards = _fetch_paginated("ward_events")

    if not players:
        logger.warning("No match_players data found.")
        return

    players_df = pd.DataFrame(players)
    wards_df = pd.DataFrame(wards) if wards else None
    logger.info(f"Loaded {len(players_df)} player records, {len(wards) if wards else 0} ward events")

    benchmarks = compute_support_benchmarks(players_df, wards_df)

    if not dry_run and benchmarks:
        for i in range(0, len(benchmarks), 500):
            db.upsert_benchmarks(benchmarks[i:i+500])
        logger.info(f"Upserted {len(benchmarks)} support benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} support benchmarks")


def step_targeting(dry_run: bool = False):
    """Compute fight targeting benchmarks."""
    logger.info("=" * 60)
    logger.info("FIGHT TARGETING BENCHMARKS")
    logger.info("=" * 60)

    rows = _fetch_paginated("fight_player_stats")
    if not rows:
        logger.warning("No fight_player_stats data found.")
        return

    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} fight player stats")

    benchmarks = compute_fight_targeting_benchmarks(df)

    if not dry_run and benchmarks:
        for i in range(0, len(benchmarks), 500):
            db.upsert_benchmarks(benchmarks[i:i+500])
        logger.info(f"Upserted {len(benchmarks)} targeting benchmarks")
    else:
        logger.info(f"Dry run: would upsert {len(benchmarks)} targeting benchmarks")


def main():
    parser = argparse.ArgumentParser(description="Compute extended benchmarks for Dota Fight IQ")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--laning-only", action="store_true")
    parser.add_argument("--farming-only", action="store_true")
    parser.add_argument("--items-only", action="store_true")
    parser.add_argument("--objectives-only", action="store_true")
    parser.add_argument("--support-only", action="store_true")
    parser.add_argument("--targeting-only", action="store_true")
    args = parser.parse_args()

    start = time.time()
    logger.info("Starting extended benchmark computation")

    run_all = not any([
        args.laning_only, args.farming_only, args.items_only,
        args.objectives_only, args.support_only, args.targeting_only,
    ])

    if run_all or args.laning_only:
        step_laning(dry_run=args.dry_run)

    if run_all or args.farming_only:
        step_farming(dry_run=args.dry_run)

    if run_all or args.items_only:
        step_items(dry_run=args.dry_run)

    if run_all or args.objectives_only:
        step_objectives(dry_run=args.dry_run)

    if run_all or args.support_only:
        step_support(dry_run=args.dry_run)

    if run_all or args.targeting_only:
        step_targeting(dry_run=args.dry_run)

    elapsed = time.time() - start
    logger.info(f"Extended benchmarks complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()