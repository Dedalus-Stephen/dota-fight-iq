"""
Compute Ability & Item Usage Benchmarks

Reads fight_player_stats (joined with teamfights for context),
computes per-hero ability/item patterns, writes to new benchmark tables.

Usage:
    python -m scripts.compute_ability_benchmarks
    python -m scripts.compute_ability_benchmarks --dry-run
    python -m scripts.compute_ability_benchmarks --abilities-only
    python -m scripts.compute_ability_benchmarks --items-only
    python -m scripts.compute_ability_benchmarks --kills-only
"""

import argparse
import logging
import time

import pandas as pd

from app.core.database import get_supabase
from app.ml.ability_benchmarks import (
    compute_ability_usage_benchmarks,
    compute_item_usage_benchmarks,
    compute_kill_priority_benchmarks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ability_benchmarks")


def _fetch_merged_fight_data() -> pd.DataFrame:
    """
    Fetch fight_player_stats joined with teamfights context.

    fight_player_stats has: teamfight_id, match_id, hero_id, ability_uses, item_uses, killed
    teamfights has: id, match_id, fight_index, start_time, end_time, duration, deaths_count

    We join on fight_player_stats.teamfight_id = teamfights.id
    """
    sb = get_supabase()

    # Step 1: Fetch all fight_player_stats
    logger.info("Fetching fight_player_stats...")
    all_stats = []
    page_size = 1000
    offset = 0
    while True:
        batch = (
            sb.table("fight_player_stats")
            .select("teamfight_id,match_id,hero_id,ability_uses,item_uses,killed")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        all_stats.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    logger.info(f"  Fetched {len(all_stats)} fight_player_stats rows")

    if not all_stats:
        return pd.DataFrame()

    # Step 2: Fetch all teamfights
    logger.info("Fetching teamfights...")
    all_fights = []
    offset = 0
    while True:
        batch = (
            sb.table("teamfights")
            .select("id,match_id,fight_index,start_time,end_time,duration,deaths_count")
            .range(offset, offset + page_size - 1)
            .execute()
            .data
        )
        all_fights.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    logger.info(f"  Fetched {len(all_fights)} teamfight rows")

    if not all_fights:
        return pd.DataFrame()

    # Step 3: Join in Python (fight_player_stats.teamfight_id = teamfights.id)
    stats_df = pd.DataFrame(all_stats)
    fights_df = pd.DataFrame(all_fights)

    # teamfights.id → rename to teamfight_id for the join
    fights_df = fights_df.rename(columns={"id": "teamfight_id"})

    merged = stats_df.merge(
        fights_df[["teamfight_id", "start_time", "duration", "deaths_count"]],
        on="teamfight_id",
        how="left",
    )

    logger.info(
        f"  Merged: {len(merged)} rows, "
        f"{merged['hero_id'].nunique()} unique heroes"
    )
    return merged


def run_abilities(dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("ABILITY USAGE BENCHMARKS")
    logger.info("=" * 60)

    t0 = time.time()
    merged = _fetch_merged_fight_data()
    if merged.empty:
        logger.warning("No data found.")
        return

    benchmarks = compute_ability_usage_benchmarks(merged)

    if not dry_run and benchmarks:
        sb = get_supabase()
        for i in range(0, len(benchmarks), 500):
            sb.table("ability_usage_benchmarks").upsert(
                benchmarks[i:i + 500],
                on_conflict="hero_id,ability_key,time_bucket,size_bucket,patch"
            ).execute()
        logger.info(f"Upserted {len(benchmarks)} ability benchmarks in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Dry run: {len(benchmarks)} ability benchmarks computed")


def run_items(dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("ITEM USAGE BENCHMARKS")
    logger.info("=" * 60)

    t0 = time.time()
    merged = _fetch_merged_fight_data()
    if merged.empty:
        logger.warning("No data found.")
        return

    benchmarks = compute_item_usage_benchmarks(merged)

    if not dry_run and benchmarks:
        sb = get_supabase()
        for i in range(0, len(benchmarks), 500):
            sb.table("item_usage_benchmarks").upsert(
                benchmarks[i:i + 500],
                on_conflict="hero_id,item_key,time_bucket,size_bucket,patch"
            ).execute()
        logger.info(f"Upserted {len(benchmarks)} item benchmarks in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Dry run: {len(benchmarks)} item benchmarks computed")


def run_kills(dry_run: bool = False):
    logger.info("=" * 60)
    logger.info("KILL PRIORITY BENCHMARKS")
    logger.info("=" * 60)

    t0 = time.time()
    merged = _fetch_merged_fight_data()
    if merged.empty:
        logger.warning("No data found.")
        return

    benchmarks = compute_kill_priority_benchmarks(merged)

    if not dry_run and benchmarks:
        sb = get_supabase()
        for i in range(0, len(benchmarks), 500):
            sb.table("kill_priority_benchmarks").upsert(
                benchmarks[i:i + 500],
                on_conflict="hero_id,target_hero_id,time_bucket,patch"
            ).execute()
        logger.info(f"Upserted {len(benchmarks)} kill priority benchmarks in {time.time() - t0:.1f}s")
    else:
        logger.info(f"Dry run: {len(benchmarks)} kill priority benchmarks computed")


def main():
    parser = argparse.ArgumentParser(description="Compute ability/item usage benchmarks")
    parser.add_argument("--dry-run", action="store_true", help="Compute only, don't write to DB")
    parser.add_argument("--abilities-only", action="store_true")
    parser.add_argument("--items-only", action="store_true")
    parser.add_argument("--kills-only", action="store_true")
    args = parser.parse_args()

    run_all = not (args.abilities_only or args.items_only or args.kills_only)

    if run_all or args.abilities_only:
        run_abilities(args.dry_run)
    if run_all or args.items_only:
        run_items(args.dry_run)
    if run_all or args.kills_only:
        run_kills(args.dry_run)

    logger.info("Done!")


if __name__ == "__main__":
    main()