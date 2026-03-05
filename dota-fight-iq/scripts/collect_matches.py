"""
Data Collection Script

Discovers high-MMR match IDs and processes them through the full pipeline.

Discovery strategy:
  1. OpenDota /proPlayers → get Immortal account IDs (one API call, hundreds of players)
  2. STRATZ player.matches → get recent ranked match IDs per player (50 matches per call)
  3. Deduplicate and store in match_collection_pool
  4. Process each match through the full pipeline (OpenDota fight data + STRATZ positions)

Usage:
    python -m scripts.collect_matches --discover --players 20
    python -m scripts.collect_matches --fetch-pending --limit 100
    python -m scripts.collect_matches --stats
    python -m scripts.collect_matches --retry-parses
    python -m scripts.collect_matches  (does discover + fetch)
"""

import asyncio
import argparse
import logging

from app.clients.opendota import OpenDotaClient
from app.clients.stratz import StratzClient
from app.core import database as db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def discover_matches(num_players: int = 20, matches_per_player: int = 50):
    """
    Step 1: Get pro/Immortal player IDs from OpenDota.
    Step 2: For each player, get their recent ranked matches from STRATZ.
    Step 3: Store unique match IDs in match_collection_pool.
    """
    opendota = OpenDotaClient()
    stratz = StratzClient()
    total_discovered = 0
    seen_match_ids = set()

    try:
        # Step 1: Get pro player list from OpenDota (1 API call → hundreds of players)
        logger.info("Fetching pro player list from OpenDota...")
        pro_players = await opendota.get_pro_players()

        if not pro_players:
            logger.error("Failed to fetch pro players from OpenDota")
            return 0

        # Filter to players with account IDs, sort by most recently active
        valid_players = [
            p for p in pro_players
            if p.get("account_id") and p.get("account_id") > 0
        ]
        logger.info(f"Found {len(valid_players)} pro players, using top {num_players}")

        # Take the requested number of players
        selected_players = valid_players[:num_players]

        # Step 2: For each player, discover their recent matches via STRATZ
        for i, player in enumerate(selected_players):
            account_id = player["account_id"]
            player_name = player.get("name") or player.get("personaname") or str(account_id)

            logger.info(f"[{i+1}/{num_players}] Discovering matches for {player_name} ({account_id})...")

            try:
                matches = await stratz.get_player_matches(
                    steam_id=account_id,
                    take=matches_per_player,
                )
            except Exception as e:
                logger.warning(f"  STRATZ failed for {player_name}: {e}")
                continue

            if not matches:
                logger.info(f"  No matches found for {player_name}")
                continue

            pool_entries = []
            for m in matches:
                match_id = m.get("id")
                if not match_id or match_id in seen_match_ids:
                    continue
                seen_match_ids.add(match_id)

                hero_ids_in_match = [
                    p.get("heroId") for p in m.get("players", []) if p.get("heroId")
                ]
                pool_entries.append({
                    "match_id": match_id,
                    "source": "stratz_player",
                    "hero_ids": hero_ids_in_match,
                    "avg_rank": 80,  # Pro players are Immortal
                    "status": "pending",
                })

            if pool_entries:
                db.insert_match_pool(pool_entries)
                total_discovered += len(pool_entries)
                logger.info(f"  → {len(pool_entries)} new matches from {player_name}")

            # Respect STRATZ rate limits
            await asyncio.sleep(0.3)

    finally:
        await opendota.close()
        await stratz.close()

    logger.info(f"Discovery complete: {total_discovered} unique matches from {num_players} players")
    return total_discovered


async def fetch_pending_matches(limit: int = 50):
    """
    Process pending matches from the collection pool.
    Each match goes through the full pipeline: OpenDota + STRATZ → extract → store.
    """
    from app.services.match_processor import MatchProcessor
    from app.core.database import _reset_client

    processor = MatchProcessor()

    pending = db.get_unprocessed_matches(limit=limit)
    logger.info(f"Found {len(pending)} pending matches to process")

    processed = 0
    failed = 0
    parse_requested = 0

    try:
        for i, entry in enumerate(pending):
            match_id = entry["match_id"]
            logger.info(f"[{i+1}/{len(pending)}] Processing match {match_id}...")

            try:
                db.update_match_pool_status(match_id, "fetching")
                result = await processor.process_match(match_id)

                if result["status"] == "parse_requested":
                    db.update_match_pool_status(match_id, "parse_requested")
                    parse_requested += 1
                    logger.info(f"  → Parse requested (not yet parsed on OpenDota)")
                elif result["status"] == "processed":
                    db.update_match_pool_status(match_id, "processed")
                    processed += 1
                    logger.info(
                        f"  → Processed: {result.get('fights', 0)} fights, "
                        f"{result.get('positions', 0)} positions, "
                        f"{result.get('wards', 0)} wards"
                    )

            except Exception as e:
                error_msg = str(e)
                if "ConnectionTerminated" in error_msg or "RemoteProtocolError" in error_msg:
                    logger.warning(f"  Connection error, resetting and retrying...")
                    _reset_client()
                    try:
                        result = await processor.process_match(match_id)
                        if result["status"] == "processed":
                            db.update_match_pool_status(match_id, "processed")
                            processed += 1
                            logger.info(f"  → Processed on retry")
                        else:
                            db.update_match_pool_status(match_id, result["status"])
                    except Exception as retry_err:
                        logger.error(f"  Failed on retry: {retry_err}")
                        db.update_match_pool_status(match_id, "failed")
                        failed += 1
                else:
                    logger.error(f"  Failed: {e}")
                    db.update_match_pool_status(match_id, "failed")
                    failed += 1

            # Respect OpenDota rate limits
            await asyncio.sleep(1.2)

            # Log progress every 10 matches
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {processed} processed, {parse_requested} parse_requested, "
                    f"{failed} failed out of {i+1} attempted"
                )

    finally:
        await processor.close()

    logger.info(
        f"\nCollection complete:\n"
        f"  Processed:       {processed}\n"
        f"  Parse requested: {parse_requested}\n"
        f"  Failed:          {failed}\n"
        f"  Total attempted: {processed + parse_requested + failed}"
    )
    return {"processed": processed, "parse_requested": parse_requested, "failed": failed}


async def retry_parse_requested(limit: int = 50):
    """
    Retry matches that were previously in 'parse_requested' status.
    OpenDota may have finished parsing them by now.
    """
    sb = db.get_supabase()
    result = (
        sb.table("match_collection_pool")
        .select("*")
        .eq("status", "parse_requested")
        .limit(limit)
        .execute()
    )
    if not result.data:
        logger.info("No parse_requested matches to retry")
        return

    count = 0
    for entry in result.data:
        db.update_match_pool_status(entry["match_id"], "pending")
        count += 1

    logger.info(f"Reset {count} parse_requested matches to pending for retry")


async def show_stats():
    """Show collection statistics."""
    sb = db.get_supabase()

    total = sb.table("match_collection_pool").select("match_id", count="exact").execute()
    pending = sb.table("match_collection_pool").select("match_id", count="exact").eq("status", "pending").execute()
    processed_pool = sb.table("match_collection_pool").select("match_id", count="exact").eq("status", "processed").execute()
    parse_req = sb.table("match_collection_pool").select("match_id", count="exact").eq("status", "parse_requested").execute()
    failed_pool = sb.table("match_collection_pool").select("match_id", count="exact").eq("status", "failed").execute()

    matches = sb.table("matches").select("match_id", count="exact").execute()
    fights = sb.table("teamfights").select("id", count="exact").execute()
    fight_stats = sb.table("fight_player_stats").select("id", count="exact").execute()
    positions = sb.table("player_positions").select("id", count="exact").execute()

    logger.info(
        f"\n{'='*50}\n"
        f"Collection Pool:\n"
        f"  Total discovered: {total.count}\n"
        f"  Pending:          {pending.count}\n"
        f"  Processed:        {processed_pool.count}\n"
        f"  Parse requested:  {parse_req.count}\n"
        f"  Failed:           {failed_pool.count}\n"
        f"\nStored Data:\n"
        f"  Matches:          {matches.count}\n"
        f"  Teamfights:       {fights.count}\n"
        f"  Fight samples:    {fight_stats.count}\n"
        f"  Position events:  {positions.count}\n"
        f"{'='*50}"
    )


async def main():
    parser = argparse.ArgumentParser(description="Dota Fight IQ Data Collection")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover new match IDs from pro players via OpenDota + STRATZ",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=20,
        help="Number of pro players to query for discovery (default: 20)",
    )
    parser.add_argument(
        "--matches-per-player",
        type=int,
        default=50,
        help="Matches to fetch per player during discovery (default: 50)",
    )
    parser.add_argument(
        "--fetch-pending",
        action="store_true",
        help="Process pending matches through the full pipeline",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max matches to process in this run (default: 50)",
    )
    parser.add_argument(
        "--retry-parses",
        action="store_true",
        help="Retry matches that needed parsing (may be ready now)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics",
    )

    args = parser.parse_args()

    if args.stats:
        await show_stats()
        return

    if args.retry_parses:
        await retry_parse_requested(limit=args.limit)

    if args.discover:
        await discover_matches(
            num_players=args.players,
            matches_per_player=args.matches_per_player,
        )

    if args.fetch_pending:
        await fetch_pending_matches(limit=args.limit)

    if not any([args.discover, args.fetch_pending, args.retry_parses, args.stats]):
        logger.info("Running full collection pipeline...")
        await discover_matches(num_players=args.players, matches_per_player=args.matches_per_player)
        await fetch_pending_matches(limit=args.limit)


if __name__ == "__main__":
    asyncio.run(main())