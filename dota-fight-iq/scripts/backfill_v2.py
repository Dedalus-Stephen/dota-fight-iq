"""
Backfill v2 Features

Re-processes existing matches stored in S3 through the new extraction pipeline
to populate the v2 tables (objectives, laning, itemization, farming, chat,
fight_context, enhanced wards, enhanced fight_player_stats).

Usage:
    python -m scripts.backfill_v2 --limit 100
    python -m scripts.backfill_v2 --match-id 8718181027
    python -m scripts.backfill_v2 --all
"""

import asyncio
import argparse
import logging

from app.core.storage import get_storage
from app.core import database as db
from app.core.database import get_supabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_processed_match_ids(limit: int | None = None) -> list[int]:
    """Get all match IDs that have been processed (have s3_key)."""
    sb = get_supabase()
    query = (
        sb.table("matches")
        .select("match_id, s3_key")
        .not_.is_("s3_key", "null")
        .order("match_id", desc=True)
    )
    if limit:
        query = query.limit(limit)
    result = query.execute()
    return [r["match_id"] for r in result.data]


async def backfill_match(match_id: int, storage) -> dict:
    """Re-extract v2 data for a single match from S3 raw JSON."""
    from app.services.match_processor import MatchProcessor

    # Load raw OpenDota JSON from S3/local storage
    od_data = storage.get_raw_match(match_id, source="opendota")
    if not od_data:
        return {"match_id": match_id, "status": "skipped", "reason": "no raw data"}

    processor = MatchProcessor()
    sb = get_supabase()
    results = {"match_id": match_id, "status": "ok"}

    # 1. Objectives
    try:
        sb.table("match_objectives").delete().eq("match_id", match_id).execute()
        objectives = processor._extract_objectives(od_data)
        if objectives:
            sb.table("match_objectives").insert(objectives).execute()
        results["objectives"] = len(objectives)
    except Exception as e:
        logger.warning(f"  Objectives failed: {e}")
        results["objectives"] = f"error: {e}"

    # 2. Laning
    try:
        sb.table("laning_analysis").delete().eq("match_id", match_id).execute()
        laning = processor._extract_laning(od_data)
        if laning:
            sb.table("laning_analysis").insert(laning).execute()
        results["laning"] = len(laning)
    except Exception as e:
        logger.warning(f"  Laning failed: {e}")
        results["laning"] = f"error: {e}"

    # 3. Itemization
    try:
        sb.table("itemization_analysis").delete().eq("match_id", match_id).execute()
        items = processor._extract_itemization(od_data)
        if items:
            sb.table("itemization_analysis").insert(items).execute()
        results["items"] = len(items)
    except Exception as e:
        logger.warning(f"  Itemization failed: {e}")
        results["items"] = f"error: {e}"

    # 4. Farming
    try:
        sb.table("farming_analysis").delete().eq("match_id", match_id).execute()
        farming = processor._extract_farming(od_data)
        if farming:
            sb.table("farming_analysis").insert(farming).execute()
        results["farming"] = len(farming)
    except Exception as e:
        logger.warning(f"  Farming failed: {e}")
        results["farming"] = f"error: {e}"

    # 5. Chat
    try:
        sb.table("chat_analysis").delete().eq("match_id", match_id).execute()
        chat = processor._extract_chat(od_data)
        if chat:
            sb.table("chat_analysis").insert(chat).execute()
        results["chat"] = len(chat)
    except Exception as e:
        logger.warning(f"  Chat failed: {e}")
        results["chat"] = f"error: {e}"

    # 6. Enhanced wards (replace existing)
    try:
        sb.table("ward_events").delete().eq("match_id", match_id).execute()
        wards = processor._extract_ward_details(od_data)
        if wards:
            sb.table("ward_events").insert(wards).execute()
        results["wards"] = len(wards)
    except Exception as e:
        logger.warning(f"  Wards failed: {e}")
        results["wards"] = f"error: {e}"

    # 7. Fight context (needs teamfight IDs from DB)
    try:
        sb.table("fight_context").delete().eq("match_id", match_id).execute()
        teamfights = (
            sb.table("teamfights")
            .select("id, fight_index, start_time, end_time")
            .eq("match_id", match_id)
            .order("fight_index")
            .execute()
        ).data

        if teamfights:
            players = od_data.get("players", [])
            contexts = []
            for tf in teamfights:
                try:
                    record = {
                        "start_time": tf["start_time"],
                        "end_time": tf["end_time"],
                    }
                    ctx = processor._build_fight_context(
                        od_data, record, tf["id"], players
                    )
                    if ctx:
                        contexts.append(ctx)
                except Exception:
                    pass
            if contexts:
                sb.table("fight_context").insert(contexts).execute()
            results["fight_context"] = len(contexts)
    except Exception as e:
        logger.warning(f"  Fight context failed: {e}")
        results["fight_context"] = f"error: {e}"

    # 8. Update fight_player_stats with ability_targets/damage_targets
    try:
        teamfights_raw = od_data.get("teamfights", [])
        updated_fps = 0
        for idx, fight in enumerate(teamfights_raw):
            fight_players = fight.get("players", [])
            for player_idx, fp in enumerate(fight_players):
                ability_targets = fp.get("ability_targets", {})
                damage_targets = fp.get("damage_targets", {})
                deaths_pos = fp.get("deaths_pos", {})

                if ability_targets or damage_targets or deaths_pos:
                    hero_id = (
                        od_data["players"][player_idx].get("hero_id")
                        if player_idx < len(od_data.get("players", []))
                        else None
                    )
                    if hero_id:
                        # Find the fight_player_stats row
                        tf_row = next(
                            (t for t in teamfights if t.get("fight_index") == idx),
                            None,
                        ) if teamfights else None

                        if tf_row:
                            sb.table("fight_player_stats").update({
                                "ability_targets": ability_targets,
                                "damage_targets": damage_targets,
                                "deaths_pos": deaths_pos,
                            }).eq(
                                "teamfight_id", tf_row["id"]
                            ).eq(
                                "hero_id", hero_id
                            ).execute()
                            updated_fps += 1

        results["fight_player_stats_updated"] = updated_fps
    except Exception as e:
        logger.warning(f"  Fight player stats update failed: {e}")
        results["fight_player_stats_updated"] = f"error: {e}"

    # 9. Update match_players with support stats
    try:
        for p in od_data.get("players", []):
            slot = p.get("player_slot")
            updates = {
                "obs_placed": p.get("obs_placed", 0),
                "sen_placed": p.get("sen_placed", 0),
                "camps_stacked": p.get("camps_stacked", 0),
                "creeps_stacked": p.get("creeps_stacked", 0),
                "teamfight_participation": p.get("teamfight_participation"),
                "stuns": p.get("stuns", 0),
                "rune_pickups": p.get("rune_pickups", 0),
                "actions_per_min": p.get("actions_per_min", 0),
            }
            sb.table("match_players").update(updates).eq(
                "match_id", match_id
            ).eq("player_slot", slot).execute()
        results["support_stats"] = "ok"
    except Exception as e:
        logger.warning(f"  Support stats failed: {e}")
        results["support_stats"] = f"error: {e}"

    await processor.close()
    return results


async def main():
    parser = argparse.ArgumentParser(description="Backfill v2 feature data")
    parser.add_argument("--limit", type=int, help="Max matches to process")
    parser.add_argument("--match-id", type=int, help="Process a single match")
    parser.add_argument("--all", action="store_true", help="Process all matches")
    args = parser.parse_args()

    storage = get_storage()

    if args.match_id:
        match_ids = [args.match_id]
    elif args.all:
        match_ids = get_processed_match_ids()
    else:
        match_ids = get_processed_match_ids(limit=args.limit or 10)

    logger.info(f"Backfilling {len(match_ids)} matches with v2 features")

    success = 0
    failed = 0

    for i, mid in enumerate(match_ids):
        logger.info(f"[{i+1}/{len(match_ids)}] Backfilling match {mid}")
        try:
            result = await backfill_match(mid, storage)
            if result["status"] == "ok":
                success += 1
                logger.info(f"  → Done: {result}")
            else:
                logger.info(f"  → Skipped: {result.get('reason')}")
        except Exception as e:
            failed += 1
            logger.error(f"  → Failed: {e}")

    logger.info(
        f"\nBackfill complete: {success} succeeded, {failed} failed, "
        f"{len(match_ids) - success - failed} skipped"
    )


if __name__ == "__main__":
    asyncio.run(main())