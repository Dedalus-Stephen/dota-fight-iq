"""
Fight Action Comparison API

GET /api/fights/{match_id}/{fight_index}/actions/{hero_id}

Returns side-by-side "what you did" vs "7k+ pattern" for a player in a fight.
Uses hero_id (not player_slot) since fight_player_stats doesn't have player_slot.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.core.database import get_supabase
from app.ml.ability_benchmarks import build_action_comparison
from app.ml.feature_engineering import time_bucket, size_bucket

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/fights", tags=["fights"])


@router.get("/{match_id}/{fight_index}/actions/{hero_id}")
async def get_fight_actions(match_id: int, fight_index: int, hero_id: int):
    """
    Get action comparison for a specific hero in a fight.

    Path params:
        match_id:    the match
        fight_index: 0-based fight index within the match
        hero_id:     the hero to analyze

    Returns:
        comparison, fight_context, inventory_at_fight, kill_timeline
    """
    sb = get_supabase()

    # 1. Get the teamfight record (need its id + context)
    fight_result = (
        sb.table("teamfights")
        .select("*")
        .eq("match_id", match_id)
        .eq("fight_index", fight_index)
        .execute()
    )
    if not fight_result.data:
        raise HTTPException(404, detail="Fight not found")
    fight = fight_result.data[0]
    teamfight_id = fight["id"]

    # 2. Get all fight_player_stats for this fight via teamfight_id
    stats_result = (
        sb.table("fight_player_stats")
        .select("*")
        .eq("teamfight_id", teamfight_id)
        .execute()
    )
    if not stats_result.data:
        raise HTTPException(404, detail="No fight stats found for this fight")

    # Find the player by hero_id
    player_stat = None
    for stat in stats_result.data:
        if stat.get("hero_id") == hero_id:
            player_stat = stat
            break

    if not player_stat:
        raise HTTPException(404, detail=f"Hero {hero_id} not found in this fight")

    # 3. Fight context
    fight_start = fight.get("start_time", 0)
    fight_end = fight.get("end_time", 0)
    fight_dur = fight.get("duration") or (fight_end - fight_start) or 1
    deaths_count = fight.get("deaths_count", 0)

    tb = time_bucket(fight_start)
    sb_val = size_bucket(deaths_count)

    fight_context = {
        "start_time": fight_start,
        "end_time": fight_end,
        "duration": fight_dur,
        "deaths_count": deaths_count,
        "time_bucket": tb,
        "size_bucket": sb_val,
    }

    # 4. Fetch ability benchmarks for this hero
    ability_benchmarks = _get_ability_benchmarks(sb, hero_id, tb, sb_val)

    # 5. Fetch item benchmarks
    item_benchmarks = _get_item_benchmarks(sb, hero_id)

    # 6. Fetch kill priority benchmarks
    kill_benchmarks = _get_kill_benchmarks(sb, hero_id)

    # 7. Build comparison
    comparison = build_action_comparison(
        player_fight_stat=player_stat,
        hero_id=hero_id,
        fight_context=fight_context,
        ability_benchmarks=ability_benchmarks,
        item_benchmarks=item_benchmarks,
        kill_benchmarks=kill_benchmarks,
    )

    # 8. Reconstruct inventory at fight time
    inventory = _reconstruct_inventory(sb, match_id, hero_id, fight_start)

    # 9. Kill timeline from kills_log
    kill_timeline = _extract_kill_timeline(sb, match_id, hero_id, fight_start, fight_end)

    return {
        "hero_id": hero_id,
        "fight_context": fight_context,
        "comparison": comparison,
        "inventory_at_fight": inventory,
        "kill_timeline": kill_timeline,
    }


def _get_ability_benchmarks(sb, hero_id: int, tb: str, sb_val: str) -> list[dict]:
    """Fetch ability benchmarks. Try context-specific, fall back to all."""
    result = (
        sb.table("ability_usage_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .eq("time_bucket", tb)
        .eq("size_bucket", sb_val)
        .execute()
    )
    if result.data and len(result.data) >= 2:
        return result.data

    result = (
        sb.table("ability_usage_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .eq("time_bucket", "all")
        .eq("size_bucket", "all")
        .execute()
    )
    return result.data or []


def _get_item_benchmarks(sb, hero_id: int) -> list[dict]:
    result = (
        sb.table("item_usage_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .execute()
    )
    return result.data or []


def _get_kill_benchmarks(sb, hero_id: int) -> list[dict]:
    result = (
        sb.table("kill_priority_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .eq("time_bucket", "all")
        .execute()
    )
    return result.data or []


def _reconstruct_inventory(sb, match_id: int, hero_id: int, fight_start: int) -> list[str]:
    """Reconstruct inventory at fight start from purchase_log."""
    result = (
        sb.table("match_players")
        .select("purchase_log, items")
        .eq("match_id", match_id)
        .eq("hero_id", hero_id)
        .execute()
    )
    if not result.data:
        return []

    player = result.data[0]
    purchase_log = player.get("purchase_log") or []

    if not purchase_log:
        items = player.get("items") or {}
        return [v for v in items.values() if v and v != 0]

    consumables = {
        "tango", "flask", "clarity", "enchanted_mango", "ward_observer",
        "ward_sentry", "tpscroll", "smoke_of_deceit", "dust", "famango",
        "faerie_fire", "branches", "circlet", "slippers", "mantle",
        "gauntlets", "quelling_blade", "stout_shield", "magic_stick",
        "ring_of_protection", "ring_of_regen", "sobi_mask",
    }

    major_items = [
        entry.get("key", "")
        for entry in purchase_log
        if entry.get("time", 0) <= fight_start
        and entry.get("key", "") not in consumables
        and not entry.get("key", "").startswith("recipe")
    ]

    return major_items[-6:] if len(major_items) > 6 else major_items


def _extract_kill_timeline(sb, match_id: int, hero_id: int, fight_start: int, fight_end: int) -> list[dict]:
    """Extract timestamped kills during fight from kills_log."""
    result = (
        sb.table("match_players")
        .select("kills_log")
        .eq("match_id", match_id)
        .eq("hero_id", hero_id)
        .execute()
    )
    if not result.data:
        return []

    kills_log = result.data[0].get("kills_log") or []
    fight_kills = []
    for kill in kills_log:
        t = kill.get("time", 0)
        if fight_start <= t <= fight_end:
            fight_kills.append({
                "time": t,
                "relative_time": t - fight_start,
                "target": kill.get("key", "").replace("npc_dota_hero_", ""),
            })

    return fight_kills