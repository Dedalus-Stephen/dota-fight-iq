"""
items_enhanced.py — Enhanced items analysis endpoint.

Adds two new features to the existing items analysis:
1. Purchase timeline with 7k+ median timestamps per item
2. ML-recommended item build progression (inventory snapshots)

This module provides functions that can be integrated into the existing
analysis_service.py or used as standalone route enhancements.
"""

from collections import defaultdict
from app.ml.build_recommender import get_recommended_item_build


def enrich_purchase_timeline(
    db,
    purchase_log: list[dict],
    hero_id: int,
    position: int | None = None,
) -> list[dict]:
    """
    Enrich each purchase log entry with the 7k+ median purchase time for that item.
    
    Input purchase_log format: [{time: seconds, key: "item_name"}, ...]
    
    Returns enriched list:
    [
        {
            "time": 600,
            "key": "power_treads",
            "display_name": "Power Treads",
            "median_7k_time": 540,
            "delta_seconds": 60,       # positive = slower than 7k+
            "percentile": 35,
            "is_major": True,          # cost >= 1000g
        },
        ...
    ]
    """
    if not purchase_log:
        return []

    # Fetch item timing benchmarks for this hero
    query = db.table("item_timing_benchmarks").select("*").eq("hero_id", hero_id)
    if position:
        # Try position-specific first
        pos_result = query.eq("position", position).execute()
        if pos_result.data and len(pos_result.data) >= 3:
            benchmarks = {b["item_key"]: b for b in pos_result.data}
        else:
            # Fall back to all-position
            all_result = (
                db.table("item_timing_benchmarks")
                .select("*")
                .eq("hero_id", hero_id)
                .is_("position", "null")
                .execute()
            )
            benchmarks = {b["item_key"]: b for b in (all_result.data or [])}
    else:
        result = query.execute()
        benchmarks = {b["item_key"]: b for b in (result.data or [])}

    enriched = []
    for purchase in purchase_log:
        item_key = purchase.get("key", "")
        purchase_time = purchase.get("time", 0)

        bench = benchmarks.get(item_key)

        entry = {
            "time": purchase_time,
            "key": item_key,
            "display_name": _format_item_name(item_key),
            "median_7k_time": None,
            "delta_seconds": None,
            "percentile": None,
            "is_major": _is_major_item(item_key),
            "purchase_rate": None,
        }

        if bench:
            median = bench.get("median_time")
            if median is not None and purchase_time >= 0:
                entry["median_7k_time"] = median
                entry["delta_seconds"] = purchase_time - median
                entry["purchase_rate"] = bench.get("purchase_rate")

                # Compute approximate percentile
                p25 = bench.get("p25_time", median)
                p75 = bench.get("p75_time", median)
                entry["percentile"] = _estimate_percentile(purchase_time, p25, median, p75)

        enriched.append(entry)

    return enriched


def get_recommended_build_for_player(
    db,
    hero_id: int,
    position: int,
    enemy_hero_ids: list[int],
    ally_hero_ids: list[int],
) -> dict:
    """
    Get the ML-recommended item build progression for a player's situation.
    
    Returns inventory snapshots showing what a 7k+ player would build.
    """
    return get_recommended_item_build(
        db, hero_id, position, enemy_hero_ids, ally_hero_ids, top_k=20
    )


def _estimate_percentile(value: int, p25: int, median: int, p75: int) -> float:
    """Estimate percentile for item timing (lower = better, so invert)."""
    if p25 == p75:
        return 50.0

    # For items, faster (lower time) is better
    if value <= p25:
        raw = 75 + 25 * (p25 - value) / max(p25, 1)
    elif value <= median:
        raw = 50 + 25 * (median - value) / max(median - p25, 1)
    elif value <= p75:
        raw = 25 + 25 * (p75 - value) / max(p75 - median, 1)
    else:
        raw = max(0, 25 * (1 - (value - p75) / max(p75, 1)))

    return round(min(100, max(0, raw)), 1)


# Items that cost >= 1000 gold are considered "major"
MAJOR_ITEMS = {
    "blink", "force_staff", "black_king_bar", "desolator", "butterfly",
    "monkey_king_bar", "daedalus", "assault", "heart", "satanic",
    "skadi", "abyssal_blade", "manta", "orchid", "bloodthorn",
    "nullifier", "aghanims_scepter", "aghanims_shard", "refresher",
    "shivas_guard", "linkens_sphere", "lotus_orb", "aeon_disk",
    "hurricane_pike", "dragon_lance", "echo_sabre", "maelstrom",
    "mjollnir", "diffusal_blade", "sange_and_yasha", "kaya_and_sange",
    "yasha_and_kaya", "ethereal_blade", "radiance", "battle_fury",
    "hand_of_midas", "helm_of_the_dominator", "helm_of_the_overlord",
    "vladmir", "pipe", "crimson_guard", "guardian_greaves",
    "arcane_boots", "power_treads", "phase_boots", "boots_of_travel",
    "boots_of_travel_2", "solar_crest", "rod_of_atos", "gleipnir",
    "witch_blade", "falcon_blade", "pavise", "veil_of_discord",
    "spirit_vessel", "urn_of_shadows", "euls_scepter", "glimmer_cape",
    "shadow_blade", "silver_edge", "heavens_halberd", "hood_of_defiance",
    "eternal_shroud", "blade_mail", "armlet", "mask_of_madness",
    "crystalys", "meteor_hammer",
}


def _is_major_item(item_key: str) -> bool:
    """Check if an item is considered a major purchase."""
    clean = item_key.replace("item_", "")
    return clean in MAJOR_ITEMS


def _format_item_name(raw: str) -> str:
    """Format item key to display name."""
    if not raw:
        return "Empty"
    return raw.replace("item_", "").replace("_", " ").title()