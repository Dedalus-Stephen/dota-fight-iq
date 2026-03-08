"""
build_extractor.py — Extract item build snapshots and ability builds from parsed matches.

This module processes the raw OpenDota match data (already stored in S3/Postgres)
and populates the new tables:
  - item_build_snapshots: inventory state after each major item completion
  - ability_builds: full ability upgrade order
  - match_context_vectors: encoded match context for similarity search

Run as part of the match processing pipeline or as a batch backfill.
"""

import logging
from collections import defaultdict
from app.ml.build_recommender import encode_match_context

logger = logging.getLogger(__name__)

# Items costing >= this threshold are "major" and trigger a snapshot
MAJOR_ITEM_COST_THRESHOLD = 900

# Map of item keys to approximate gold cost (for major item detection)
# In production, this would be loaded from a data file or API
ITEM_COSTS = {
    "power_treads": 1400, "phase_boots": 1500, "arcane_boots": 1300,
    "boots_of_travel": 2500, "guardian_greaves": 5375,
    "blink": 2250, "force_staff": 2175, "glimmer_cape": 1950,
    "black_king_bar": 4050, "desolator": 3500, "butterfly": 4975,
    "monkey_king_bar": 4975, "daedalus": 5150, "assault": 5250,
    "heart": 5200, "satanic": 5050, "skadi": 5300,
    "abyssal_blade": 6250, "manta": 4600, "orchid": 3475,
    "bloodthorn": 6625, "nullifier": 4725, "aghanims_scepter": 4200,
    "aghanims_shard": 1400, "refresher": 5200,
    "shivas_guard": 4750, "linkens_sphere": 4600, "lotus_orb": 3850,
    "aeon_disk": 3000, "hurricane_pike": 4475, "echo_sabre": 2500,
    "maelstrom": 2700, "mjollnir": 5600, "diffusal_blade": 2500,
    "sange_and_yasha": 4100, "kaya_and_sange": 4100,
    "yasha_and_kaya": 4100, "ethereal_blade": 4650,
    "radiance": 5150, "battle_fury": 4100, "hand_of_midas": 2200,
    "helm_of_the_dominator": 2400, "pipe": 3475, "crimson_guard": 3600,
    "solar_crest": 2625, "rod_of_atos": 2750, "gleipnir": 5650,
    "witch_blade": 2600, "spirit_vessel": 2920, "euls_scepter": 2725,
    "shadow_blade": 3000, "silver_edge": 5450, "heavens_halberd": 3550,
    "eternal_shroud": 3300, "blade_mail": 2100, "armlet": 2500,
    "mask_of_madness": 1775, "crystalys": 2000, "dragon_lance": 1900,
    "falcon_blade": 1125, "veil_of_discord": 1525, "pavise": 1100,
    "hood_of_defiance": 1500, "meteor_hammer": 2350,
    "urn_of_shadows": 880, "magic_wand": 450, "bracer": 505,
    "wraith_band": 505, "null_talisman": 505,
}


def extract_item_build_snapshots(
    match_id: int,
    player_data: dict,
    hero_id: int,
    position: int | None = None,
    enemy_hero_ids: list[int] | None = None,
) -> list[dict]:
    """
    Extract inventory snapshots at each major item completion.
    
    Uses the purchase_log from OpenDota parsed data to reconstruct
    inventory state over time.
    
    Args:
        match_id: The match ID
        player_data: Player dict from OpenDota (with purchase_log, items, etc.)
        hero_id: Hero ID
        position: Player position (1-5)
        enemy_hero_ids: Enemy team hero IDs
    
    Returns:
        List of snapshot dicts ready to insert into item_build_snapshots table.
    """
    purchase_log = player_data.get("purchase_log") or []
    if not purchase_log:
        return []

    snapshots = []
    current_inventory = []  # running inventory state
    snapshot_index = 0

    for purchase in purchase_log:
        item_key = purchase.get("key", "")
        game_time = purchase.get("time", 0)

        # Add to inventory (simplified — doesn't handle component combining)
        current_inventory.append(item_key)

        # Check if this is a major item
        cost = ITEM_COSTS.get(item_key, 0)
        if cost >= MAJOR_ITEM_COST_THRESHOLD:
            # Remove components that built into this item
            cleaned_inv = _clean_inventory_after_combine(
                current_inventory, item_key
            )

            snapshots.append({
                "match_id": match_id,
                "hero_id": hero_id,
                "account_id": player_data.get("account_id"),
                "position": position,
                "snapshot_index": snapshot_index,
                "game_time": game_time,
                "completed_item": item_key,
                "inventory": cleaned_inv[:6],  # main 6 slots
                "backpack": cleaned_inv[6:9] if len(cleaned_inv) > 6 else [],
                "net_worth": _estimate_net_worth_at_time(player_data, game_time),
                "enemy_hero_ids": enemy_hero_ids,
            })
            snapshot_index += 1

            # Update running inventory
            current_inventory = cleaned_inv

    return snapshots


def extract_ability_build(
    match_id: int,
    player_data: dict,
    hero_id: int,
    position: int | None = None,
    enemy_hero_ids: list[int] | None = None,
    ally_hero_ids: list[int] | None = None,
    patch: str | None = None,
    avg_rank: int | None = None,
) -> dict | None:
    """
    Extract ability upgrade order from parsed match data.
    
    Args:
        match_id: The match ID
        player_data: Player dict from OpenDota
        hero_id: Hero ID
        position: Player position
        enemy_hero_ids: Enemy heroes
        ally_hero_ids: Allied heroes
        patch: Game patch
        avg_rank: Average rank in match
    
    Returns:
        Dict ready to insert into ability_builds table, or None if no data.
    """
    ability_upgrades = player_data.get("ability_upgrades_arr") or []
    if not ability_upgrades:
        # Try the detailed format
        upgrades_detailed = player_data.get("ability_upgrades") or []
        if upgrades_detailed:
            ability_upgrades = [u.get("ability") for u in upgrades_detailed]

    if not ability_upgrades:
        return None

    # Convert ability IDs to names if needed
    # OpenDota sometimes gives IDs, sometimes names
    ability_order = []
    for ability in ability_upgrades:
        if isinstance(ability, int):
            # Would need ability ID → name mapping from constants
            ability_order.append(str(ability))
        else:
            ability_order.append(str(ability))

    # Extract talent choices
    talent_choices = {}
    talent_levels = {10, 15, 20, 25}
    for i, ability in enumerate(ability_order):
        level = i + 1
        if level in talent_levels:
            if "special_bonus" in str(ability):
                # Determine left/right
                talent_choices[level] = {
                    "choice": "left" if int(str(ability)[-1]) % 2 == 0 else "right",
                    "ability_id": ability,
                }

    return {
        "match_id": match_id,
        "hero_id": hero_id,
        "account_id": player_data.get("account_id"),
        "position": position,
        "ability_order": ability_order,
        "talent_choices": talent_choices,
        "enemy_hero_ids": enemy_hero_ids or [],
        "ally_hero_ids": ally_hero_ids or [],
        "patch": patch,
        "avg_rank": avg_rank,
    }


def extract_match_context(
    match_id: int,
    player_data: dict,
    hero_id: int,
    position: int,
    enemy_hero_ids: list[int],
    ally_hero_ids: list[int],
    patch: str | None = None,
    avg_rank: int | None = None,
) -> dict:
    """
    Create a match context vector for similarity search.
    
    Returns:
        Dict ready to insert into match_context_vectors table.
    """
    embedding = encode_match_context(hero_id, position, enemy_hero_ids, ally_hero_ids)

    return {
        "match_id": match_id,
        "account_id": player_data.get("account_id"),
        "hero_id": hero_id,
        "position": position,
        "enemy_hero_ids": sorted(enemy_hero_ids),
        "ally_hero_ids": sorted(ally_hero_ids),
        "embedding": embedding,
        "patch": patch,
        "avg_rank": avg_rank,
    }


def process_match_builds(db, match_id: int, od_data: dict) -> dict:
    """
    Full pipeline: extract and store item builds, ability builds, and context vectors
    for all players in a match.
    
    Args:
        db: Supabase client
        match_id: Match ID
        od_data: Full OpenDota match response
    
    Returns:
        Summary dict with counts of extracted data.
    """
    players = od_data.get("players", [])
    patch = str(od_data.get("patch", ""))
    avg_rank = od_data.get("avg_rank_tier")

    radiant_heroes = sorted([
        p["hero_id"] for p in players if p.get("player_slot", 0) < 128
    ])
    dire_heroes = sorted([
        p["hero_id"] for p in players if p.get("player_slot", 0) >= 128
    ])

    item_snapshots_total = 0
    ability_builds_total = 0
    contexts_total = 0

    for player in players:
        hero_id = player.get("hero_id")
        if not hero_id:
            continue

        is_radiant = player.get("player_slot", 0) < 128
        position = player.get("lane") or 0
        enemy_ids = dire_heroes if is_radiant else radiant_heroes
        ally_ids = [h for h in (radiant_heroes if is_radiant else dire_heroes) if h != hero_id]

        # 1. Item build snapshots
        try:
            snapshots = extract_item_build_snapshots(
                match_id, player, hero_id, position, enemy_ids
            )
            if snapshots:
                db.table("item_build_snapshots").upsert(
                    snapshots,
                    on_conflict="match_id,hero_id,snapshot_index"
                ).execute()
                item_snapshots_total += len(snapshots)
        except Exception as e:
            logger.warning(f"Failed to extract item snapshots for hero {hero_id}: {e}")

        # 2. Ability build
        try:
            ability_build = extract_ability_build(
                match_id, player, hero_id, position,
                enemy_ids, ally_ids, patch, avg_rank
            )
            if ability_build:
                db.table("ability_builds").upsert(
                    ability_build,
                    on_conflict="match_id,hero_id"
                ).execute()
                ability_builds_total += 1
        except Exception as e:
            logger.warning(f"Failed to extract ability build for hero {hero_id}: {e}")

        # 3. Match context vector
        try:
            context = extract_match_context(
                match_id, player, hero_id, position,
                enemy_ids, ally_ids, patch, avg_rank
            )
            db.table("match_context_vectors").upsert(
                context,
                on_conflict="match_id,hero_id"
            ).execute()
            contexts_total += 1
        except Exception as e:
            logger.warning(f"Failed to store context vector for hero {hero_id}: {e}")

    return {
        "item_snapshots": item_snapshots_total,
        "ability_builds": ability_builds_total,
        "context_vectors": contexts_total,
    }


# ── Private helpers ──────────────────────────────────

# Simplified component mapping for inventory cleaning
ITEM_COMPONENTS = {
    "power_treads": ["boots", "gloves", "belt_of_strength"],
    "phase_boots": ["boots", "chainmail", "blades_of_attack"],
    "arcane_boots": ["boots", "energy_booster"],
    "blink": [],  # no components
    "force_staff": ["staff_of_wizardry", "ring_of_regen", "recipe_force_staff"],
    "black_king_bar": ["ogre_axe", "mithril_hammer", "recipe_black_king_bar"],
    "desolator": ["mithril_hammer", "mithril_hammer", "blight_stone"],
    "manta": ["yasha", "ultimate_orb", "recipe_manta"],
    "battle_fury": ["broadsword", "claymore", "quelling_blade", "ring_of_health", "void_stone"],
    "aghanims_scepter": ["point_booster", "ogre_axe", "blade_of_alacrity", "staff_of_wizardry"],
    "echo_sabre": ["ogre_axe", "oblivion_staff"],
    "maelstrom": ["javelin", "mithril_hammer", "recipe_maelstrom"],
    # ... extend as needed
}


def _clean_inventory_after_combine(inventory: list[str], completed_item: str) -> list[str]:
    """Remove component items when a major item is completed."""
    components = ITEM_COMPONENTS.get(completed_item, [])
    cleaned = list(inventory)

    for component in components:
        if component in cleaned:
            cleaned.remove(component)

    # Remove consumables that are likely used up
    consumables = {"tango", "clarity", "enchanted_mango", "faerie_fire", "tango_single"}
    cleaned = [item for item in cleaned if item not in consumables or item == completed_item]

    return cleaned


def _estimate_net_worth_at_time(player_data: dict, game_time: int) -> int | None:
    """Estimate net worth at a given game time using gold_t array."""
    gold_t = player_data.get("gold_t") or []
    minute = max(0, game_time // 60)

    if minute < len(gold_t):
        return gold_t[minute]

    return None