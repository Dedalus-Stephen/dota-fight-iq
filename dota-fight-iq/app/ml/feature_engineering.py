"""
Feature Engineering

Transforms raw fight_player_stats rows into ML-ready feature vectors.
Two feature sets:
    1. Per-player fight features  → used for Fight IQ scoring
    2. Per-fight team features    → used for fight outcome prediction
    3. Per-fight structural features → used for DBSCAN clustering
"""

import logging
import numpy as np
import pandas as pd

from app.ml import TIME_BUCKETS, NW_BUCKETS, DURATION_BUCKETS, SIZE_BUCKETS

logger = logging.getLogger(__name__)


# ── Context Bucketing ─────────────────────────────────────

def time_bucket(game_time_sec: int) -> str:
    """Assign game time (seconds) to a time bucket."""
    minutes = game_time_sec / 60
    if minutes < 15:
        return "0-15"
    elif minutes < 25:
        return "15-25"
    elif minutes < 35:
        return "25-35"
    elif minutes < 45:
        return "35-45"
    else:
        return "45+"


def nw_bucket(player_nw: int, avg_nw: float) -> str:
    """Assign player net worth relative to game average into a bucket."""
    if avg_nw <= 0:
        return "average"
    ratio = player_nw / avg_nw
    if ratio < 0.75:
        return "below_avg"
    elif ratio < 1.15:
        return "average"
    elif ratio < 1.5:
        return "above_avg"
    else:
        return "far_ahead"


def duration_bucket(fight_duration_sec: int) -> str:
    """Assign fight duration to a bucket."""
    if fight_duration_sec < 10:
        return "short"
    elif fight_duration_sec <= 20:
        return "medium"
    else:
        return "long"


def size_bucket(total_deaths: int) -> str:
    """Assign fight size based on total deaths."""
    if total_deaths <= 4:
        return "skirmish"
    elif total_deaths <= 7:
        return "teamfight"
    else:
        return "bloodbath"


# ── Per-Player Fight Features ─────────────────────────────
# These are the ~20 features used by the Fight IQ model.

def extract_player_fight_features(
    fight_stat: dict,
    teamfight: dict,
    match: dict,
    all_players: list[dict],
) -> dict:
    """
    Extract ML features for a single player's performance in a single fight.

    Args:
        fight_stat:  row from fight_player_stats table
        teamfight:   row from teamfights table (the fight this stat belongs to)
        match:       row from matches table
        all_players: all match_players rows for this match

    Returns:
        Dict of feature_name → float, plus context bucket columns.
    """
    fight_dur = max(teamfight.get("duration", 1) or 1, 1)  # avoid div by zero
    hero_id = fight_stat.get("hero_id", 0)

    # Ability usage
    ability_uses = fight_stat.get("ability_uses") or {}
    total_ability_casts = sum(ability_uses.values()) if isinstance(ability_uses, dict) else 0
    ability_casts_per_sec = total_ability_casts / fight_dur

    # Item usage
    item_uses = fight_stat.get("item_uses") or {}
    total_item_activations = sum(item_uses.values()) if isinstance(item_uses, dict) else 0

    # Key item flags
    bkb_used = 1 if item_uses.get("black_king_bar", 0) > 0 else 0
    blink_used = 1 if item_uses.get("blink", 0) > 0 else 0

    # Damage
    damage = fight_stat.get("damage", 0) or 0
    damage_per_sec = damage / fight_dur

    # Find player's net worth from match_players
    player_nw = 0
    for p in all_players:
        if p.get("hero_id") == hero_id:
            player_nw = p.get("net_worth", 0) or 0
            break
    damage_per_nw = damage / max(player_nw, 1) * 1000  # damage per 1000 gold invested

    # Healing
    healing = fight_stat.get("healing", 0) or 0
    healing_per_sec = healing / fight_dur

    # Deaths & kills
    deaths = fight_stat.get("deaths", 0) or 0
    killed = fight_stat.get("killed") or {}
    kills = sum(killed.values()) if isinstance(killed, dict) else 0
    survived = 1 if deaths == 0 else 0

    # Buybacks
    buybacks = fight_stat.get("buybacks", 0) or 0

    # Gold/XP extraction
    gold_delta = fight_stat.get("gold_delta", 0) or 0
    xp_delta = fight_stat.get("xp_delta", 0) or 0

    # Fight context
    fight_start = teamfight.get("start_time", 0) or 0
    deaths_count = teamfight.get("deaths_count", 0) or 0

    # Average NW across all players in the match
    all_nw = [p.get("net_worth", 0) or 0 for p in all_players]
    avg_nw = np.mean(all_nw) if all_nw else 1

    # Context buckets
    t_bucket = time_bucket(fight_start)
    n_bucket = nw_bucket(player_nw, avg_nw)
    d_bucket = duration_bucket(fight_dur)
    s_bucket = size_bucket(deaths_count)

    return {
        # ── Core features (used by XGBoost) ──
        "ability_casts_per_sec": round(ability_casts_per_sec, 4),
        "total_ability_casts": total_ability_casts,
        "item_activations": total_item_activations,
        "bkb_used": bkb_used,
        "blink_used": blink_used,
        "damage_per_sec": round(damage_per_sec, 2),
        "damage_per_nw": round(damage_per_nw, 4),
        "damage_total": damage,
        "healing_per_sec": round(healing_per_sec, 2),
        "healing_total": healing,
        "kills": kills,
        "deaths": deaths,
        "survived": survived,
        "buybacks": buybacks,
        "gold_delta": gold_delta,
        "xp_delta": xp_delta,
        "fight_duration": fight_dur,
        "fight_size": deaths_count,
        "game_time": fight_start,
        "player_net_worth": player_nw,
        # ── Context buckets (for stratified benchmarks) ──
        "hero_id": hero_id,
        "time_bucket": t_bucket,
        "nw_bucket": n_bucket,
        "duration_bucket": d_bucket,
        "size_bucket": s_bucket,
    }


# Feature columns used by the Fight IQ XGBoost model (no context/ID columns)
FIGHT_IQ_FEATURE_COLS = [
    "ability_casts_per_sec",
    "total_ability_casts",
    "item_activations",
    "bkb_used",
    "blink_used",
    "damage_per_sec",
    "damage_per_nw",
    "damage_total",
    "healing_per_sec",
    "healing_total",
    "kills",
    "deaths",
    "survived",
    "buybacks",
    "gold_delta",
    "xp_delta",
    "fight_duration",
    "fight_size",
    "game_time",
    "player_net_worth",
]


# ── Per-Fight Team Features (for outcome prediction) ─────

def extract_fight_outcome_features(
    teamfight: dict,
    fight_stats: list[dict],
    match: dict,
    all_players: list[dict],
) -> dict | None:
    """
    Extract team-level features for fight outcome prediction.
    Returns features + label (radiant_won_fight).

    Args:
        teamfight:   row from teamfights table
        fight_stats: all fight_player_stats for this fight (10 rows)
        match:       matches row
        all_players: match_players rows
    """
    if len(fight_stats) != 10:
        return None  # skip incomplete fights

    radiant_stats = [s for s in fight_stats if s.get("is_radiant")]
    dire_stats = [s for s in fight_stats if not s.get("is_radiant")]

    if not radiant_stats or not dire_stats:
        return None

    radiant_players = [p for p in all_players if p.get("is_radiant")]
    dire_players = [p for p in all_players if not p.get("is_radiant")]

    def team_nw(players):
        return sum(p.get("net_worth", 0) or 0 for p in players)

    def team_damage(stats):
        return sum(s.get("damage", 0) or 0 for s in stats)

    def team_deaths(stats):
        return sum(s.get("deaths", 0) or 0 for s in stats)

    def team_kills(stats):
        total = 0
        for s in stats:
            killed = s.get("killed") or {}
            total += sum(killed.values()) if isinstance(killed, dict) else 0
        return total

    r_nw = team_nw(radiant_players)
    d_nw = team_nw(dire_players)
    nw_advantage = r_nw - d_nw  # positive = radiant ahead

    # Determine who won the fight by net gold swing
    gold_swing = teamfight.get("gold_swing", 0) or 0
    radiant_kills = teamfight.get("radiant_kills", 0) or 0
    dire_kills = teamfight.get("dire_kills", 0) or 0

    # Label: radiant won the fight if they got more kills
    # Tie-break by gold swing
    if radiant_kills > dire_kills:
        radiant_won = 1
    elif dire_kills > radiant_kills:
        radiant_won = 0
    else:
        radiant_won = 1 if gold_swing > 0 else 0

    fight_start = teamfight.get("start_time", 0) or 0

    return {
        "nw_advantage_radiant": nw_advantage,
        "radiant_total_nw": r_nw,
        "dire_total_nw": d_nw,
        "fight_time": fight_start,
        "fight_duration": max(teamfight.get("duration", 1) or 1, 1),
        "total_deaths": teamfight.get("deaths_count", 0) or 0,
        "radiant_hero_count": len(radiant_stats),
        "dire_hero_count": len(dire_stats),
        # Label
        "radiant_won_fight": radiant_won,
    }


FIGHT_OUTCOME_FEATURE_COLS = [
    "nw_advantage_radiant",
    "radiant_total_nw",
    "dire_total_nw",
    "fight_time",
    "fight_duration",
    "total_deaths",
    "radiant_hero_count",
    "dire_hero_count",
]


# ── Fight Structural Features (for DBSCAN clustering) ────

def extract_clustering_features(
    teamfight: dict,
    fight_stats: list[dict],
) -> dict:
    """
    Extract structural features that define the "type" of fight.
    Used by DBSCAN to discover fight archetypes.
    """
    deaths_count = teamfight.get("deaths_count", 0) or 0
    duration = max(teamfight.get("duration", 1) or 1, 1)
    fight_start = teamfight.get("start_time", 0) or 0

    # Hero participation (how many players were actually involved)
    active_players = 0
    for s in fight_stats:
        damage = s.get("damage", 0) or 0
        healing = s.get("healing", 0) or 0
        ability_uses = s.get("ability_uses") or {}
        casts = sum(ability_uses.values()) if isinstance(ability_uses, dict) else 0
        if damage > 0 or healing > 0 or casts > 0:
            active_players += 1

    # Damage concentration — did one player do most damage?
    damages = [s.get("damage", 0) or 0 for s in fight_stats]
    total_dmg = sum(damages) or 1
    max_dmg_share = max(damages) / total_dmg

    # Kill distribution
    radiant_kills = teamfight.get("radiant_kills", 0) or 0
    dire_kills = teamfight.get("dire_kills", 0) or 0
    kill_imbalance = abs(radiant_kills - dire_kills) / max(deaths_count, 1)

    # Gold swing magnitude
    gold_swing = abs(teamfight.get("gold_swing", 0) or 0)

    return {
        "fight_time_minutes": fight_start / 60,
        "duration_sec": duration,
        "deaths_count": deaths_count,
        "active_players": active_players,
        "max_damage_share": round(max_dmg_share, 4),
        "kill_imbalance": round(kill_imbalance, 4),
        "gold_swing_abs": gold_swing,
        # IDs for labeling after clustering
        "teamfight_id": teamfight.get("id"),
        "match_id": teamfight.get("match_id"),
    }


CLUSTERING_FEATURE_COLS = [
    "fight_time_minutes",
    "duration_sec",
    "deaths_count",
    "active_players",
    "max_damage_share",
    "kill_imbalance",
    "gold_swing_abs",
]


# ── Similarity Vector (for pgvector search) ──────────────

def build_similarity_vector(player_features: dict) -> list[float]:
    """
    Build a 32-dim vector for pgvector similarity search.
    Encodes the most discriminating features of a player's fight performance.
    Normalized to [0, 1] range for cosine similarity.
    """
    # We use a subset of features that best characterize a fight for "find similar"
    raw = [
        player_features.get("ability_casts_per_sec", 0),
        player_features.get("damage_per_sec", 0),
        player_features.get("damage_per_nw", 0),
        player_features.get("healing_per_sec", 0),
        player_features.get("kills", 0),
        player_features.get("deaths", 0),
        player_features.get("gold_delta", 0),
        player_features.get("xp_delta", 0),
        player_features.get("item_activations", 0),
        player_features.get("bkb_used", 0),
        player_features.get("blink_used", 0),
        player_features.get("survived", 0),
        player_features.get("fight_duration", 0),
        player_features.get("fight_size", 0),
        player_features.get("game_time", 0) / 3600,  # normalize to ~0-1
        player_features.get("player_net_worth", 0) / 30000,  # normalize
    ]
    # Pad to 32 dims (reserved for future features like hero category, position)
    raw.extend([0.0] * (32 - len(raw)))
    return [float(x) for x in raw[:32]]


# ── Batch Feature Extraction ─────────────────────────────

def build_training_dataframe(
    fights: list[dict],
    fight_stats: list[dict],
    matches: list[dict],
    match_players: list[dict],
) -> pd.DataFrame:
    """
    Build a full training DataFrame from DB data.
    Each row = one player's performance in one fight.

    Args:
        fights:        list of teamfights rows
        fight_stats:   list of fight_player_stats rows
        matches:       list of matches rows
        match_players: list of match_players rows

    Returns:
        DataFrame with all features + context columns.
    """
    # Index data for fast lookup
    match_map = {m["match_id"]: m for m in matches}
    players_by_match = {}
    for p in match_players:
        mid = p["match_id"]
        if mid not in players_by_match:
            players_by_match[mid] = []
        players_by_match[mid].append(p)

    fight_map = {f["id"]: f for f in fights}

    # Group fight_stats by teamfight_id
    stats_by_fight = {}
    for s in fight_stats:
        tid = s["teamfight_id"]
        if tid not in stats_by_fight:
            stats_by_fight[tid] = []
        stats_by_fight[tid].append(s)

    rows = []
    for fight in fights:
        fight_id = fight["id"]
        match_id = fight["match_id"]
        match = match_map.get(match_id)
        if not match:
            continue

        players = players_by_match.get(match_id, [])
        stats = stats_by_fight.get(fight_id, [])

        for stat in stats:
            try:
                features = extract_player_fight_features(stat, fight, match, players)
                features["teamfight_id"] = fight_id
                features["match_id"] = match_id
                features["account_id"] = stat.get("account_id")
                rows.append(features)
            except Exception as e:
                logger.warning(f"Feature extraction failed for fight {fight_id}: {e}")

    df = pd.DataFrame(rows)
    logger.info(f"Built training DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df
