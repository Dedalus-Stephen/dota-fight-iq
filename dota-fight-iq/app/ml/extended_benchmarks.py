"""
Extended Benchmark Aggregation Pipeline

Builds on the existing fight benchmarks to cover ALL analysis dimensions:
    - Fight: ability usage, targeting, damage efficiency (existing, enhanced)
    - Laning: CS, gold, XP, kills at 5/10 min per hero per lane
    - Farming: GPM, idle time, farm window efficiency per hero per role
    - Itemization: item timing per hero, item choice rates vs enemy comp
    - Objectives: timing for towers, roshan, first blood
    - Support: wards/min, ward lifespan, stacking, participation per hero

All benchmarks are computed from 7k+ MMR data stored in Supabase.
Each benchmark is a percentile distribution (p25, median, p75, p90) stratified
by hero_id and relevant context (lane, role, game phase, etc.).

Usage:
    python -m scripts.compute_extended_benchmarks
    python -m scripts.compute_extended_benchmarks --laning-only
    python -m scripts.compute_extended_benchmarks --dry-run
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5  # Minimum samples for a benchmark bucket


# ═══════════════════════════════════════════════════════════
# Shared Helpers
# ═══════════════════════════════════════════════════════════

def _percentiles(values: pd.Series) -> dict:
    """Compute standard percentile set from a series."""
    vals = values.dropna()
    if len(vals) == 0:
        return {"p25": None, "median": None, "p75": None, "p90": None, "sample_count": 0}
    return {
        "p25": round(float(np.percentile(vals, 25)), 4),
        "median": round(float(np.percentile(vals, 50)), 4),
        "p75": round(float(np.percentile(vals, 75)), 4),
        "p90": round(float(np.percentile(vals, 90)), 4),
        "sample_count": int(len(vals)),
    }


def _compute_percentile(value: float, p25: float, median: float, p75: float, p90: float) -> float:
    """Interpolate a player's value into a percentile (0-100)."""
    if any(v is None for v in [p25, median, p75, p90]):
        return 50.0

    knots = [(0, 0), (25, p25), (50, median), (75, p75), (90, p90), (100, p90 * 1.3 if p90 > 0 else 1)]

    if value <= knots[0][1]:
        return 0.0
    if value >= knots[-1][1]:
        return 100.0

    for i in range(len(knots) - 1):
        pct_lo, val_lo = knots[i]
        pct_hi, val_hi = knots[i + 1]
        if val_lo <= value <= val_hi:
            if val_hi == val_lo:
                return float(pct_lo)
            ratio = (value - val_lo) / (val_hi - val_lo)
            return round(pct_lo + ratio * (pct_hi - pct_lo), 1)

    return 50.0


def percentile_from_benchmark(value: float, bench: dict) -> float:
    """Convenience wrapper for computing percentile from a benchmark dict."""
    return _compute_percentile(
        value,
        bench.get("p25", 0),
        bench.get("median", 0),
        bench.get("p75", 0),
        bench.get("p90", 0),
    )


# ═══════════════════════════════════════════════════════════
# Laning Benchmarks
# ═══════════════════════════════════════════════════════════

LANING_METRICS = [
    "lh_at_5min", "lh_at_10min", "dn_at_5min", "dn_at_10min",
    "gold_at_5min", "gold_at_10min", "xp_at_5min", "xp_at_10min",
    "cs_per_min_5", "cs_per_min_10",
    "kills_in_lane", "deaths_in_lane",
    "lane_gold_advantage", "lane_xp_advantage",
]


def compute_laning_benchmarks(df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute laning phase benchmarks from laning_analysis data.
    Stratified by hero_id × lane.

    Args:
        df: DataFrame from laning_analysis table (one row per player per match)

    Returns:
        List of benchmark dicts for hero_benchmarks table.
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    for (hero_id, lane), group in df.groupby(["hero_id", "lane"]):
        if pd.isna(hero_id) or pd.isna(lane):
            continue

        for metric in LANING_METRICS:
            if metric not in group.columns:
                continue

            pcts = _percentiles(group[metric])

            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": "laning",
                "nw_bucket": f"lane_{int(lane)}",
                "duration_bucket": "all",
                "size_bucket": "all",
                "metric_name": metric,
                **pcts,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} laning benchmark rows from {len(df)} samples")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Farming Benchmarks
# ═══════════════════════════════════════════════════════════

FARMING_METRICS = [
    "gpm", "xpm", "total_last_hits", "total_denies",
    "estimated_idle_minutes",
]

# Per-minute gold thresholds at key timestamps
FARMING_TIME_METRICS = [
    ("gold_at_15", 15), ("gold_at_20", 20), ("gold_at_25", 25),
    ("gold_at_30", 30), ("gold_at_35", 35),
]


def compute_farming_benchmarks(df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute farming efficiency benchmarks from farming_analysis data.
    Stratified by hero_id only (role-based stratification can be added when role detection improves).
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    # Add gold-at-time columns from gold_t arrays
    for col_name, minute in FARMING_TIME_METRICS:
        def _get_gold_at(gold_t, m=minute):
            if isinstance(gold_t, list) and m < len(gold_t):
                return gold_t[m]
            return None
        df[col_name] = df["gold_t"].apply(_get_gold_at)

    all_metrics = FARMING_METRICS + [m[0] for m in FARMING_TIME_METRICS]

    for hero_id, group in df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue

        for metric in all_metrics:
            if metric not in group.columns:
                continue

            pcts = _percentiles(group[metric])

            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": "farming",
                "nw_bucket": "all",
                "duration_bucket": "all",
                "size_bucket": "all",
                "metric_name": metric,
                **pcts,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} farming benchmark rows from {len(df)} samples")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Itemization Benchmarks
# ═══════════════════════════════════════════════════════════

def compute_item_timing_benchmarks(df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute item timing benchmarks from itemization_analysis data.
    Produces rows for item_timing_benchmarks table.

    For each hero × item combination: average/median/p25/p75 purchase time,
    and what percentage of games the item is bought.
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    for hero_id, group in df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue

        total_games = len(group)

        # Flatten all item_timings across matches for this hero
        all_timings = {}  # {item_key: [time_values]}
        for _, row in group.iterrows():
            timings = row.get("item_timings")
            if isinstance(timings, dict):
                for item_key, time_val in timings.items():
                    if time_val is not None and time_val > 0:
                        all_timings.setdefault(item_key, []).append(time_val)

        for item_key, times in all_timings.items():
            if len(times) < MIN_SAMPLES:
                continue

            arr = np.array(times)
            benchmarks.append({
                "hero_id": int(hero_id),
                "item_key": item_key,
                "role": None,  # Can be refined later with role detection
                "avg_time": round(float(np.mean(arr)), 1),
                "median_time": round(float(np.median(arr)), 1),
                "p25_time": round(float(np.percentile(arr, 25)), 1),
                "p75_time": round(float(np.percentile(arr, 75)), 1),
                "purchase_rate": round(len(times) / total_games, 4),
                "sample_count": len(times),
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} item timing benchmarks from {len(df)} samples")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Objective Benchmarks
# ═══════════════════════════════════════════════════════════

OBJECTIVE_TYPES_TO_BENCHMARK = [
    "CHAT_MESSAGE_FIRSTBLOOD",
    "CHAT_MESSAGE_ROSHAN_KILL",
]

BUILDING_SUBTYPES_TO_BENCHMARK = [
    "tower1_mid", "tower1_top", "tower1_bot",
    "tower2_mid", "tower2_top", "tower2_bot",
    "tower3_mid", "tower3_top", "tower3_bot",
]


def compute_objective_benchmarks(df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute objective timing benchmarks from match_objectives data.
    Aggregated across all matches (not per-hero).
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    def _obj_benchmark(obj_type: str, times: pd.Series) -> dict:
        """Build an objective benchmark dict with correct column names."""
        pcts = _percentiles(times)
        vals = times.dropna()
        return {
            "objective_type": obj_type,
            "avg_time": round(float(vals.mean()), 1) if len(vals) > 0 else None,
            "median_time": pcts["median"],
            "p25_time": pcts["p25"],
            "p75_time": pcts["p75"],
            "sample_count": pcts["sample_count"],
            "patch": patch,
            "context": None,
            "updated_at": now,
        }

    # Event-based objectives (first blood, roshan)
    for obj_type in OBJECTIVE_TYPES_TO_BENCHMARK:
        subset = df[df["type"] == obj_type]
        if len(subset) < MIN_SAMPLES:
            continue
        benchmarks.append(_obj_benchmark(obj_type, subset["time"]))

    # Building-based objectives (towers) — use subtype
    for subtype in BUILDING_SUBTYPES_TO_BENCHMARK:
        subset = df[df["subtype"] == subtype]
        if len(subset) < MIN_SAMPLES:
            continue
        benchmarks.append(_obj_benchmark(f"building_{subtype}", subset["time"]))

    logger.info(f"Computed {len(benchmarks)} objective benchmarks from {len(df)} samples")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Support Benchmarks
# ═══════════════════════════════════════════════════════════

SUPPORT_METRICS = [
    "obs_placed", "sen_placed", "camps_stacked",
    "teamfight_participation", "stuns",
    "hero_healing",
]


def compute_support_benchmarks(
    players_df: pd.DataFrame,
    wards_df: pd.DataFrame | None = None,
    patch: str = "current",
) -> list[dict]:
    """
    Compute support efficiency benchmarks from match_players data.
    Filters to support players (lowest 2 NW on each team) and stratifies by hero.

    Also computes ward-specific benchmarks if ward data is provided.
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    # Identify supports: for each match, take the 2 lowest-NW players per side
    support_slots = set()
    for match_id, match_group in players_df.groupby("match_id"):
        for is_rad in [True, False]:
            side = match_group[match_group["is_radiant"] == is_rad].sort_values("net_worth")
            if len(side) >= 2:
                support_slots.update(side.head(2).index.tolist())

    supports = players_df.loc[players_df.index.isin(support_slots)]

    if len(supports) == 0:
        logger.warning("No support players identified")
        return benchmarks

    # Compute duration-normalized metrics
    # We need match duration to compute per-minute rates
    # For now, use the raw values (the match_players table has these as totals)

    for hero_id, group in supports.groupby("hero_id"):
        if pd.isna(hero_id):
            continue

        for metric in SUPPORT_METRICS:
            if metric not in group.columns:
                continue

            pcts = _percentiles(group[metric])

            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": "support",
                "nw_bucket": "all",
                "duration_bucket": "all",
                "size_bucket": "all",
                "metric_name": metric,
                **pcts,
                "patch": patch,
                "updated_at": now,
            })

    # Ward benchmarks: avg lifespan, wards per match, deward rate
    if wards_df is not None and len(wards_df) > 0:
        obs_wards = wards_df[wards_df["ward_type"] == "observer"]

        for hero_id, group in obs_wards.groupby("hero_id"):
            if pd.isna(hero_id):
                continue

            # Observer lifespan
            lifespans = group["duration_alive"].dropna()
            if len(lifespans) >= MIN_SAMPLES:
                pcts = _percentiles(lifespans)
                benchmarks.append({
                    "hero_id": int(hero_id),
                    "time_bucket": "support",
                    "nw_bucket": "all",
                    "duration_bucket": "all",
                    "size_bucket": "all",
                    "metric_name": "obs_avg_lifespan",
                    **pcts,
                    "patch": patch,
                    "updated_at": now,
                })

            # Deward rate per hero
            if len(group) >= MIN_SAMPLES:
                deward_rate = group["is_dewarded"].mean()
                # Store as a single-value benchmark (median = the rate)
                benchmarks.append({
                    "hero_id": int(hero_id),
                    "time_bucket": "support",
                    "nw_bucket": "all",
                    "duration_bucket": "all",
                    "size_bucket": "all",
                    "metric_name": "obs_deward_rate",
                    "p25": round(float(deward_rate), 4),
                    "median": round(float(deward_rate), 4),
                    "p75": round(float(deward_rate), 4),
                    "p90": round(float(deward_rate), 4),
                    "sample_count": len(group),
                    "patch": patch,
                    "updated_at": now,
                })

    logger.info(f"Computed {len(benchmarks)} support benchmarks from {len(supports)} support player records")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Fight Targeting Benchmarks (Enhanced)
# ═══════════════════════════════════════════════════════════

def compute_fight_targeting_benchmarks(fight_stats_df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute fight-level benchmarks for targeting efficiency.
    Metrics: damage efficiency per target, ult target priority, etc.

    This uses the ability_targets and damage_targets JSONB fields.
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    for hero_id, group in fight_stats_df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue

        # Compute per-fight metrics from JSONB fields
        dmg_efficiencies = []
        target_counts = []

        for _, row in group.iterrows():
            damage_targets = row.get("damage_targets")
            ability_targets = row.get("ability_targets")
            total_damage = row.get("damage", 0) or 0

            if isinstance(damage_targets, dict) and damage_targets:
                # Count unique targets
                all_targets = set()
                for ability, targets in damage_targets.items():
                    if isinstance(targets, dict):
                        all_targets.update(targets.keys())
                target_counts.append(len(all_targets))

                # Damage concentration: what % of damage went to the primary target
                all_dmg_by_target = {}
                for ability, targets in damage_targets.items():
                    if isinstance(targets, dict):
                        for target, dmg in targets.items():
                            all_dmg_by_target[target] = all_dmg_by_target.get(target, 0) + dmg

                if all_dmg_by_target and total_damage > 0:
                    max_target_dmg = max(all_dmg_by_target.values())
                    concentration = max_target_dmg / total_damage
                    dmg_efficiencies.append(concentration)

        if len(dmg_efficiencies) >= MIN_SAMPLES:
            pcts = _percentiles(pd.Series(dmg_efficiencies))
            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": "fight",
                "nw_bucket": "all",
                "duration_bucket": "all",
                "size_bucket": "all",
                "metric_name": "damage_concentration",
                **pcts,
                "patch": patch,
                "updated_at": now,
            })

        if len(target_counts) >= MIN_SAMPLES:
            pcts = _percentiles(pd.Series(target_counts))
            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": "fight",
                "nw_bucket": "all",
                "duration_bucket": "all",
                "size_bucket": "all",
                "metric_name": "unique_targets_per_fight",
                **pcts,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} fight targeting benchmarks")
    return benchmarks


# ═══════════════════════════════════════════════════════════
# Recommendation Generator (Enhanced)
# ═══════════════════════════════════════════════════════════

# Templates keyed by (feature_category, metric_name)
RECOMMENDATION_TEMPLATES = {
    # Laning
    ("laning", "lh_at_10min"): {
        "low": "Your last hits at 10 minutes ({value}) are below the 7k+ median ({median}). Focus on not missing creeps under tower and maintaining lane equilibrium.",
        "high": "Strong laning CS ({value} LH at 10 min, p{percentile}). You're out-farming most 7k+ players on {hero} in this lane.",
    },
    ("laning", "cs_per_min_10"): {
        "low": "Your CS rate ({value}/min) falls behind 7k+ {hero} players ({median}/min). Practice last-hitting patterns and creep aggro management.",
        "high": "Excellent CS efficiency at {value}/min (p{percentile}). Your last-hitting fundamentals are solid.",
    },
    ("laning", "deaths_in_lane"): {
        "low": "You died {value} time(s) in lane. 7k+ {hero} players average {median} deaths. Review your positioning during enemy power spikes.",
        "high": "Clean laning phase with only {value} death(s) — strong survival awareness.",
    },
    # Farming
    ("farming", "gpm"): {
        "low": "Your GPM ({value}) is below the 7k+ median for {hero} ({median}). Look for farming patterns between fights — don't stand around waiting.",
        "high": "Strong farm at {value} GPM (p{percentile}). You're converting map space into gold efficiently.",
    },
    ("farming", "estimated_idle_minutes"): {
        "low": "You had {value} idle minutes — time with minimal gold income. 7k+ players average {median}. Always be farming, stacking, or rotating with purpose.",
        "high": "Low idle time ({value} min) — you're staying active on the map consistently.",
    },
    # Items
    ("items", "timing"): {
        "low": "Your {item} came at {value_fmt}. 7k+ {hero} players get it by {median_fmt} on average. Faster farming or fewer unnecessary purchases could help.",
        "high": "Fast {item} timing at {value_fmt} (7k+ median: {median_fmt}). Your item progression is on track.",
    },
    # Support
    ("support", "obs_placed"): {
        "low": "You placed {value} observers this game. 7k+ {hero} supports average {median}. More frequent warding gives your team critical information.",
        "high": "Good ward count ({value} observers, p{percentile}). Your vision game is solid.",
    },
    ("support", "teamfight_participation"): {
        "low": "Your teamfight participation ({value_pct}%) is below 7k+ median ({median_pct}%). Make sure you're positioned to join fights, not farming alone.",
        "high": "High teamfight presence at {value_pct}% (p{percentile}). You're showing up where it matters.",
    },
    ("support", "camps_stacked"): {
        "low": "Only {value} camps stacked. 7k+ supports on {hero} average {median}. Stacking accelerates your cores significantly — make it a habit.",
        "high": "Great stacking with {value} camps (p{percentile}). Your cores thank you.",
    },
    # Fight targeting
    ("fight", "damage_concentration"): {
        "low": "Your damage is spread across too many targets ({value_pct}% on primary). 7k+ players focus {median_pct}% on their priority target. Pick one and commit.",
        "high": "Good target focus — {value_pct}% damage on primary target (p{percentile}).",
    },
    # Objectives
    ("objectives", "post_fight_conversion"): {
        "low": "You only converted {value_pct}% of won fights into objectives. 7k+ teams convert {median_pct}%. After winning a fight, immediately push a tower or take Roshan.",
        "high": "Strong objective conversion at {value_pct}% (p{percentile}). You're translating fight wins into map control.",
    },
}


def generate_extended_recommendations(
    category: str,
    player_metrics: dict,
    benchmarks: dict,  # {metric_name: benchmark_dict}
    hero_name: str = "this hero",
) -> list[dict]:
    """
    Generate actionable recommendations by comparing player metrics to benchmarks.

    Args:
        category:       "laning", "farming", "items", "support", "fight", "objectives"
        player_metrics: {metric_name: value}
        benchmarks:     {metric_name: {p25, median, p75, p90, sample_count}}
        hero_name:      display name for templates

    Returns:
        List of recommendation dicts sorted by priority.
    """
    recs = []

    for metric_name, value in player_metrics.items():
        bench = benchmarks.get(metric_name)
        if not bench or bench.get("median") is None:
            continue

        pct = percentile_from_benchmark(value, bench)
        median = bench["median"]
        is_low = pct < 40
        is_high = pct >= 70

        # Determine if "low" is bad (most metrics) or good (deaths, idle time)
        inverted_metrics = {"deaths_in_lane", "estimated_idle_minutes", "obs_deward_rate"}
        if metric_name in inverted_metrics:
            is_low, is_high = is_high, is_low

        template_key = (category, metric_name)
        templates = RECOMMENDATION_TEMPLATES.get(template_key)

        if templates:
            variant = "high" if is_high else "low" if is_low else None
            if variant and variant in templates:
                text = templates[variant].format(
                    value=round(value, 1) if isinstance(value, float) else value,
                    median=round(median, 1) if isinstance(median, float) else median,
                    percentile=round(pct),
                    hero=hero_name,
                    value_pct=round(value * 100) if value <= 1 else round(value, 1),
                    median_pct=round(median * 100) if median <= 1 else round(median, 1),
                    value_fmt=_format_time(value) if value > 60 else round(value),
                    median_fmt=_format_time(median) if median > 60 else round(median),
                    item=metric_name,
                )

                recs.append({
                    "metric": metric_name,
                    "category": category,
                    "value": round(value, 4) if isinstance(value, float) else value,
                    "median": round(median, 4) if isinstance(median, float) else median,
                    "percentile": round(pct, 1),
                    "is_strength": is_high and metric_name not in inverted_metrics,
                    "is_improvement_area": is_low and metric_name not in inverted_metrics,
                    "text": text,
                    "priority": abs(50 - pct),
                    "sample_count": bench.get("sample_count", 0),
                })
        else:
            # Generic recommendation
            if is_low or is_high:
                direction = "above" if is_high else "below"
                recs.append({
                    "metric": metric_name,
                    "category": category,
                    "value": round(value, 4) if isinstance(value, float) else value,
                    "median": round(median, 4) if isinstance(median, float) else median,
                    "percentile": round(pct, 1),
                    "is_strength": is_high and metric_name not in inverted_metrics,
                    "is_improvement_area": is_low and metric_name not in inverted_metrics,
                    "text": f"Your {metric_name.replace('_', ' ')} ({round(value, 1)}) is {direction} the 7k+ median ({round(median, 1)}) at p{round(pct)}.",
                    "priority": abs(50 - pct),
                    "sample_count": bench.get("sample_count", 0),
                })

    recs.sort(key=lambda r: r["priority"], reverse=True)
    return recs


def _format_time(seconds):
    """Format seconds as M:SS."""
    if seconds is None:
        return "—"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ═══════════════════════════════════════════════════════════
# Score Computation
# ═══════════════════════════════════════════════════════════

def compute_dimension_score(player_metrics: dict, benchmarks: dict, inverted: set = None) -> float:
    """
    Compute a 0-100 score for a dimension by averaging percentile ranks.

    Args:
        player_metrics: {metric_name: value}
        benchmarks:     {metric_name: benchmark_dict}
        inverted:       set of metric names where lower is better (e.g. deaths)

    Returns:
        Weighted average percentile (0-100).
    """
    inverted = inverted or set()
    percentiles = []

    for metric, value in player_metrics.items():
        bench = benchmarks.get(metric)
        if not bench or bench.get("median") is None:
            continue

        pct = percentile_from_benchmark(value, bench)

        # For inverted metrics, flip the percentile
        if metric in inverted:
            pct = 100 - pct

        percentiles.append(pct)

    if not percentiles:
        return 50.0  # neutral if no data

    return round(sum(percentiles) / len(percentiles), 1)


# ═══════════════════════════════════════════════════════════
# Composite Match Score (Feature 9)
# ═══════════════════════════════════════════════════════════

# Role-specific weights
ROLE_WEIGHTS = {
    "pos1": {"fight": 0.25, "laning": 0.20, "farming": 0.25, "items": 0.10, "objectives": 0.10, "support": 0.00, "deaths": 0.10},
    "pos2": {"fight": 0.30, "laning": 0.20, "farming": 0.15, "items": 0.10, "objectives": 0.10, "support": 0.05, "deaths": 0.10},
    "pos3": {"fight": 0.25, "laning": 0.15, "farming": 0.15, "items": 0.10, "objectives": 0.15, "support": 0.10, "deaths": 0.10},
    "pos4": {"fight": 0.20, "laning": 0.10, "farming": 0.05, "items": 0.10, "objectives": 0.10, "support": 0.25, "deaths": 0.20},
    "pos5": {"fight": 0.20, "laning": 0.10, "farming": 0.05, "items": 0.10, "objectives": 0.10, "support": 0.30, "deaths": 0.15},
}


def compute_overall_match_score(
    sub_scores: dict,
    role: str = "pos1",
) -> tuple[float, dict]:
    """
    Compute weighted overall score from dimension sub-scores.

    Args:
        sub_scores: {"fight": 75.0, "laning": 60.0, "farming": 82.0, ...}
        role:       "pos1" through "pos5"

    Returns:
        (overall_score, weights_used)
    """
    weights = ROLE_WEIGHTS.get(role, ROLE_WEIGHTS["pos1"])

    weighted_sum = 0.0
    total_weight = 0.0

    for dimension, weight in weights.items():
        if dimension in sub_scores and sub_scores[dimension] is not None:
            weighted_sum += sub_scores[dimension] * weight
            total_weight += weight

    if total_weight == 0:
        return 50.0, weights

    overall = round(weighted_sum / total_weight, 1)
    return overall, weights