"""
Benchmark Aggregation Pipeline

Computes percentile benchmarks (p25, median, p75, p90) for each metric,
stratified by hero × time_bucket × nw_bucket × duration_bucket × size_bucket.

This is pure math (no ML), but it's the foundation everything else builds on.
The Fight IQ model uses these percentiles as the scoring reference.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.ml import MIN_BENCHMARK_SAMPLES
from app.ml.feature_engineering import FIGHT_IQ_FEATURE_COLS

logger = logging.getLogger(__name__)

# Metrics we compute benchmarks for (subset of feature cols that are meaningful as benchmarks)
BENCHMARK_METRICS = [
    "ability_casts_per_sec",
    "total_ability_casts",
    "item_activations",
    "damage_per_sec",
    "damage_per_nw",
    "damage_total",
    "healing_per_sec",
    "kills",
    "deaths",
    "survived",
    "gold_delta",
    "xp_delta",
]


def compute_benchmarks(df: pd.DataFrame, patch: str = "current") -> list[dict]:
    """
    Compute percentile benchmarks from a training DataFrame.

    Args:
        df:    DataFrame from build_training_dataframe() — each row is one player-fight.
        patch: Current game patch string for versioning.

    Returns:
        List of benchmark dicts ready for upsert into hero_benchmarks table.
    """
    required_cols = ["hero_id", "time_bucket", "nw_bucket", "duration_bucket", "size_bucket"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    group_cols = ["hero_id", "time_bucket", "nw_bucket", "duration_bucket", "size_bucket"]

    benchmarks = []
    grouped = df.groupby(group_cols)

    for group_key, group_df in grouped:
        hero_id, t_bucket, n_bucket, d_bucket, s_bucket = group_key
        sample_count = len(group_df)

        for metric in BENCHMARK_METRICS:
            if metric not in group_df.columns:
                continue

            values = group_df[metric].dropna()
            if len(values) < MIN_BENCHMARK_SAMPLES:
                # Not enough data for reliable percentiles — still store with low sample_count
                # so the frontend can show a warning
                pass

            benchmarks.append({
                "hero_id": int(hero_id),
                "time_bucket": t_bucket,
                "nw_bucket": n_bucket,
                "duration_bucket": d_bucket,
                "size_bucket": s_bucket,
                "metric_name": metric,
                "p25": round(float(np.percentile(values, 25)), 4) if len(values) > 0 else None,
                "median": round(float(np.percentile(values, 50)), 4) if len(values) > 0 else None,
                "p75": round(float(np.percentile(values, 75)), 4) if len(values) > 0 else None,
                "p90": round(float(np.percentile(values, 90)), 4) if len(values) > 0 else None,
                "sample_count": int(sample_count),
                "patch": patch,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

    logger.info(
        f"Computed {len(benchmarks)} benchmark rows from {len(df)} fight samples "
        f"across {df['hero_id'].nunique()} heroes"
    )
    return benchmarks


def compute_player_percentile(
    value: float,
    benchmark: dict,
) -> float | None:
    """
    Given a player's metric value and the benchmark row, return their percentile (0-100).
    Uses linear interpolation between p25/median/p75/p90 knots.

    Returns None if benchmark data is insufficient.
    """
    p25 = benchmark.get("p25")
    median = benchmark.get("median")
    p75 = benchmark.get("p75")
    p90 = benchmark.get("p90")

    if any(v is None for v in [p25, median, p75, p90]):
        return None

    # Build interpolation knots: (percentile, value)
    knots = [(0, 0), (25, p25), (50, median), (75, p75), (90, p90), (100, p90 * 1.3)]

    # Handle edge cases
    if value <= knots[0][1]:
        return 0.0
    if value >= knots[-1][1]:
        return 100.0

    # Linear interpolation
    for i in range(len(knots) - 1):
        pct_lo, val_lo = knots[i]
        pct_hi, val_hi = knots[i + 1]
        if val_lo <= value <= val_hi:
            if val_hi == val_lo:
                return float(pct_lo)
            ratio = (value - val_lo) / (val_hi - val_lo)
            return round(pct_lo + ratio * (pct_hi - pct_lo), 1)

    return 50.0  # fallback


def compute_deltas(
    player_features: dict,
    benchmarks_for_context: dict[str, dict],
) -> list[dict]:
    """
    Compute the gap between a player's metrics and the benchmarks.

    Args:
        player_features:       feature dict from extract_player_fight_features()
        benchmarks_for_context: {metric_name: benchmark_row} for the player's context

    Returns:
        List of delta dicts, sorted by impact (largest gap first).
        Each: {metric, value, median, percentile, delta, priority}
    """
    deltas = []

    for metric in BENCHMARK_METRICS:
        bench = benchmarks_for_context.get(metric)
        if not bench:
            continue

        value = player_features.get(metric, 0)
        median = bench.get("median")
        if median is None:
            continue

        percentile = compute_player_percentile(value, bench)
        if percentile is None:
            continue

        # Delta: how far below/above median (negative = below)
        delta = value - median

        # Priority: distance from median percentile (50), weighted
        # Lower percentile = higher priority for improvement
        priority = abs(50 - percentile)

        deltas.append({
            "metric": metric,
            "value": round(value, 4),
            "median": round(median, 4),
            "p75": round(bench.get("p75", 0), 4),
            "p90": round(bench.get("p90", 0), 4),
            "percentile": percentile,
            "delta": round(delta, 4),
            "priority": round(priority, 1),
            "sample_count": bench.get("sample_count", 0),
        })

    # Sort by priority descending (biggest gaps first)
    deltas.sort(key=lambda d: d["priority"], reverse=True)
    return deltas


def generate_recommendations(deltas: list[dict], top_n: int = 5) -> list[dict]:
    """
    Generate actionable recommendations from benchmark deltas.
    Phase 2: template-based. Phase 4+: LLM-enhanced.

    Returns top_n recommendations sorted by impact.
    """
    TEMPLATES = {
        "ability_casts_per_sec": {
            "low": "You cast abilities {delta:.1f}/sec slower than median ({median:.1f}/sec). "
                   "At 7k+ MMR, players use abilities more aggressively in fights — "
                   "practice casting faster, especially during chaotic 5v5s.",
            "high": "Your ability cast rate ({value:.1f}/sec) is above the 7k+ median. Keep it up.",
        },
        "damage_per_sec": {
            "low": "Your damage output ({value:.0f}/sec) is below the {percentile:.0f}th percentile. "
                   "Median for this context is {median:.0f}/sec. "
                   "Focus on positioning to maintain uptime during the fight.",
            "high": "Damage output is strong at {value:.0f}/sec (p{percentile:.0f}). Solid execution.",
        },
        "damage_per_nw": {
            "low": "Low damage relative to your net worth — you're not converting gold into fight impact. "
                   "Consider whether your item build enables fight contribution.",
            "high": "Good gold-to-damage efficiency.",
        },
        "survived": {
            "low": "You died in this fight. At 7k+, {median:.0%} of players in your context survive. "
                   "Review your positioning and BKB timing.",
            "high": "Good survival.",
        },
        "bkb_used": {
            "low": "You didn't use BKB. High-MMR players activate it in this fight context — "
                   "not using BKB when you have it is one of the biggest execution gaps below Immortal.",
            "high": "BKB activated — good.",
        },
        "gold_delta": {
            "low": "You came out of this fight with less gold than expected ({value:+.0f} vs median {median:+.0f}). "
                   "This could mean dying too early or not securing kills.",
            "high": "Strong gold extraction from this fight.",
        },
        "item_activations": {
            "low": "Low active item usage ({value:.0f} vs median {median:.0f}). "
                   "High-MMR players use all their active items during fights.",
            "high": "Good active item usage.",
        },
    }

    recommendations = []
    for delta in deltas[:top_n]:
        metric = delta["metric"]
        template_set = TEMPLATES.get(metric)
        if not template_set:
            continue

        is_low = delta["percentile"] < 50
        template = template_set["low"] if is_low else template_set["high"]

        try:
            text = template.format(**delta)
        except (KeyError, ValueError):
            text = (
                f"{metric}: your value is {delta['value']}, "
                f"median is {delta['median']} (p{delta['percentile']:.0f})"
            )

        recommendations.append({
            "metric": metric,
            "text": text,
            "percentile": delta["percentile"],
            "priority": delta["priority"],
            "is_improvement_area": is_low,
        })

    return recommendations
