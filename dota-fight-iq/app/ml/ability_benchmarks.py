"""
Ability & Item Usage Benchmark Pipeline

Computes per-hero ability/item usage patterns from 7k+ MMR fight data.
Powers the "Action Comparison" feature.
"""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.ml.feature_engineering import time_bucket, size_bucket

logger = logging.getLogger(__name__)

MIN_SAMPLES = 5

IGNORED_ITEMS = {
    "tango", "tango_single", "flask", "clarity", "enchanted_mango",
    "famango", "faerie_fire", "ward_observer", "ward_sentry",
    "tpscroll", "smoke_of_deceit", "dust", "tome_of_knowledge",
    "quelling_blade",
}

KEY_ITEMS = {
    "black_king_bar", "blink", "refresher", "satanic", "nullifier",
    "orchid", "bloodthorn", "sheepstick", "ethereal_blade", "silver_edge",
    "mask_of_madness", "manta", "hurricane_pike", "shivas_guard",
    "crimson_guard", "pipe", "mekansm", "guardian_greaves",
    "blade_mail", "lotus_orb", "aeon_disk", "linken_sphere",
    "ghost", "glimmer_cape", "force_staff",
}

IGNORED_ABILITIES = {
    "plus_high_five", "plus_guild_banner",
}


def compute_ability_usage_benchmarks(
    merged_df: pd.DataFrame,
    patch: str = "current",
) -> list[dict]:
    """
    Compute per-hero, per-ability usage benchmarks.

    Args:
        merged_df: DataFrame with columns from fight_player_stats JOIN teamfights:
                   hero_id, ability_uses, start_time, duration, deaths_count
    """
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    merged_df["_tb"] = merged_df["start_time"].apply(lambda t: time_bucket(t or 0))
    merged_df["_sb"] = merged_df["deaths_count"].apply(lambda d: size_bucket(d or 0))

    for hero_id, hero_group in merged_df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue
        hero_id = int(hero_id)

        # Compute for "all" context only (most data per bucket)
        total_fights = len(hero_group)
        if total_fights < MIN_SAMPLES:
            continue

        ability_stats = {}

        for _, row in hero_group.iterrows():
            ability_uses = row.get("ability_uses")
            fight_dur = max(row.get("duration", 1) or 1, 1)

            if not isinstance(ability_uses, dict):
                continue

            for ability, count in ability_uses.items():
                if ability in IGNORED_ABILITIES:
                    continue
                if ability not in ability_stats:
                    ability_stats[ability] = {"casts": [], "casts_per_sec": []}
                ability_stats[ability]["casts"].append(count)
                ability_stats[ability]["casts_per_sec"].append(count / fight_dur)

        for ability_key, stats in ability_stats.items():
            fights_with = len(stats["casts"])
            if fights_with < MIN_SAMPLES:
                continue

            usage_rate = fights_with / total_fights
            casts_series = pd.Series(stats["casts"])

            benchmarks.append({
                "hero_id": hero_id,
                "ability_key": ability_key,
                "time_bucket": "all",
                "size_bucket": "all",
                "usage_rate": round(usage_rate, 4),
                "avg_casts": round(float(casts_series.mean()), 2),
                "median_casts": round(float(casts_series.median()), 2),
                "p75_casts": round(float(np.percentile(casts_series, 75)), 2),
                "p90_casts": round(float(np.percentile(casts_series, 90)), 2),
                "avg_casts_per_sec": round(float(np.mean(stats["casts_per_sec"])), 4),
                "sample_count": total_fights,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} ability usage benchmarks")
    return benchmarks


def compute_item_usage_benchmarks(
    merged_df: pd.DataFrame,
    patch: str = "current",
) -> list[dict]:
    """Compute per-hero, per-item activation benchmarks."""
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    for hero_id, hero_group in merged_df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue
        hero_id = int(hero_id)
        total_fights = len(hero_group)
        if total_fights < MIN_SAMPLES:
            continue

        item_stats = {}
        for _, row in hero_group.iterrows():
            item_uses = row.get("item_uses")
            if not isinstance(item_uses, dict):
                continue
            for item_key, count in item_uses.items():
                if item_key in IGNORED_ITEMS:
                    continue
                if item_key not in item_stats:
                    item_stats[item_key] = []
                item_stats[item_key].append(count)

        for item_key, activations in item_stats.items():
            fights_with = len(activations)
            if fights_with < MIN_SAMPLES:
                continue

            act_series = pd.Series(activations)
            benchmarks.append({
                "hero_id": hero_id,
                "item_key": item_key,
                "time_bucket": "all",
                "size_bucket": "all",
                "usage_rate": round(fights_with / total_fights, 4),
                "avg_activations": round(float(act_series.mean()), 2),
                "median_activations": round(float(act_series.median()), 2),
                "p75_activations": round(float(np.percentile(act_series, 75)), 2),
                "p90_activations": round(float(np.percentile(act_series, 90)), 2),
                "sample_count": total_fights,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} item usage benchmarks")
    return benchmarks


def compute_kill_priority_benchmarks(
    merged_df: pd.DataFrame,
    patch: str = "current",
) -> list[dict]:
    """Compute kill priority: which heroes does each hero tend to kill?"""
    benchmarks = []
    now = datetime.now(timezone.utc).isoformat()

    for hero_id, hero_group in merged_df.groupby("hero_id"):
        if pd.isna(hero_id):
            continue
        hero_id = int(hero_id)
        total_fights = len(hero_group)
        if total_fights < MIN_SAMPLES:
            continue

        kill_counts = {}
        for _, row in hero_group.iterrows():
            killed = row.get("killed")
            if not isinstance(killed, dict):
                continue
            for target_key, count in killed.items():
                kill_counts[target_key] = kill_counts.get(target_key, 0) + count

        for target_key, total_kills in kill_counts.items():
            target_name = target_key.replace("npc_dota_hero_", "")
            benchmarks.append({
                "hero_id": hero_id,
                "target_hero_id": target_name,
                "time_bucket": "all",
                "kill_rate": round(total_kills / total_fights, 4),
                "avg_kills": round(total_kills / total_fights, 4),
                "sample_count": total_fights,
                "patch": patch,
                "updated_at": now,
            })

    logger.info(f"Computed {len(benchmarks)} kill priority benchmarks")
    return benchmarks


# ── Action Comparison Builder ─────────────────────────────

def build_action_comparison(
    player_fight_stat: dict,
    hero_id: int,
    fight_context: dict,
    ability_benchmarks: list[dict],
    item_benchmarks: list[dict],
    kill_benchmarks: list[dict] | None = None,
) -> dict:
    """
    Build side-by-side comparison for one player in one fight.
    Returns structured dict for the frontend ActionComparison component.
    """
    user_abilities = player_fight_stat.get("ability_uses") or {}
    user_items = player_fight_stat.get("item_uses") or {}
    user_kills = player_fight_stat.get("killed") or {}

    # ── Abilities ─────────────────────────────────────────
    ability_bench_map = {b["ability_key"]: b for b in ability_benchmarks}
    all_ability_keys = set(user_abilities.keys()) | set(ability_bench_map.keys())

    abilities = []
    abilities_used = 0
    abilities_expected = 0
    unused_abilities = []

    for key in sorted(all_ability_keys):
        if key in IGNORED_ABILITIES:
            continue
        user_casts = user_abilities.get(key, 0)
        bench = ability_bench_map.get(key)

        entry = {
            "key": key,
            "name": _fmt_ability(key),
            "user_casts": user_casts,
            "pro_avg_casts": bench["avg_casts"] if bench else None,
            "pro_median_casts": bench["median_casts"] if bench else None,
            "pro_p75_casts": bench["p75_casts"] if bench else None,
            "pro_usage_rate": bench["usage_rate"] if bench else None,
            "pro_casts_per_sec": bench.get("avg_casts_per_sec") if bench else None,
            "sample_count": bench["sample_count"] if bench else 0,
        }

        if user_casts == 0 and bench and bench["usage_rate"] >= 0.5:
            entry["status"] = "unused"
            unused_abilities.append(key)
        elif user_casts == 0 and (not bench or bench["usage_rate"] < 0.5):
            entry["status"] = "optional"
        elif bench and user_casts < bench["median_casts"] * 0.6:
            entry["status"] = "underused"
        elif bench and user_casts >= bench["p75_casts"]:
            entry["status"] = "good"
        elif not bench and user_casts > 0:
            entry["status"] = "extra"
        else:
            entry["status"] = "ok"

        if user_casts > 0 or (bench and bench["usage_rate"] >= 0.3):
            abilities.append(entry)
            if user_casts > 0:
                abilities_used += 1
            if bench and bench["usage_rate"] >= 0.5:
                abilities_expected += 1

    status_order = {"unused": 0, "underused": 1, "ok": 2, "good": 3, "optional": 4, "extra": 5}
    abilities.sort(key=lambda a: status_order.get(a["status"], 3))

    # ── Items ─────────────────────────────────────────────
    item_bench_map = {b["item_key"]: b for b in item_benchmarks}
    all_item_keys = set(user_items.keys()) | set(item_bench_map.keys())

    items = []
    items_activated = 0
    items_expected = 0
    missed_key_items = []

    for key in sorted(all_item_keys):
        if key in IGNORED_ITEMS:
            continue
        user_act = user_items.get(key, 0)
        bench = item_bench_map.get(key)

        entry = {
            "key": key,
            "name": _fmt_item(key),
            "user_activations": user_act,
            "pro_avg_activations": bench["avg_activations"] if bench else None,
            "pro_usage_rate": bench["usage_rate"] if bench else None,
            "pro_median_activations": bench["median_activations"] if bench else None,
            "is_key_item": key in KEY_ITEMS,
            "sample_count": bench["sample_count"] if bench else 0,
        }

        if user_act == 0 and bench and bench["usage_rate"] >= 0.5:
            entry["status"] = "missed"
            if key in KEY_ITEMS:
                missed_key_items.append(key)
        elif user_act == 0:
            entry["status"] = "optional"
        elif bench:
            entry["status"] = "used"
        else:
            entry["status"] = "extra"

        if user_act > 0:
            items_activated += 1
        if bench and bench["usage_rate"] >= 0.5:
            items_expected += 1

        if user_act > 0 or (bench and bench["usage_rate"] >= 0.2) or key in KEY_ITEMS:
            items.append(entry)

    items.sort(key=lambda i: ({"missed": 0, "used": 1, "optional": 2, "extra": 3}.get(i["status"], 2), not i["is_key_item"]))

    # ── Kills ─────────────────────────────────────────────
    kills_data = None
    if kill_benchmarks:
        pro_priority = sorted(
            [{"target": b["target_hero_id"], "kill_rate": b["kill_rate"]} for b in kill_benchmarks],
            key=lambda x: -x["kill_rate"],
        )[:5]
        kills_data = {"user_kills": user_kills, "pro_priority": pro_priority}

    return {
        "abilities": abilities,
        "items": items,
        "kills": kills_data,
        "summary": {
            "abilities_used": abilities_used,
            "abilities_expected": abilities_expected,
            "items_activated": items_activated,
            "items_expected": items_expected,
            "missed_key_items": [_fmt_item(k) for k in missed_key_items],
            "unused_abilities": [_fmt_ability(k) for k in unused_abilities],
        },
    }


def _fmt_ability(key: str) -> str:
    parts = key.split("_")
    for i in range(1, min(4, len(parts))):
        candidate = "_".join(parts[i:])
        if len(candidate) > 2:
            return candidate.replace("_", " ").title()
    return key.replace("_", " ").title()


def _fmt_item(key: str) -> str:
    return key.replace("_", " ").title()