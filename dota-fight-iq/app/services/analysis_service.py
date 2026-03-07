"""
Analysis Service

Bridges the gap between raw data and ML-powered insights.
For each analysis dimension (laning, farming, items, objectives, support),
this service:
    1. Fetches the player's raw metrics
    2. Looks up the relevant 7k+ benchmarks for their hero/context
    3. Computes percentile ranks for every metric
    4. Generates a dimension score (0-100)
    5. Produces actionable recommendations

Used by API endpoints to enrich raw stats with benchmark comparisons.
"""

import logging
from app.core import database as db
from app.core.database import get_supabase
from app.ml.extended_benchmarks import (
    percentile_from_benchmark,
    compute_dimension_score,
    compute_overall_match_score,
    generate_extended_recommendations,
    LANING_METRICS,
    FARMING_METRICS,
    SUPPORT_METRICS,
)

logger = logging.getLogger(__name__)


class AnalysisService:
    """Stateless service — all state comes from DB lookups."""

    # ── Benchmark Fetching ─────────────────────────────────

    @staticmethod
    def _get_benchmarks_for_hero(
        hero_id: int,
        time_bucket: str,
        nw_bucket: str = "all",
    ) -> dict[str, dict]:
        """
        Fetch all benchmark rows for a hero in a given context.
        Returns {metric_name: {p25, median, p75, p90, sample_count}}.
        """
        sb = get_supabase()
        result = (
            sb.table("hero_benchmarks")
            .select("*")
            .eq("hero_id", hero_id)
            .eq("time_bucket", time_bucket)
            .eq("nw_bucket", nw_bucket)
            .execute()
        )
        benchmarks = {}
        for row in (result.data or []):
            benchmarks[row["metric_name"]] = row
        return benchmarks

    @staticmethod
    def _get_item_benchmarks_for_hero(hero_id: int) -> dict[str, dict]:
        """Fetch item timing benchmarks for a hero."""
        rows = db.get_item_timing_benchmarks(hero_id)
        return {row["item_key"]: row for row in rows}

    @staticmethod
    def _get_objective_benchmarks() -> dict[str, dict]:
        """Fetch all objective timing benchmarks."""
        rows = db.get_objective_benchmarks()
        return {row["objective_type"]: row for row in rows}

    # ── Laning Analysis ────────────────────────────────────

    def analyze_laning(self, match_id: int) -> list[dict]:
        """
        Enrich laning data with benchmark comparisons and recommendations.
        Returns list of player analysis dicts.
        """
        laning_data = db.get_laning_for_match(match_id)
        if not laning_data:
            return []

        results = []
        for player in laning_data:
            hero_id = player.get("hero_id")
            lane = player.get("lane")
            if not hero_id:
                results.append(player)
                continue

            # Fetch benchmarks for this hero in this lane
            nw_bucket = f"lane_{lane}" if lane else "all"
            benchmarks = self._get_benchmarks_for_hero(hero_id, "laning", nw_bucket)

            # Build metrics dict from player data
            player_metrics = {}
            for metric in LANING_METRICS:
                val = player.get(metric)
                if val is not None:
                    player_metrics[metric] = val

            # Compute percentiles for each metric
            comparisons = []
            for metric, value in player_metrics.items():
                bench = benchmarks.get(metric)
                if bench and bench.get("median") is not None:
                    pct = percentile_from_benchmark(value, bench)
                    comparisons.append({
                        "metric": metric,
                        "value": value,
                        "median": bench["median"],
                        "p75": bench.get("p75"),
                        "p90": bench.get("p90"),
                        "percentile": round(pct, 1),
                        "sample_count": bench.get("sample_count", 0),
                    })

            # Dimension score
            inverted = {"deaths_in_lane"}
            laning_score = compute_dimension_score(player_metrics, benchmarks, inverted)

            # Recommendations
            from app.lib.constants import HERO_NAMES
            hero_name = HERO_NAMES.get(hero_id, f"Hero {hero_id}")
            recommendations = generate_extended_recommendations(
                "laning", player_metrics, benchmarks, hero_name
            )

            results.append({
                **player,
                "benchmarks": comparisons,
                "laning_score": laning_score,
                "recommendations": recommendations,
            })

        return results

    # ── Farming Analysis ───────────────────────────────────

    def analyze_farming(self, match_id: int) -> list[dict]:
        """Enrich farming data with benchmark comparisons."""
        farming_data = db.get_farming_for_match(match_id)
        if not farming_data:
            return []

        results = []
        for player in farming_data:
            hero_id = player.get("hero_id")
            if not hero_id:
                results.append(player)
                continue

            benchmarks = self._get_benchmarks_for_hero(hero_id, "farming")

            player_metrics = {}
            for metric in FARMING_METRICS:
                val = player.get(metric)
                if val is not None:
                    player_metrics[metric] = val

            # Also add gold-at-time metrics from gold_t array
            gold_t = player.get("gold_t") or []
            for minute in [15, 20, 25, 30, 35]:
                if minute < len(gold_t):
                    player_metrics[f"gold_at_{minute}"] = gold_t[minute]

            comparisons = []
            for metric, value in player_metrics.items():
                bench = benchmarks.get(metric)
                if bench and bench.get("median") is not None:
                    pct = percentile_from_benchmark(value, bench)
                    comparisons.append({
                        "metric": metric,
                        "value": value,
                        "median": bench["median"],
                        "p75": bench.get("p75"),
                        "p90": bench.get("p90"),
                        "percentile": round(pct, 1),
                        "sample_count": bench.get("sample_count", 0),
                    })

            inverted = {"estimated_idle_minutes"}
            farming_score = compute_dimension_score(player_metrics, benchmarks, inverted)

            from app.lib.constants import HERO_NAMES
            hero_name = HERO_NAMES.get(hero_id, f"Hero {hero_id}")
            recommendations = generate_extended_recommendations(
                "farming", player_metrics, benchmarks, hero_name
            )

            results.append({
                **player,
                "benchmarks": comparisons,
                "farming_score": farming_score,
                "recommendations": recommendations,
            })

        return results

    # ── Itemization Analysis ───────────────────────────────

    def analyze_items(self, match_id: int) -> list[dict]:
        """Enrich itemization data with timing benchmarks."""
        item_data = db.get_itemization_for_match(match_id)
        if not item_data:
            return []

        results = []
        for player in item_data:
            hero_id = player.get("hero_id")
            if not hero_id:
                results.append(player)
                continue

            item_benchmarks = self._get_item_benchmarks_for_hero(hero_id)
            item_timings = player.get("item_timings") or {}

            # Compare each key item's timing to benchmarks
            item_comparisons = []
            for item_key, player_time in item_timings.items():
                bench = item_benchmarks.get(item_key)
                if bench and bench.get("median_time") is not None:
                    # For items, lower time = better, so we invert the percentile
                    median = bench["median_time"]
                    p25 = bench.get("p25_time", median)
                    p75 = bench.get("p75_time", median)

                    # Faster than p25 = great, slower than p75 = bad
                    if p25 == p75:
                        pct = 50.0
                    else:
                        # Invert: lower time = higher percentile
                        raw_pct = percentile_from_benchmark(
                            player_time,
                            {"p25": p25, "median": median, "p75": p75, "p90": bench.get("p75_time", p75)},
                        )
                        pct = 100 - raw_pct  # invert so faster = higher

                    delta_seconds = player_time - median

                    item_comparisons.append({
                        "item": item_key,
                        "your_time": player_time,
                        "median_time": median,
                        "p25_time": p25,
                        "p75_time": p75,
                        "percentile": round(pct, 1),
                        "delta_seconds": round(delta_seconds),
                        "purchase_rate": bench.get("purchase_rate"),
                        "sample_count": bench.get("sample_count", 0),
                        "is_fast": delta_seconds < 0,
                    })

            # Sort by delta (most delayed items first)
            item_comparisons.sort(key=lambda x: x["delta_seconds"], reverse=True)

            # Generate item-specific recommendations
            recommendations = []
            from app.lib.constants import HERO_NAMES
            hero_name = HERO_NAMES.get(hero_id, f"Hero {hero_id}")

            for ic in item_comparisons:
                item_name = ic["item"].replace("_", " ").title()
                if ic["delta_seconds"] > 120:  # More than 2 min late
                    recommendations.append({
                        "metric": ic["item"],
                        "category": "items",
                        "value": ic["your_time"],
                        "median": ic["median_time"],
                        "percentile": ic["percentile"],
                        "is_strength": False,
                        "is_improvement_area": True,
                        "text": f"Your {item_name} came {abs(ic['delta_seconds']) // 60}:{abs(ic['delta_seconds']) % 60:02d} later than 7k+ median for {hero_name}. Look for faster farming patterns or fewer unnecessary purchases.",
                        "priority": min(ic["delta_seconds"] / 60, 30),
                        "sample_count": ic["sample_count"],
                    })
                elif ic["delta_seconds"] < -120:  # More than 2 min early
                    recommendations.append({
                        "metric": ic["item"],
                        "category": "items",
                        "value": ic["your_time"],
                        "median": ic["median_time"],
                        "percentile": ic["percentile"],
                        "is_strength": True,
                        "is_improvement_area": False,
                        "text": f"Fast {item_name} timing — {abs(ic['delta_seconds']) // 60}:{abs(ic['delta_seconds']) % 60:02d} ahead of 7k+ median. Strong item progression.",
                        "priority": min(abs(ic["delta_seconds"]) / 60, 30),
                        "sample_count": ic["sample_count"],
                    })

            # Item score: average percentile across timed items
            if item_comparisons:
                item_score = round(
                    sum(ic["percentile"] for ic in item_comparisons) / len(item_comparisons), 1
                )
            else:
                item_score = 50.0

            recommendations.sort(key=lambda r: r["priority"], reverse=True)

            results.append({
                **player,
                "item_benchmarks": item_comparisons,
                "item_score": item_score,
                "recommendations": recommendations,
            })

        return results

    # ── Objectives Analysis ────────────────────────────────

    def analyze_objectives(self, match_id: int) -> dict:
        """Enrich objectives with timing benchmarks."""
        objectives = db.get_objectives_for_match(match_id)
        obj_benchmarks = self._get_objective_benchmarks()

        if not objectives:
            return {"objectives": [], "comparisons": [], "recommendations": []}

        comparisons = []
        recommendations = []

        # Compare key objective timings
        for obj in objectives:
            obj_type = obj.get("type", "")
            subtype = obj.get("subtype", "")
            time_val = obj.get("time", 0)

            # Check both direct type and building subtype
            bench_key = obj_type
            if obj_type == "building_kill" and subtype:
                bench_key = f"building_{subtype}"

            bench = obj_benchmarks.get(bench_key)
            if bench and bench.get("median") is not None:
                pct = percentile_from_benchmark(time_val, bench)
                delta = time_val - bench["median"]

                comp = {
                    "objective": bench_key,
                    "your_time": time_val,
                    "median_time": bench["median"],
                    "p25_time": bench.get("p25"),
                    "p75_time": bench.get("p75"),
                    "percentile": round(pct, 1),
                    "delta_seconds": round(delta),
                    "sample_count": bench.get("sample_count", 0),
                }
                comparisons.append(comp)

                # Generate recommendation for significantly late objectives
                if delta > 120:  # 2+ min late
                    obj_name = bench_key.replace("_", " ").replace("building ", "").title()
                    recommendations.append({
                        "metric": bench_key,
                        "category": "objectives",
                        "percentile": round(pct, 1),
                        "is_strength": False,
                        "is_improvement_area": True,
                        "text": f"{obj_name} fell {abs(delta) // 60} min behind 7k+ pace (yours: {time_val // 60}:{time_val % 60:02d}, median: {int(bench['median']) // 60}:{int(bench['median']) % 60:02d}). Prioritize objectives after winning fights.",
                        "priority": min(delta / 60, 20),
                        "sample_count": bench.get("sample_count", 0),
                    })
                elif delta < -120:
                    obj_name = bench_key.replace("_", " ").replace("building ", "").title()
                    recommendations.append({
                        "metric": bench_key,
                        "category": "objectives",
                        "percentile": round(pct, 1),
                        "is_strength": True,
                        "is_improvement_area": False,
                        "text": f"Fast {obj_name} — {abs(delta) // 60} min ahead of 7k+ pace. Strong map pressure.",
                        "priority": min(abs(delta) / 60, 20),
                        "sample_count": bench.get("sample_count", 0),
                    })

        # Objective conversion rate (fights won → objectives taken)
        # This would need fight data cross-referenced, deferred for now

        recommendations.sort(key=lambda r: r["priority"], reverse=True)

        return {
            "objectives": objectives,
            "comparisons": comparisons,
            "recommendations": recommendations,
        }

    # ── Support Analysis ───────────────────────────────────

    def analyze_supports(self, match_id: int) -> dict:
        """Enrich support data with benchmarks and recommendations."""
        players = db.get_match_players(match_id)
        wards = db.get_wards_for_match(match_id)

        if not players:
            return {"supports": [], "all_wards": []}

        # Identify supports by lowest NW per side
        radiant = sorted(
            [p for p in players if p.get("is_radiant")],
            key=lambda p: p.get("net_worth") or 0,
        )
        dire = sorted(
            [p for p in players if not p.get("is_radiant")],
            key=lambda p: p.get("net_worth") or 0,
        )

        support_players = []
        if len(radiant) >= 2:
            support_players.extend(radiant[:2])
        if len(dire) >= 2:
            support_players.extend(dire[:2])

        # Group wards by player
        wards_by_player = {}
        for w in wards:
            aid = w.get("account_id")
            wards_by_player.setdefault(aid, []).append(w)

        results = []
        for s in support_players:
            hero_id = s.get("hero_id")
            aid = s.get("account_id")
            player_wards = wards_by_player.get(aid, [])

            # Fetch support benchmarks for this hero
            benchmarks = self._get_benchmarks_for_hero(hero_id, "support")

            # Build metrics
            player_metrics = {}
            for metric in SUPPORT_METRICS:
                val = s.get(metric)
                if val is not None:
                    player_metrics[metric] = val

            # Ward stats
            obs = [w for w in player_wards if w.get("ward_type") == "observer"]
            sen = [w for w in player_wards if w.get("ward_type") == "sentry"]
            avg_obs_life = (
                sum(w.get("duration_alive") or 0 for w in obs) / len(obs)
                if obs else 0
            )
            dewarded = sum(1 for w in obs if w.get("is_dewarded"))

            # Compute comparisons
            comparisons = []
            for metric, value in player_metrics.items():
                bench = benchmarks.get(metric)
                if bench and bench.get("median") is not None:
                    pct = percentile_from_benchmark(value, bench)
                    comparisons.append({
                        "metric": metric,
                        "value": value,
                        "median": bench["median"],
                        "p75": bench.get("p75"),
                        "p90": bench.get("p90"),
                        "percentile": round(pct, 1),
                        "sample_count": bench.get("sample_count", 0),
                    })

            # Ward lifespan benchmark
            obs_life_bench = benchmarks.get("obs_avg_lifespan")
            if obs_life_bench and avg_obs_life > 0:
                pct = percentile_from_benchmark(avg_obs_life, obs_life_bench)
                comparisons.append({
                    "metric": "obs_avg_lifespan",
                    "value": round(avg_obs_life),
                    "median": obs_life_bench.get("median"),
                    "percentile": round(pct, 1),
                    "sample_count": obs_life_bench.get("sample_count", 0),
                })

            # Score
            support_score = compute_dimension_score(player_metrics, benchmarks)

            # Recommendations
            from app.lib.constants import HERO_NAMES
            hero_name = HERO_NAMES.get(hero_id, f"Hero {hero_id}")
            recommendations = generate_extended_recommendations(
                "support", player_metrics, benchmarks, hero_name
            )

            results.append({
                **s,
                "ward_summary": {
                    "obs_count": len(obs),
                    "sen_count": len(sen),
                    "avg_obs_lifespan": round(avg_obs_life),
                    "obs_dewarded": dewarded,
                },
                "wards": player_wards,
                "benchmarks": comparisons,
                "support_score": support_score,
                "recommendations": recommendations,
            })

        return {
            "supports": results,
            "all_wards": wards,
        }

    # ── Overall Match Rankings (Feature 9) ─────────────────

    def compute_match_rankings(self, match_id: int) -> list[dict]:
        """
        Compute overall scores for all 10 players and rank them.
        Pulls together sub-scores from all dimensions.
        """
        players = db.get_match_players(match_id)
        if not players:
            return []

        # Get all dimension data
        laning = {p["player_slot"]: p for p in self.analyze_laning(match_id)}
        farming = {p["player_slot"]: p for p in self.analyze_farming(match_id)}
        items_data = {p["player_slot"]: p for p in self.analyze_items(match_id)}

        # Fight scores from fight_scores table
        sb = get_supabase()
        fight_scores_raw = (
            sb.table("fight_scores")
            .select("*")
            .eq("match_id", match_id)
            .execute()
        ).data or []

        # Average fight score per player
        fight_avg = {}
        for fs in fight_scores_raw:
            slot = fs.get("player_slot")
            if slot not in fight_avg:
                fight_avg[slot] = []
            if fs.get("fight_iq_score"):
                fight_avg[slot].append(fs["fight_iq_score"])

        fight_scores = {
            slot: sum(scores) / len(scores)
            for slot, scores in fight_avg.items()
            if scores
        }

        rankings = []
        for p in players:
            slot = p.get("player_slot")
            hero_id = p.get("hero_id")
            is_radiant = p.get("is_radiant")

            # Determine role (rough heuristic: sorted by NW within team)
            same_side = sorted(
                [pl for pl in players if pl.get("is_radiant") == is_radiant],
                key=lambda pl: pl.get("net_worth") or 0,
                reverse=True,
            )
            pos_idx = next(
                (i for i, pl in enumerate(same_side) if pl.get("player_slot") == slot),
                2,
            )
            role = f"pos{pos_idx + 1}"

            sub_scores = {
                "fight": fight_scores.get(slot),
                "laning": laning.get(slot, {}).get("laning_score"),
                "farming": farming.get(slot, {}).get("farming_score"),
                "items": items_data.get(slot, {}).get("item_score"),
                "support": None,  # Only for pos 4/5
                "deaths": None,  # Future: death avoidance score
                "objectives": None,  # Future: objective contribution
            }

            overall, weights = compute_overall_match_score(sub_scores, role)

            rankings.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "hero_id": hero_id,
                "player_slot": slot,
                "is_radiant": is_radiant,
                "role": role,
                "overall_score": overall,
                "fight_score": sub_scores.get("fight"),
                "laning_score": sub_scores.get("laning"),
                "farming_score": sub_scores.get("farming"),
                "itemization_score": sub_scores.get("items"),
                "objective_score": sub_scores.get("objectives"),
                "vision_score": sub_scores.get("support"),
                "death_avoidance_score": sub_scores.get("deaths"),
                "weights_used": weights,
            })

        # Rank by overall score
        rankings.sort(key=lambda r: r["overall_score"] or 0, reverse=True)
        for i, r in enumerate(rankings):
            r["match_rank"] = i + 1

        return rankings


# ── Singleton ─────────────────────────────────────────────

_service: AnalysisService | None = None


def get_analysis_service() -> AnalysisService:
    global _service
    if _service is None:
        _service = AnalysisService()
    return _service