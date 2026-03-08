"""
abilities.py — FastAPI route for ability build analysis.

GET /api/matches/{match_id}/abilities
Returns per-player ability build data with 7k+ benchmarks and recommendations.
"""

from fastapi import APIRouter, HTTPException
from app.core.database import get_supabase
from app.ml.build_recommender import (
    get_recommended_ability_build,
    encode_match_context,
)

router = APIRouter()


@router.get("/api/matches/{match_id}/abilities")
async def get_abilities_analysis(match_id: int):
    """
    Ability build analysis for all players in a match.
    
    Returns:
    - Per-player ability build with level-by-level comparison to 7k+ builds
    - ML-recommended ability order based on similar 7k+ matches
    - Talent choice comparison
    - Ability build score (0-100)
    """
    db = get_supabase()

    # 1. Get match info
    match_result = db.table("matches").select("*").eq("match_id", match_id).single().execute()
    if not match_result.data:
        raise HTTPException(status_code=404, detail="Match not found")

    # 2. Get all players in this match
    players_result = (
        db.table("match_players")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    if not players_result.data:
        raise HTTPException(status_code=404, detail="No player data found")

    players = players_result.data

    # 3. Determine teams
    radiant_heroes = sorted([p["hero_id"] for p in players if p.get("is_radiant")])
    dire_heroes = sorted([p["hero_id"] for p in players if not p.get("is_radiant")])

    # 4. Process each player
    results = []
    for player in players:
        hero_id = player["hero_id"]
        is_radiant = player.get("is_radiant", True)
        position = player.get("lane") or player.get("role") or 0
        enemy_ids = dire_heroes if is_radiant else radiant_heroes
        ally_ids = [h for h in (radiant_heroes if is_radiant else dire_heroes) if h != hero_id]

        # Get player's actual ability build from the match
        player_build = _get_player_ability_build(db, match_id, hero_id)

        # Get ML-recommended build based on similar 7k+ matches
        recommended = get_recommended_ability_build(
            db, hero_id, position, enemy_ids, ally_ids, top_k=20
        )

        # Compute ability build score
        score = _compute_ability_build_score(player_build, recommended)

        # Generate recommendations
        recommendations = _generate_ability_recommendations(
            player_build, recommended, hero_id
        )

        results.append({
            "player_slot": player.get("player_slot"),
            "hero_id": hero_id,
            "is_radiant": is_radiant,
            "position": position,
            "player_build": player_build,
            "recommended_build": recommended,
            "ability_build_score": score,
            "recommendations": recommendations,
            "enemy_hero_ids": enemy_ids,
        })

    return {"match_id": match_id, "players": results}


def _get_player_ability_build(db, match_id: int, hero_id: int) -> dict:
    """Fetch the player's actual ability upgrade order from the match."""
    # Try ability_builds table first (if populated from parsed data)
    result = (
        db.table("ability_builds")
        .select("*")
        .eq("match_id", match_id)
        .eq("hero_id", hero_id)
        .execute()
    )

    if result.data:
        build = result.data[0]
        return {
            "ability_order": build.get("ability_order", []),
            "talent_choices": build.get("talent_choices", {}),
        }

    # Fall back to match_players ability_upgrades if available
    player_result = (
        db.table("match_players")
        .select("ability_upgrades")
        .eq("match_id", match_id)
        .eq("hero_id", hero_id)
        .single()
        .execute()
    )

    if player_result.data and player_result.data.get("ability_upgrades"):
        upgrades = player_result.data["ability_upgrades"]
        # OpenDota format: [{ability: id, time: seconds, level: 1}, ...]
        ability_order = [u.get("ability") or u.get("ability_id", "") for u in upgrades]
        return {
            "ability_order": ability_order,
            "talent_choices": _extract_talents_from_upgrades(upgrades),
        }

    return {"ability_order": [], "talent_choices": {}}


def _extract_talents_from_upgrades(upgrades: list) -> dict:
    """Extract talent choices from ability upgrade list."""
    talents = {}
    talent_levels = [10, 15, 20, 25]

    for u in upgrades:
        level = u.get("level")
        if level in talent_levels:
            ability = u.get("ability") or u.get("ability_id", "")
            # Talent abilities typically have "special_bonus" or "talent" in name
            if "special_bonus" in str(ability) or "talent" in str(ability):
                # Determine left/right based on ability ID convention
                # Even IDs = left, odd = right (Dota convention)
                side = "left" if ability % 2 == 0 else "right" if isinstance(ability, int) else "unknown"
                talents[level] = {"choice": side, "ability_id": ability}

    return talents


def _compute_ability_build_score(player_build: dict, recommended: dict) -> float | None:
    """
    Score how closely the player's ability build matches the 7k+ recommended build.
    
    Scoring:
    - Each level where player matches the top recommendation: full points
    - Each level where player matches a common alternative (>20% pick rate): partial points
    - Talent choices: bonus points for matching recommended talents
    """
    if not recommended.get("level_breakdown") or not player_build.get("ability_order"):
        return None

    player_order = player_build["ability_order"]
    level_breakdown = recommended["level_breakdown"]

    total_points = 0
    max_points = 0

    for entry in level_breakdown:
        level = entry["level"]
        level_idx = level - 1

        if level_idx >= len(player_order):
            continue

        player_choice = player_order[level_idx]
        pick_rates = entry.get("pick_rates", {})
        recommended_ability = entry["recommended"]

        max_points += 1.0

        if str(player_choice) == str(recommended_ability):
            total_points += 1.0
        elif str(player_choice) in pick_rates:
            # Partial credit based on how popular this choice is among 7k+ players
            rate = pick_rates[str(player_choice)]
            total_points += min(rate * 1.5, 0.8)  # cap at 0.8

    # Talent bonus (up to 20% of total score)
    talent_score = _score_talents(
        player_build.get("talent_choices", {}),
        recommended.get("talent_choices", {})
    )

    if max_points == 0:
        return None

    base_score = (total_points / max_points) * 80  # 80% from skill order
    talent_bonus = talent_score * 20  # 20% from talents

    return round(base_score + talent_bonus, 1)


def _score_talents(player_talents: dict, recommended_talents: dict) -> float:
    """Score talent choices (0.0 to 1.0)."""
    if not recommended_talents:
        return 0.5  # neutral if no data

    matches = 0
    total = 0

    for level, rec in recommended_talents.items():
        level_key = str(level)
        total += 1

        player_choice = player_talents.get(level_key) or player_talents.get(int(level_key) if level_key.isdigit() else level_key)
        if not player_choice:
            continue

        player_side = player_choice.get("choice") if isinstance(player_choice, dict) else player_choice
        if player_side == rec.get("recommended"):
            matches += 1

    return matches / total if total > 0 else 0.5


def _generate_ability_recommendations(
    player_build: dict, recommended: dict, hero_id: int
) -> list[dict]:
    """Generate actionable recommendations comparing player build to 7k+."""
    recs = []
    player_order = player_build.get("ability_order", [])
    level_breakdown = recommended.get("level_breakdown", [])

    # Find levels where player deviated significantly
    deviations = []
    for entry in level_breakdown:
        level = entry["level"]
        level_idx = level - 1

        if level_idx >= len(player_order):
            continue

        player_choice = str(player_order[level_idx])
        recommended_ability = entry["recommended"]
        pick_rates = entry.get("pick_rates", {})

        rec_rate = pick_rates.get(recommended_ability, 0)
        player_rate = pick_rates.get(player_choice, 0)

        # Significant deviation: player chose something with <15% pick rate
        # when the recommended had >50% pick rate
        if player_choice != recommended_ability and rec_rate > 0.5 and player_rate < 0.15:
            deviations.append({
                "level": level,
                "player_choice": player_choice,
                "recommended": recommended_ability,
                "rec_rate": rec_rate,
            })

    if deviations:
        # Group early vs late deviations
        early = [d for d in deviations if d["level"] <= 10]
        late = [d for d in deviations if d["level"] > 10]

        if early:
            levels = ", ".join(str(d["level"]) for d in early[:3])
            rec_ability = _format_ability_name(early[0]["recommended"])
            recs.append({
                "type": "skill_order",
                "severity": "high",
                "text": f"At level(s) {levels}, {round(early[0]['rec_rate'] * 100)}% of 7k+ players skill {rec_ability}. "
                        f"Consider prioritizing this ability earlier.",
                "is_improvement_area": True,
            })

        if late:
            levels = ", ".join(str(d["level"]) for d in late[:3])
            recs.append({
                "type": "skill_order",
                "severity": "medium",
                "text": f"Mid-late skill deviations at level(s) {levels}. "
                        f"Your skill build diverged from the 7k+ consensus.",
                "is_improvement_area": True,
            })

    # Talent recommendations
    player_talents = player_build.get("talent_choices", {})
    rec_talents = recommended.get("talent_choices", {})

    for level, rec_talent in rec_talents.items():
        level_key = str(level)
        player_choice = player_talents.get(level_key) or player_talents.get(int(level_key) if str(level_key).isdigit() else level_key)
        
        if not player_choice:
            continue

        player_side = player_choice.get("choice") if isinstance(player_choice, dict) else player_choice
        rec_side = rec_talent.get("recommended")
        dominant_rate = max(rec_talent.get("left_rate", 0), rec_talent.get("right_rate", 0))

        if player_side and player_side != rec_side and dominant_rate >= 0.7:
            talent_name = rec_talent.get(f"{rec_side}_name", f"the {rec_side} talent")
            recs.append({
                "type": "talent",
                "severity": "medium",
                "text": f"Level {level} talent: {round(dominant_rate * 100)}% of 7k+ players pick {talent_name}.",
                "is_improvement_area": True,
            })

    # Positive feedback if build is close
    if not deviations:
        recs.append({
            "type": "positive",
            "severity": "info",
            "text": "Your ability build closely matches the 7k+ recommended order. Well done!",
            "is_improvement_area": False,
        })

    return recs


def _format_ability_name(raw: str) -> str:
    """Format an ability key into a readable name."""
    parts = raw.split("_")
    # Skip hero prefix (usually first 1-3 parts)
    for i in range(1, min(4, len(parts))):
        candidate = "_".join(parts[i:])
        if len(candidate) > 2:
            return candidate.replace("_", " ").title()
    return raw.replace("_", " ").title()