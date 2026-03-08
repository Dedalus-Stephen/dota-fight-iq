"""
build_recommender.py — ML-powered context-aware build recommendations.

Uses pgvector cosine similarity to find the most similar 7k+ matches
based on hero, position, and enemy lineup, then aggregates their
item builds and ability builds into recommendations.

This is the core recommendation engine for:
  - "What items should a 7k+ player build in this situation?"
  - "What ability build would a 7k+ player use here?"
"""

import numpy as np
from collections import Counter, defaultdict


# ── Hero Encoding ─────────────────────────────────────────────
# We use a compact encoding scheme:
# [hero_one_hot(1)] + [position_one_hot(5)] + [enemy_5_hot(~140)] + [ally_4_hot(~140)]
# But for efficiency with pgvector, we use a learned or hashed 64-dim embedding.

NUM_HEROES = 140  # approximate, padded


def encode_match_context(hero_id: int, position: int, enemy_ids: list[int], ally_ids: list[int]) -> list[float]:
    """
    Encode a match context into a 64-dimensional vector for similarity search.
    
    Uses a hash-based embedding that captures:
    - Hero identity (dims 0-15)
    - Position (dims 16-19)  
    - Enemy lineup composition (dims 20-41)
    - Ally lineup composition (dims 42-63)
    """
    vec = np.zeros(64, dtype=np.float32)
    
    # Hero identity — distribute across 16 dims using hash
    for i in range(16):
        vec[i] = np.sin(hero_id * (i + 1) * 0.7)
    
    # Position — one-hot in dims 16-19 (pos 1-5 mapped to 4 dims)
    if 1 <= position <= 5:
        pos_idx = min(position - 1, 3)
        vec[16 + pos_idx] = 1.0
    
    # Enemy lineup — hash each enemy hero across 22 dims (20-41)
    for eid in sorted(enemy_ids)[:5]:
        for j in range(22):
            vec[20 + j] += np.sin(eid * (j + 1) * 0.31) / 5.0
    
    # Ally lineup — hash each ally hero across 22 dims (42-63)
    for aid in sorted(ally_ids)[:4]:
        for j in range(22):
            vec[42 + j] += np.sin(aid * (j + 1) * 0.37) / 4.0
    
    # L2 normalize for cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    
    return vec.tolist()


# ── Item Build Recommender ────────────────────────────────────

def get_recommended_item_build(
    db,
    hero_id: int,
    position: int,
    enemy_hero_ids: list[int],
    ally_hero_ids: list[int],
    top_k: int = 20,
) -> dict:
    """
    Find the most similar 7k+ matches and aggregate their item builds
    into a recommended item progression.
    
    Returns:
        {
            "snapshots": [
                {
                    "index": 0,
                    "median_time": 600,
                    "completed_item": "power_treads",
                    "typical_inventory": ["tango", "magic_wand", "power_treads"],
                    "frequency": 0.85,  # % of similar matches with this item at this stage
                },
                ...
            ],
            "similar_matches_count": 20,
            "confidence": "high" | "medium" | "low",
        }
    """
    # 1. Encode the context
    query_vec = encode_match_context(hero_id, position, enemy_hero_ids, ally_hero_ids)
    
    # 2. Find similar matches via pgvector
    similar_matches = _find_similar_matches(db, hero_id, query_vec, top_k)
    
    if not similar_matches:
        return {"snapshots": [], "similar_matches_count": 0, "confidence": "low"}
    
    match_ids = [m["match_id"] for m in similar_matches]
    
    # 3. Fetch item build snapshots for these matches
    snapshots_result = (
        db.table("item_build_snapshots")
        .select("*")
        .eq("hero_id", hero_id)
        .in_("match_id", match_ids)
        .order("snapshot_index")
        .execute()
    )
    
    if not snapshots_result.data:
        return {"snapshots": [], "similar_matches_count": len(match_ids), "confidence": "low"}
    
    # 4. Aggregate snapshots by index
    aggregated = _aggregate_item_snapshots(snapshots_result.data, len(match_ids))
    
    confidence = "high" if len(match_ids) >= 15 else "medium" if len(match_ids) >= 5 else "low"
    
    return {
        "snapshots": aggregated,
        "similar_matches_count": len(match_ids),
        "confidence": confidence,
    }


def _find_similar_matches(db, hero_id: int, query_vec: list[float], top_k: int) -> list[dict]:
    """Find top_k most similar match contexts using pgvector cosine similarity."""
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"
    
    # Use Supabase RPC for vector similarity search
    result = db.rpc(
        "match_context_similarity_search",
        {
            "query_embedding": vec_str,
            "query_hero_id": hero_id,
            "match_limit": top_k,
        }
    ).execute()
    
    return result.data or []


def _aggregate_item_snapshots(snapshots: list[dict], total_matches: int) -> list[dict]:
    """
    Aggregate item build snapshots from multiple matches into a recommended progression.
    
    Groups by snapshot_index, finds the most common completed_item at each stage,
    and computes median timing + typical inventory.
    """
    by_index = defaultdict(list)
    for snap in snapshots:
        by_index[snap["snapshot_index"]].append(snap)
    
    result = []
    for idx in sorted(by_index.keys()):
        group = by_index[idx]
        
        # Most common completed item at this stage
        item_counts = Counter(s["completed_item"] for s in group)
        top_item, top_count = item_counts.most_common(1)[0]
        
        # Filter to only matches where this was the completed item
        matching = [s for s in group if s["completed_item"] == top_item]
        
        # Median game time
        times = sorted(s["game_time"] for s in matching)
        median_time = times[len(times) // 2] if times else 0
        
        # Most common inventory state
        inv_counter = Counter(tuple(s["inventory"]) for s in matching)
        typical_inv = list(inv_counter.most_common(1)[0][0]) if inv_counter else []
        
        # Alternative items at this stage
        alternatives = [
            {"item": item, "frequency": round(count / len(group), 2)}
            for item, count in item_counts.most_common(3)
            if item != top_item and count / len(group) >= 0.15
        ]
        
        result.append({
            "index": idx,
            "median_time": median_time,
            "completed_item": top_item,
            "typical_inventory": typical_inv,
            "frequency": round(top_count / max(len(group), 1), 2),
            "alternatives": alternatives,
            "sample_count": len(group),
        })
    
    return result


# ── Ability Build Recommender ─────────────────────────────────

def get_recommended_ability_build(
    db,
    hero_id: int,
    position: int,
    enemy_hero_ids: list[int],
    ally_hero_ids: list[int],
    top_k: int = 20,
) -> dict:
    """
    Find similar 7k+ matches and aggregate their ability builds
    into a recommended skill order + talent choices.
    
    Returns:
        {
            "recommended_order": ["ability_q", "ability_w", "ability_q", ...],  # level 1-25
            "level_breakdown": [
                {
                    "level": 1,
                    "recommended": "templar_assassin_refraction",
                    "pick_rates": {
                        "templar_assassin_refraction": 0.85,
                        "templar_assassin_meld": 0.10,
                        "templar_assassin_psi_blades": 0.05,
                    }
                },
                ...
            ],
            "talent_choices": {
                10: {"recommended": "left", "left_rate": 0.72, "right_rate": 0.28,
                     "left_name": "+25 Attack Speed", "right_name": "+200 Refraction Damage"},
                15: {...}, 20: {...}, 25: {...}
            },
            "similar_matches_count": 20,
            "confidence": "high" | "medium" | "low",
        }
    """
    # 1. Encode context
    query_vec = encode_match_context(hero_id, position, enemy_hero_ids, ally_hero_ids)
    
    # 2. Find similar matches
    similar_matches = _find_similar_matches(db, hero_id, query_vec, top_k)
    
    if not similar_matches:
        # Fall back to hero-level benchmarks
        return _fallback_ability_build(db, hero_id, position)
    
    match_ids = [m["match_id"] for m in similar_matches]
    
    # 3. Fetch ability builds for these matches
    builds_result = (
        db.table("ability_builds")
        .select("*")
        .eq("hero_id", hero_id)
        .in_("match_id", match_ids)
        .execute()
    )
    
    if not builds_result.data:
        return _fallback_ability_build(db, hero_id, position)
    
    # 4. Aggregate
    builds = builds_result.data
    aggregated = _aggregate_ability_builds(builds)
    talents = _aggregate_talent_choices(builds)
    
    confidence = "high" if len(builds) >= 15 else "medium" if len(builds) >= 5 else "low"
    
    return {
        "recommended_order": aggregated["recommended_order"],
        "level_breakdown": aggregated["level_breakdown"],
        "talent_choices": talents,
        "similar_matches_count": len(builds),
        "confidence": confidence,
    }


def _aggregate_ability_builds(builds: list[dict]) -> dict:
    """Aggregate multiple ability builds into a recommended order."""
    max_levels = max(len(b.get("ability_order", [])) for b in builds)
    
    recommended_order = []
    level_breakdown = []
    
    for level_idx in range(min(max_levels, 30)):
        level = level_idx + 1
        
        # Count which ability was chosen at this level
        choices = Counter()
        for b in builds:
            order = b.get("ability_order", [])
            if level_idx < len(order):
                choices[order[level_idx]] += 1
        
        if not choices:
            break
        
        top_ability = choices.most_common(1)[0][0]
        total = sum(choices.values())
        
        pick_rates = {
            ability: round(count / total, 2)
            for ability, count in choices.most_common()
        }
        
        recommended_order.append(top_ability)
        level_breakdown.append({
            "level": level,
            "recommended": top_ability,
            "pick_rates": pick_rates,
        })
    
    return {
        "recommended_order": recommended_order,
        "level_breakdown": level_breakdown,
    }


def _aggregate_talent_choices(builds: list[dict]) -> dict:
    """Aggregate talent choices across similar matches."""
    talent_levels = [10, 15, 20, 25]
    result = {}
    
    for tl in talent_levels:
        left_count = 0
        right_count = 0
        left_name = None
        right_name = None
        
        for b in builds:
            tc = b.get("talent_choices") or {}
            choice = tc.get(str(tl)) or tc.get(tl)
            if choice:
                if isinstance(choice, dict):
                    side = choice.get("choice", "")
                    if not left_name:
                        left_name = choice.get("left_name")
                    if not right_name:
                        right_name = choice.get("right_name")
                else:
                    side = choice
                
                if side == "left":
                    left_count += 1
                elif side == "right":
                    right_count += 1
        
        total = left_count + right_count
        if total > 0:
            result[tl] = {
                "recommended": "left" if left_count >= right_count else "right",
                "left_rate": round(left_count / total, 2),
                "right_rate": round(right_count / total, 2),
                "left_name": left_name,
                "right_name": right_name,
                "sample_count": total,
            }
    
    return result


def _fallback_ability_build(db, hero_id: int, position: int | None) -> dict:
    """Fall back to aggregated ability build benchmarks when no similar matches found."""
    query = db.table("ability_build_benchmarks").select("*").eq("hero_id", hero_id)
    
    if position:
        query = query.eq("position", position)
    
    result = query.order("level").execute()
    
    if not result.data:
        return {
            "recommended_order": [],
            "level_breakdown": [],
            "talent_choices": {},
            "similar_matches_count": 0,
            "confidence": "low",
        }
    
    # Group by level, pick highest pick_rate ability at each level
    by_level = defaultdict(list)
    for row in result.data:
        by_level[row["level"]].append(row)
    
    recommended_order = []
    level_breakdown = []
    
    for level in sorted(by_level.keys()):
        options = by_level[level]
        options.sort(key=lambda x: -x["pick_rate"])
        top = options[0]
        
        pick_rates = {o["ability_key"]: o["pick_rate"] for o in options}
        
        recommended_order.append(top["ability_key"])
        level_breakdown.append({
            "level": level,
            "recommended": top["ability_key"],
            "pick_rates": pick_rates,
        })
    
    # Talent choices
    talent_result = (
        db.table("talent_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .execute()
    )
    
    talents = {}
    for t in (talent_result.data or []):
        talents[t["talent_level"]] = {
            "recommended": "left" if (t.get("left_pick_rate") or 0) >= (t.get("right_pick_rate") or 0) else "right",
            "left_rate": t.get("left_pick_rate", 0),
            "right_rate": t.get("right_pick_rate", 0),
            "left_name": t.get("left_name"),
            "right_name": t.get("right_name"),
            "sample_count": t.get("sample_count", 0),
        }
    
    return {
        "recommended_order": recommended_order,
        "level_breakdown": level_breakdown,
        "talent_choices": talents,
        "similar_matches_count": sum(o.get("sample_count", 0) for o in result.data[:1]),
        "confidence": "medium",
    }


# ── Supabase RPC Function (run this in SQL editor) ───────────
# This should be added to the migration, but for reference:
"""
CREATE OR REPLACE FUNCTION match_context_similarity_search(
    query_embedding vector(64),
    query_hero_id INT,
    match_limit INT DEFAULT 20
)
RETURNS TABLE(
    match_id BIGINT,
    hero_id INT,
    position INT,
    enemy_hero_ids INT[],
    similarity FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        m.match_id,
        m.hero_id,
        m.position,
        m.enemy_hero_ids,
        1 - (m.embedding <=> query_embedding) AS similarity
    FROM match_context_vectors m
    WHERE m.hero_id = query_hero_id
    ORDER BY m.embedding <=> query_embedding
    LIMIT match_limit;
$$;
"""