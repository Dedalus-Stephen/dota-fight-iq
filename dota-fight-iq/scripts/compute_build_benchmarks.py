from collections import defaultdict
from app.core.database import get_supabase

def compute_item_timing_benchmarks():
    sb = get_supabase()
    
    # Get all item build snapshots grouped by hero + item
    snapshots = sb.table("item_build_snapshots").select("hero_id, position, completed_item, game_time").execute().data
    
    by_key = defaultdict(list)
    for s in snapshots:
        key = (s["hero_id"], s.get("position"), s["completed_item"])
        by_key[key].append(s["game_time"])
    
    benchmarks = []
    for (hero_id, position, item_key), times in by_key.items():
        if len(times) < 3:
            continue
        times.sort()
        n = len(times)
        benchmarks.append({
            "hero_id": hero_id,
            "position": position,
            "item_key": item_key,
            "median_time": times[n // 2],
            "p25_time": times[n // 4],
            "p75_time": times[3 * n // 4],
            "avg_time": sum(times) / n,
            "purchase_rate": None,  # computed separately from total match count
            "sample_count": n,
        })
    
    if benchmarks:
        sb.table("item_timing_benchmarks").upsert(benchmarks).execute()
        print(f"Upserted {len(benchmarks)} item timing benchmarks")

def compute_ability_build_benchmarks():
    sb = get_supabase()
    
    builds = sb.table("ability_builds").select("hero_id, position, ability_order").execute().data
    
    # Count ability picks per hero per level
    counts = defaultdict(lambda: defaultdict(int))  # (hero, position, level) -> {ability: count}
    totals = defaultdict(int)
    
    for b in builds:
        hero_id = b["hero_id"]
        position = b.get("position")
        order = b.get("ability_order", [])
        
        for level_idx, ability in enumerate(order):
            level = level_idx + 1
            key = (hero_id, position, level)
            counts[key][ability] += 1
            totals[key] += 1
    
    benchmarks = []
    for (hero_id, position, level), abilities in counts.items():
        total = totals[(hero_id, position, level)]
        for ability, count in abilities.items():
            benchmarks.append({
                "hero_id": hero_id,
                "position": position,
                "level": level,
                "ability_key": ability,
                "pick_rate": round(count / total, 3),
                "sample_count": total,
            })
    
    if benchmarks:
        # Batch upsert in chunks
        chunk_size = 500
        for i in range(0, len(benchmarks), chunk_size):
            sb.table("ability_build_benchmarks").upsert(benchmarks[i:i+chunk_size]).execute()
        print(f"Upserted {len(benchmarks)} ability build benchmarks")

if __name__ == "__main__":
    compute_item_timing_benchmarks()
    compute_ability_build_benchmarks()
    print("Done!")