"""
Dota Fight IQ — FastAPI Application

v4: All analysis endpoints return ML-powered benchmark comparisons,
    percentile ranks, dimension scores, and actionable recommendations.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core import database as db
from app.core.database import safe_db_call
from app.services.match_processor import MatchProcessor

from app.api.fight_actions import router as fight_actions_router
from app.api.abilities import router as abilities_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

processor: MatchProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor
    processor = MatchProcessor()
    from app.ml.scoring import get_scorer
    scorer = get_scorer()
    logger.info(f"ML models loaded: {scorer.get_model_info()}")
    logger.info("Dota Fight IQ API started (v4 — benchmark-powered)")
    yield
    if processor:
        await processor.close()


app = FastAPI(title="Dota Fight IQ", version="0.4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(fight_actions_router)
app.include_router(abilities_router) 


@app.get("/health")
async def health():
    from app.ml.scoring import get_scorer
    return {"status": "ok", "version": "0.4.0", "models": get_scorer().get_model_info()}


# ── Match Analysis ─────────────────────────────────────────

@app.post("/api/analyze/{match_id}")
async def analyze_match(match_id: int, background_tasks: BackgroundTasks):
    existing = safe_db_call(db.get_match, match_id)
    if existing and existing.get("processed_at"):
        return {"status": "complete", "match_id": match_id}
    analysis = safe_db_call(db.get_analysis_status, match_id)
    if analysis in ("pending", "processing"):
        return {"status": analysis, "match_id": match_id}
    background_tasks.add_task(_process_match_background, match_id)
    return {"status": "pending", "match_id": match_id}


async def _process_match_background(match_id: int):
    from app.core.database import _reset_client
    try:
        result = await processor.process_match(match_id)
        logger.info(f"Match {match_id}: {result}")
    except Exception as e:
        if "RemoteProtocolError" in str(e) or "ConnectionTerminated" in str(e):
            _reset_client()
            try:
                await processor.process_match(match_id)
            except Exception as e2:
                logger.error(f"Match {match_id} retry failed: {e2}")
        else:
            logger.error(f"Match {match_id} failed: {e}")


@app.get("/api/analyze/{match_id}/status")
async def analysis_status(match_id: int):
    existing = db.get_match(match_id)
    if existing and existing.get("processed_at"):
        return {"status": "complete", "match_id": match_id}
    analysis = db.get_analysis_status(match_id)
    return {"status": analysis or "not_found", "match_id": match_id}


# ── Match Overview (with benchmarks) ──────────────────────

@app.get("/api/matches/{match_id}/overview")
async def match_overview(match_id: int):
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(404, "Match not found")
    players = db.get_match_players(match_id)

    # Enrich with benchmark comparisons
    sb = db.get_supabase()
    for p in players:
        hero_id = p.get("hero_id")
        if not hero_id:
            continue
        try:
            from app.ml.extended_benchmarks import percentile_from_benchmark
            hero_bench = (
                sb.table("hero_benchmarks").select("metric_name, p25, median, p75, p90, sample_count")
                .eq("hero_id", hero_id)
                .in_("time_bucket", ["farming", "laning", "0-15", "15-25", "25-35", "35-45", "45+"])
                .execute()
            ).data or []

            comparisons = []
            metric_map = {"gpm": p.get("gpm"), "xpm": p.get("xpm"), "total_last_hits": p.get("last_hits")}
            for bench in hero_bench:
                m = bench["metric_name"]
                val = metric_map.get(m)
                if val is not None and bench.get("median"):
                    pct = percentile_from_benchmark(val, bench)
                    comparisons.append({"metric": m, "value": val, "median": bench["median"], "percentile": round(pct, 1), "sample_count": bench.get("sample_count", 0)})
            p["benchmark_comparisons"] = comparisons
        except Exception:
            p["benchmark_comparisons"] = []

    return {"match": match, "players": players}


# ── Fights ────────────────────────────────────────────────

@app.get("/api/fights/{match_id}")
async def get_fights(match_id: int):
    fights = db.get_teamfights_for_match(match_id)
    if not fights:
        raise HTTPException(404, "No teamfight data")
    return {"match_id": match_id, "teamfights": fights}


@app.get("/api/fights/{match_id}/{fight_index}")
async def get_fight_detail(match_id: int, fight_index: int):
    from app.ml.scoring import get_scorer
    sb = db.get_supabase()
    fight = (sb.table("teamfights").select("*, fight_player_stats(*)").eq("match_id", match_id).eq("fight_index", fight_index).execute()).data
    if not fight:
        raise HTTPException(404, f"Fight {fight_index} not found")

    fight_data = fight[0]
    scorer = get_scorer()
    match = db.get_match(match_id)
    players = db.get_match_players(match_id)

    scored_players = []
    for stat in fight_data.get("fight_player_stats", []):
        if scorer.is_loaded and scorer.fight_iq:
            try:
                analysis = scorer.score_player_fight(stat, fight_data, match, players)
                scored_players.append({**stat, "analysis": analysis})
            except Exception as e:
                logger.warning(f"Scoring failed: {e}")
                scored_players.append(stat)
        else:
            scored_players.append(stat)

    outcome = None
    if scorer.is_loaded and scorer.fight_outcome:
        outcome = scorer.predict_fight_outcome(fight_data, fight_data.get("fight_player_stats", []), match, players)

    context = db.get_fight_context(fight_data.get("id"))
    return {"match_id": match_id, "fight_index": fight_index, "fight": fight_data, "player_scores": scored_players, "outcome_prediction": outcome, "fight_context": context}


@app.get("/api/fights/{match_id}/{fight_index}/minimap")
async def get_fight_minimap(match_id: int, fight_index: int):
    sb = db.get_supabase()
    fight = (sb.table("teamfights").select("start_time, end_time").eq("match_id", match_id).eq("fight_index", fight_index).execute()).data
    if not fight:
        raise HTTPException(404)
    s, e = fight[0]["start_time"], fight[0]["end_time"]
    pos = (sb.table("player_positions").select("*").eq("match_id", match_id).gte("time", s - 10).lte("time", e + 10).order("time").execute()).data
    return {"match_id": match_id, "fight_index": fight_index, "start_time": s, "end_time": e, "positions": pos or []}


@app.get("/api/fights/{match_id}/{fight_index}/similar")
async def get_similar_fights(match_id: int, fight_index: int, hero_id: int | None = None):
    return {"match_id": match_id, "fight_index": fight_index, "similar_fights": [], "note": "Requires fight vectors"}


# ── Laning (with benchmarks) ─────────────────────────────

@app.get("/api/matches/{match_id}/laning")
async def get_laning_analysis(match_id: int):
    from app.services.analysis_service import get_analysis_service
    players = get_analysis_service().analyze_laning(match_id)
    if not players:
        raise HTTPException(404, "No laning data")
    by_lane = {}
    for p in players:
        lane = p.get("lane")
        if lane:
            by_lane.setdefault(lane, []).append(p)
    return {"match_id": match_id, "players": players, "by_lane": by_lane}


# ── Farming (with benchmarks) ────────────────────────────

@app.get("/api/matches/{match_id}/farming")
async def get_farming_analysis(match_id: int):
    from app.services.analysis_service import get_analysis_service
    players = get_analysis_service().analyze_farming(match_id)
    if not players:
        raise HTTPException(404, "No farming data")
    return {"match_id": match_id, "players": players}


# ── Items (with benchmarks) ──────────────────────────────

@app.get("/api/matches/{match_id}/items")
async def get_itemization(match_id: int):
    from app.services.analysis_service import get_analysis_service
    players = get_analysis_service().analyze_items(match_id)
    if not players:
        raise HTTPException(404, "No itemization data")
    return {"match_id": match_id, "players": players}


# ── Objectives (with benchmarks) ─────────────────────────

@app.get("/api/matches/{match_id}/objectives")
async def get_objectives(match_id: int):
    from app.services.analysis_service import get_analysis_service
    result = get_analysis_service().analyze_objectives(match_id)
    if not result.get("objectives"):
        raise HTTPException(404, "No objective data")
    return {"match_id": match_id, **result}


# ── Supports (with benchmarks) ───────────────────────────

@app.get("/api/matches/{match_id}/supports")
async def get_support_analysis(match_id: int):
    from app.services.analysis_service import get_analysis_service
    result = get_analysis_service().analyze_supports(match_id)
    if not result.get("supports"):
        raise HTTPException(404, "No support data")
    return {"match_id": match_id, **result}


# ── Rankings ──────────────────────────────────────────────

@app.get("/api/matches/{match_id}/rankings")
async def get_match_rankings(match_id: int):
    from app.services.analysis_service import get_analysis_service
    scores = db.get_match_scores(match_id)
    if not scores:
        try:
            scores = get_analysis_service().compute_match_rankings(match_id)
            if scores:
                db.upsert_match_scores(scores)
        except Exception as e:
            logger.warning(f"Ranking failed: {e}")
            raise HTTPException(404, "Rankings could not be computed")
    if not scores:
        raise HTTPException(404, "No ranking data")
    return {"match_id": match_id, "rankings": scores}


# ── Toxicity ──────────────────────────────────────────────

@app.get("/api/matches/{match_id}/toxicity")
async def get_toxicity(match_id: int):
    chat = db.get_chat_for_match(match_id)
    if not chat:
        raise HTTPException(404, "No chat data")
    return {"match_id": match_id, "players": chat, "note": "All-chat only"}


# ── Hero Benchmarks ───────────────────────────────────────

@app.get("/api/heroes/{hero_id}/benchmarks")
async def hero_benchmarks(hero_id: int):
    sb = db.get_supabase()
    benchmarks = sb.table("hero_benchmarks").select("*").eq("hero_id", hero_id).execute().data
    item_benchmarks = db.get_item_timing_benchmarks(hero_id)
    return {"hero_id": hero_id, "benchmarks": benchmarks, "item_timings": item_benchmarks}


# ── ML Models ─────────────────────────────────────────────

@app.get("/api/models/info")
async def model_info():
    from app.ml.scoring import get_scorer
    return get_scorer().get_model_info()

@app.post("/api/models/reload")
async def reload_models():
    from app.ml.scoring import reload_models as _reload
    return {"status": "reloaded", "models": _reload().get_model_info()}


# ── Admin ─────────────────────────────────────────────────

@app.get("/api/admin/collection-status")
async def collection_status():
    sb = db.get_supabase()
    return {
        "matches_stored": sb.table("matches").select("match_id", count="exact").execute().count,
        "teamfights_stored": sb.table("teamfights").select("id", count="exact").execute().count,
        "pool_pending": sb.table("match_collection_pool").select("match_id", count="exact").eq("status", "pending").execute().count,
    }

@app.get("/api/analyze/{match_id}/status")
async def get_analysis_status(match_id: int):
    # Already fully processed?
    existing = safe_db_call(db.get_match, match_id)
    if existing and existing.get("processed_at"):
        return {"status": "complete", "match_id": match_id}

    # Is there a local parse job in flight?
    job = safe_db_call(db.get_latest_parse_job, match_id)
    if job:
        return {
            "status": job["status"],   # queued | parsing | complete | failed
            "job_id": job["job_id"],
            "error": job.get("error"),
            "match_id": match_id,
        }

    # Check if it's sitting in the processing queue (OpenDota path)
    analysis = safe_db_call(db.get_analysis_status, match_id)
    if analysis:
        return {"status": analysis, "match_id": match_id}

    return {"status": "not_started", "match_id": match_id}    