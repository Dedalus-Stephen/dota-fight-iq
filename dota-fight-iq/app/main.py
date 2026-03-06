"""
Dota Fight IQ — FastAPI Application

Main API server. Handles match analysis requests and serves processed data.
Phase 2: ML scoring integrated — Fight IQ scores, outcome predictions,
         benchmarks, recommendations, fight archetypes.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core import database as db
from app.services.match_processor import MatchProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared instances
processor: MatchProcessor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global processor
    processor = MatchProcessor()

    # Load ML models into memory
    from app.ml.scoring import get_scorer
    scorer = get_scorer()
    logger.info(f"ML models loaded: {scorer.get_model_info()}")

    logger.info("Dota Fight IQ API started")
    yield
    if processor:
        await processor.close()
    logger.info("Dota Fight IQ API stopped")


app = FastAPI(
    title="Dota Fight IQ",
    description="ML-Powered Teamfight Performance Analyzer",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────

@app.get("/health")
async def health():
    from app.ml.scoring import get_scorer
    scorer = get_scorer()
    return {
        "status": "ok",
        "service": "dota-fight-iq",
        "models": scorer.get_model_info(),
    }


from app.core.database import safe_db_call

# ── Match Analysis ─────────────────────────────────────────

@app.post("/api/analyze/{match_id}")
async def analyze_match(match_id: int, background_tasks: BackgroundTasks):
    """
    Submit a match for analysis.
    Returns immediately with status. Processing happens in background.
    """
    # Check if already analyzed (with connection retry)
    existing = safe_db_call(db.get_match, match_id)
    if existing and existing.get("processed_at"):
        return {
            "status": "complete",
            "match_id": match_id,
            "message": "Match already analyzed",
        }

    # Check if analysis is in progress
    analysis = safe_db_call(db.get_analysis_status, match_id)
    if analysis in ("pending", "processing"):
        return {
            "status": analysis,
            "match_id": match_id,
            "message": "Analysis in progress",
        }

    # Queue for background processing
    background_tasks.add_task(_process_match_background, match_id)

    return {
        "status": "pending",
        "match_id": match_id,
        "message": "Match queued for analysis",
    }


async def _process_match_background(match_id: int):
    """Background task to process a match. Resets DB client on connection errors."""
    from app.core.database import _reset_client
    try:
        result = await processor.process_match(match_id)
        logger.info(f"Match {match_id} processed: {result}")
    except Exception as e:
        error_msg = str(e)
        if "RemoteProtocolError" in error_msg or "ConnectionTerminated" in error_msg:
            logger.warning(f"Connection error for match {match_id}, resetting client and retrying...")
            _reset_client()
            try:
                result = await processor.process_match(match_id)
                logger.info(f"Match {match_id} processed on retry: {result}")
            except Exception as retry_err:
                logger.error(f"Failed to process match {match_id} on retry: {retry_err}")
        else:
            logger.error(f"Failed to process match {match_id}: {e}")


@app.get("/api/analyze/{match_id}/status")
async def analysis_status(match_id: int):
    """Check processing status for a match."""
    existing = db.get_match(match_id)
    if existing and existing.get("processed_at"):
        return {"status": "complete", "match_id": match_id}

    analysis = db.get_analysis_status(match_id)
    if analysis:
        return {"status": analysis, "match_id": match_id}

    return {"status": "not_found", "match_id": match_id}


# ── Match Data ─────────────────────────────────────────────

@app.get("/api/matches/{match_id}/overview")
async def match_overview(match_id: int):
    """
    Get match-level stats with 7k+ comparisons.
    Phase 1: returns raw stats. Phase 2: adds benchmark comparisons.
    """
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(404, "Match not found. Submit it for analysis first.")

    # Get player records
    sb = db.get_supabase()
    players = (
        sb.table("match_players")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    ).data

    return {
        "match": match,
        "players": players,
    }


# ── Teamfight Data ─────────────────────────────────────────

@app.get("/api/fights/{match_id}")
async def get_fights(match_id: int):
    """Get all teamfights for a match with summary scores."""
    fights = db.get_teamfights_for_match(match_id)
    if not fights:
        raise HTTPException(404, "No teamfight data. Match may not be parsed or analyzed.")
    return {"match_id": match_id, "teamfights": fights}


@app.get("/api/fights/{match_id}/{fight_index}")
async def get_fight_detail(match_id: int, fight_index: int):
    """
    Fight deep-dive: player stats, benchmarks, recommendations, scores.
    Phase 2: full ML-powered analysis.
    """
    from app.ml.scoring import get_scorer

    sb = db.get_supabase()
    fight = (
        sb.table("teamfights")
        .select("*, fight_player_stats(*)")
        .eq("match_id", match_id)
        .eq("fight_index", fight_index)
        .execute()
    ).data

    if not fight:
        raise HTTPException(404, f"Fight {fight_index} not found in match {match_id}")

    fight_data = fight[0]
    scorer = get_scorer()

    # Get match + player data for scoring context
    match = db.get_match(match_id)
    players = db.get_match_players(match_id)

    # Score each player in the fight
    scored_players = []
    for stat in fight_data.get("fight_player_stats", []):
        if scorer.is_loaded and scorer.fight_iq:
            try:
                analysis = scorer.score_player_fight(stat, fight_data, match, players)
                scored_players.append({**stat, "analysis": analysis})
            except Exception as e:
                logger.warning(f"Scoring failed for player {stat.get('hero_id')}: {e}")
                scored_players.append(stat)
        else:
            scored_players.append(stat)

    # Fight outcome prediction
    outcome = None
    if scorer.is_loaded and scorer.fight_outcome:
        outcome = scorer.predict_fight_outcome(
            fight_data, fight_data.get("fight_player_stats", []), match, players
        )

    return {
        "match_id": match_id,
        "fight_index": fight_index,
        "fight": fight_data,
        "player_scores": scored_players,
        "outcome_prediction": outcome,
    }


# ── Minimap Position Data ─────────────────────────────────

@app.get("/api/fights/{match_id}/{fight_index}/minimap")
async def get_fight_minimap(match_id: int, fight_index: int):
    """
    Get position data for minimap replay of a specific fight.
    Returns hero positions for the fight time window.
    """
    # Get fight time boundaries
    sb = db.get_supabase()
    fight = (
        sb.table("teamfights")
        .select("start_time, end_time")
        .eq("match_id", match_id)
        .eq("fight_index", fight_index)
        .execute()
    ).data

    if not fight:
        raise HTTPException(404, f"Fight {fight_index} not found")

    start = fight[0]["start_time"] - 10  # 10 seconds before fight
    end = fight[0]["end_time"] + 5       # 5 seconds after

    positions = (
        sb.table("player_positions")
        .select("*")
        .eq("match_id", match_id)
        .gte("time", start)
        .lte("time", end)
        .order("time")
        .execute()
    ).data

    return {
        "match_id": match_id,
        "fight_index": fight_index,
        "time_range": {"start": start, "end": end},
        "positions": positions,
    }


# ── Similar Fights (pgvector) ─────────────────────────────

@app.get("/api/fights/{match_id}/{fight_index}/similar")
async def get_similar_fights(
    match_id: int,
    fight_index: int,
    hero_id: int | None = None,
    limit: int = 5,
):
    """
    Find similar 7k+ fights using pgvector cosine similarity.
    Optionally filter by hero_id for same-hero comparisons.
    """
    sb = db.get_supabase()

    # Get the source fight's vector
    fight = (
        sb.table("teamfights")
        .select("id")
        .eq("match_id", match_id)
        .eq("fight_index", fight_index)
        .execute()
    ).data
    if not fight:
        raise HTTPException(404, f"Fight {fight_index} not found")

    teamfight_id = fight[0]["id"]

    # Get the vector for this fight + hero
    query = sb.table("fight_vectors").select("embedding").eq("teamfight_id", teamfight_id)
    if hero_id:
        query = query.eq("hero_id", hero_id)
    source_vec = query.limit(1).execute().data

    if not source_vec:
        return {"similar_fights": [], "message": "No vector found for this fight. Run training pipeline first."}

    embedding = source_vec[0]["embedding"]

    # Use Supabase RPC for pgvector similarity search
    # This requires a SQL function — see migration below
    try:
        result = sb.rpc("find_similar_fights", {
            "query_embedding": embedding,
            "match_hero_id": hero_id,
            "exclude_match_id": match_id,
            "result_limit": limit,
        }).execute()
        return {"similar_fights": result.data or []}
    except Exception as e:
        logger.warning(f"Similar fight search failed: {e}")
        return {
            "similar_fights": [],
            "message": "pgvector search not available. Run the similarity search migration.",
        }


# ── Hero Benchmarks (public) ──────────────────────────────

@app.get("/api/heroes/{hero_id}/benchmarks")
async def hero_benchmarks(hero_id: int):
    """Get all benchmark data for a hero."""
    sb = db.get_supabase()
    benchmarks = (
        sb.table("hero_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .execute()
    ).data
    return {"hero_id": hero_id, "benchmarks": benchmarks}


# ── ML Model Management ──────────────────────────────────

@app.get("/api/models/info")
async def model_info():
    """Check loaded model versions and metrics."""
    from app.ml.scoring import get_scorer
    return get_scorer().get_model_info()


@app.post("/api/models/reload")
async def reload_models():
    """Reload models after retraining."""
    from app.ml.scoring import reload_models as _reload
    scorer = _reload()
    return {"status": "reloaded", "models": scorer.get_model_info()}


# ── Data Collection Status (admin) ────────────────────────

@app.get("/api/admin/collection-status")
async def collection_status():
    """Check data collection pipeline status."""
    sb = db.get_supabase()

    total_matches = sb.table("matches").select("match_id", count="exact").execute()
    total_fights = sb.table("teamfights").select("id", count="exact").execute()
    pool_pending = (
        sb.table("match_collection_pool")
        .select("match_id", count="exact")
        .eq("status", "pending")
        .execute()
    )

    return {
        "matches_stored": total_matches.count,
        "teamfights_stored": total_fights.count,
        "pool_pending": pool_pending.count,
    }