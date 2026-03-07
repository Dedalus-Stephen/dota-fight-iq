"""
Supabase Client

Handles all database operations via Supabase Python SDK.
Forces HTTP/1.1 to avoid HTTP/2 ConnectionTerminated errors with Supabase.
"""

import logging
import httpx

# ── Force HTTP/1.1 globally ───────────────────────────────
_original_httpx_client_init = httpx.Client.__init__

def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs["http2"] = False
    _original_httpx_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_client_init
# ──────────────────────────────────────────────────────────

from supabase import create_client, Client
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def _create_fresh_client() -> Client:
    """Create a new Supabase client instance."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_service_key)


def get_supabase() -> Client:
    global _client
    if _client is None:
        _client = _create_fresh_client()
    return _client


def _reset_client():
    """Reset the client when connection errors occur."""
    global _client
    logger.warning("Resetting Supabase client due to connection error")
    _client = None


def safe_db_call(fn, *args, **kwargs):
    """Execute a DB op with automatic retry on connection errors."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        error_msg = str(e)
        if any(
            k in error_msg
            for k in (
                "RemoteProtocolError",
                "ConnectionTerminated",
                "ConnectionReset",
            )
        ):
            _reset_client()
            try:
                return fn(*args, **kwargs)
            except Exception:
                _reset_client()
                raise
        raise


# ── Match Operations ───────────────────────────────────────

def delete_match_data(match_id: int):
    """
    Delete all existing data for a match before reprocessing.
    Order: children before parents (foreign key constraints).
    """
    sb = get_supabase()

    # v2 tables (no FK to teamfights, safe to delete in any order)
    for table in (
        "chat_analysis",
        "match_player_scores",
        "farming_analysis",
        "itemization_analysis",
        "laning_analysis",
        "match_objectives",
        "fight_context",
    ):
        try:
            sb.table(table).delete().eq("match_id", match_id).execute()
        except Exception as e:
            # Table may not exist yet if migration hasn't run
            logger.debug(f"Skipping cleanup for {table}: {e}")

    # v1 tables
    sb.table("fight_player_stats").delete().eq("match_id", match_id).execute()
    sb.table("teamfights").delete().eq("match_id", match_id).execute()
    sb.table("player_positions").delete().eq("match_id", match_id).execute()
    sb.table("ward_events").delete().eq("match_id", match_id).execute()
    sb.table("fight_scores").delete().eq("match_id", match_id).execute()
    sb.table("death_analysis").delete().eq("match_id", match_id).execute()

    logger.debug(f"Cleared existing data for match {match_id}")


def upsert_match(match_data: dict) -> dict:
    """Insert or update a match record."""
    sb = get_supabase()
    return (
        sb.table("matches")
        .upsert(match_data, on_conflict="match_id")
        .execute()
    )


def get_match(match_id: int) -> dict | None:
    """Get a match by ID."""
    sb = get_supabase()
    result = sb.table("matches").select("*").eq("match_id", match_id).execute()
    return result.data[0] if result.data else None


def upsert_match_players(players: list[dict]) -> dict:
    """Insert or update match player records."""
    sb = get_supabase()
    return (
        sb.table("match_players")
        .upsert(players, on_conflict="match_id,account_id")
        .execute()
    )


def get_match_players(match_id: int) -> list[dict]:
    """Get all player records for a match."""
    sb = get_supabase()
    result = (
        sb.table("match_players")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Teamfight Operations ──────────────────────────────────

def insert_teamfights(teamfights: list[dict]) -> dict:
    """Insert teamfight records for a match."""
    sb = get_supabase()
    return sb.table("teamfights").insert(teamfights).execute()


def insert_fight_player_stats(stats: list[dict]) -> dict:
    """Insert per-player fight statistics."""
    sb = get_supabase()
    return sb.table("fight_player_stats").insert(stats).execute()


def get_teamfights_for_match(match_id: int) -> list[dict]:
    """Get all teamfights for a match with player stats."""
    sb = get_supabase()
    result = (
        sb.table("teamfights")
        .select("*, fight_player_stats(*)")
        .eq("match_id", match_id)
        .order("start_time")
        .execute()
    )
    return result.data


# ── Fight Context Operations ──────────────────────────────

def get_fight_context(teamfight_id: int) -> dict | None:
    """Get team state snapshot for a specific fight."""
    sb = get_supabase()
    result = (
        sb.table("fight_context")
        .select("*")
        .eq("teamfight_id", teamfight_id)
        .execute()
    )
    return result.data[0] if result.data else None


def get_fight_contexts_for_match(match_id: int) -> list[dict]:
    """Get all fight contexts for a match."""
    sb = get_supabase()
    result = (
        sb.table("fight_context")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Objectives Operations ─────────────────────────────────

def get_objectives_for_match(match_id: int) -> list[dict]:
    """Get all objectives for a match."""
    sb = get_supabase()
    result = (
        sb.table("match_objectives")
        .select("*")
        .eq("match_id", match_id)
        .order("time")
        .execute()
    )
    return result.data or []


# ── Laning Operations ─────────────────────────────────────

def get_laning_for_match(match_id: int) -> list[dict]:
    """Get laning analysis for all players in a match."""
    sb = get_supabase()
    result = (
        sb.table("laning_analysis")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Itemization Operations ────────────────────────────────

def get_itemization_for_match(match_id: int) -> list[dict]:
    """Get itemization data for all players in a match."""
    sb = get_supabase()
    result = (
        sb.table("itemization_analysis")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Farming Operations ────────────────────────────────────

def get_farming_for_match(match_id: int) -> list[dict]:
    """Get farming analysis for all players in a match."""
    sb = get_supabase()
    result = (
        sb.table("farming_analysis")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Chat / Toxicity Operations ────────────────────────────

def get_chat_for_match(match_id: int) -> list[dict]:
    """Get chat analysis for all players in a match."""
    sb = get_supabase()
    result = (
        sb.table("chat_analysis")
        .select("*")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data or []


# ── Match Player Scores Operations ────────────────────────

def get_match_scores(match_id: int) -> list[dict]:
    """Get overall player scores/rankings for a match."""
    sb = get_supabase()
    result = (
        sb.table("match_player_scores")
        .select("*")
        .eq("match_id", match_id)
        .order("match_rank")
        .execute()
    )
    return result.data or []


def upsert_match_scores(scores: list[dict]) -> dict:
    """Insert or update match player scores."""
    sb = get_supabase()
    return (
        sb.table("match_player_scores")
        .upsert(scores, on_conflict="match_id,player_slot")
        .execute()
    )


# ── Ward Operations ───────────────────────────────────────

def get_wards_for_match(match_id: int) -> list[dict]:
    """Get all ward events for a match."""
    sb = get_supabase()
    result = (
        sb.table("ward_events")
        .select("*")
        .eq("match_id", match_id)
        .order("time")
        .execute()
    )
    return result.data or []


# ── Benchmark Operations ──────────────────────────────────

def get_hero_benchmark(
    hero_id: int,
    time_bucket: str,
    nw_bucket: str,
    metric_name: str,
) -> dict | None:
    """Get benchmark percentiles for a hero in a specific context."""
    sb = get_supabase()
    result = (
        sb.table("hero_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .eq("time_bucket", time_bucket)
        .eq("nw_bucket", nw_bucket)
        .eq("metric_name", metric_name)
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_benchmarks(benchmarks: list[dict]) -> dict:
    """Bulk upsert benchmark records."""
    sb = get_supabase()
    return (
        sb.table("hero_benchmarks")
        .upsert(
            benchmarks,
            on_conflict="hero_id,time_bucket,nw_bucket,duration_bucket,size_bucket,metric_name",
        )
        .execute()
    )


def get_item_timing_benchmarks(hero_id: int) -> list[dict]:
    """Get item timing benchmarks for a hero."""
    sb = get_supabase()
    result = (
        sb.table("item_timing_benchmarks")
        .select("*")
        .eq("hero_id", hero_id)
        .execute()
    )
    return result.data or []


def get_objective_benchmarks() -> list[dict]:
    """Get all objective timing benchmarks."""
    sb = get_supabase()
    result = (
        sb.table("objective_benchmarks")
        .select("*")
        .execute()
    )
    return result.data or []


# ── High MMR Match Pool ──────────────────────────────────

def insert_match_pool(match_ids: list[dict]) -> dict:
    """Insert discovered high-MMR match IDs for processing."""
    sb = get_supabase()
    return (
        sb.table("match_collection_pool")
        .upsert(match_ids, on_conflict="match_id")
        .execute()
    )


def get_unprocessed_matches(limit: int = 50) -> list[dict]:
    """Get match IDs that haven't been fetched/processed yet."""
    sb = get_supabase()
    result = (
        sb.table("match_collection_pool")
        .select("*")
        .eq("status", "pending")
        .limit(limit)
        .execute()
    )
    return result.data


def update_match_pool_status(match_id: int, status: str) -> dict:
    """Update collection status."""
    sb = get_supabase()
    return (
        sb.table("match_collection_pool")
        .update({"status": status})
        .eq("match_id", match_id)
        .execute()
    )


# ── Position Data ─────────────────────────────────────────

def insert_player_positions(positions: list[dict]) -> dict:
    """Bulk insert position data from STRATZ playback."""
    sb = get_supabase()
    for i in range(0, len(positions), 1000):
        chunk = positions[i : i + 1000]
        sb.table("player_positions").insert(chunk).execute()
    return {"inserted": len(positions)}


# ── User / Analysis ──────────────────────────────────────

def get_or_create_user(steam_id: str, profile: dict) -> dict:
    """Get or create a user from Steam login."""
    sb = get_supabase()
    result = sb.table("users").select("*").eq("steam_id", steam_id).execute()
    if result.data:
        return result.data[0]
    user_data = {
        "steam_id": steam_id,
        "persona_name": profile.get("personaname", ""),
        "avatar_url": profile.get("avatarfull", ""),
    }
    return sb.table("users").insert(user_data).execute().data[0]


def get_analysis_status(match_id: int) -> str | None:
    """Check if a match has been analyzed (and status)."""
    sb = get_supabase()
    result = (
        sb.table("match_analyses")
        .select("status")
        .eq("match_id", match_id)
        .execute()
    )
    return result.data[0]["status"] if result.data else None


def create_analysis(match_id: int, user_id: str) -> dict:
    """Create an analysis request."""
    sb = get_supabase()
    return (
        sb.table("match_analyses")
        .insert({"match_id": match_id, "user_id": user_id, "status": "pending"})
        .execute()
    )


def update_analysis_status(match_id: int, status: str) -> dict:
    """Update analysis status."""
    sb = get_supabase()
    return (
        sb.table("match_analyses")
        .update({"status": status})
        .eq("match_id", match_id)
        .execute()
    )