"""
OpenDota API Client

Handles: match details (with teamfight data), parse requests, explorer SQL, player data.
Rate limit: 60 calls/min free tier, 50k calls/month.
Docs: https://docs.opendota.com
"""

import httpx
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.opendota.com/api"


class OpenDotaClient:

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.opendota_api_key
        self._semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._client

    def _params(self, extra: dict | None = None) -> dict:
        """Add API key to params if available."""
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if extra:
            params.update(extra)
        return params

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _get(self, path: str, params: dict | None = None) -> dict | list | None:
        async with self._semaphore:
            client = await self._get_client()
            response = await client.get(path, params=self._params(params))

            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                await asyncio.sleep(retry_after)
                response = await client.get(path, params=self._params(params))

            response.raise_for_status()
            return response.json()

    # ── Match Data ─────────────────────────────────────────────

    async def get_match(self, match_id: int) -> dict | None:
        """
        Fetch full match details including parsed data.

        Returns teamfights[], players with kill_log, purchase_log,
        damage_inflictor, gold_t, xp_t, chat, objectives, etc.
        Returns None if match not found.
        """
        try:
            return await self._get(f"/matches/{match_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def is_parsed(self, match_data: dict) -> bool:
        """Check if a match has been parsed (has detailed replay data)."""
        # Parsed matches have teamfights and gold_t arrays
        if not match_data:
            return False
        players = match_data.get("players", [])
        if not players:
            return False
        return players[0].get("gold_t") is not None

    async def request_parse(self, match_id: int) -> dict:
        """
        Request a match replay to be parsed.
        Returns job info. Parsing may take minutes.
        """
        client = await self._get_client()
        response = await client.post(
            f"/request/{match_id}",
            params=self._params(),
        )
        response.raise_for_status()
        return response.json()

    # ── Player Data ────────────────────────────────────────────

    async def get_player(self, account_id: int) -> dict | None:
        """Get player profile (name, avatar, rank, MMR)."""
        try:
            return await self._get(f"/players/{account_id}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    async def get_player_matches(
        self,
        account_id: int,
        limit: int = 100,
        hero_id: int | None = None,
        lobby_type: int | None = None,
    ) -> list[dict]:
        """Get recent matches for a player."""
        params = {"limit": limit}
        if hero_id is not None:
            params["hero_id"] = hero_id
        if lobby_type is not None:
            params["lobby_type"] = lobby_type
        return await self._get(f"/players/{account_id}/matches", params) or []

    # ── Pro / High MMR Data ────────────────────────────────────

    async def get_pro_players(self) -> list[dict]:
        """Get list of pro players with account IDs."""
        return await self._get("/proPlayers") or []

    async def get_pro_matches(self, less_than_match_id: int | None = None) -> list[dict]:
        """Get recent pro matches."""
        params = {}
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return await self._get("/proMatches", params) or []

    # ── Explorer (SQL) ─────────────────────────────────────────

    async def explorer_query(self, sql: str) -> dict:
        """
        Run raw SQL against OpenDota's Postgres.
        Useful for bulk aggregations without fetching individual matches.

        Example:
            SELECT hero_id, AVG(kills), AVG(deaths)
            FROM player_matches
            WHERE avg_rank_tier >= 70
            GROUP BY hero_id
        """
        return await self._get("/explorer", {"sql": sql})

    # ── Public Matches ─────────────────────────────────────────

    async def get_public_matches(
        self, mmr_ascending: int | None = None, less_than_match_id: int | None = None
    ) -> list[dict]:
        """
        Get randomly sampled recent public matches.
        mmr_ascending: minimum average MMR
        """
        params = {}
        if mmr_ascending:
            params["mmr_ascending"] = mmr_ascending
        if less_than_match_id:
            params["less_than_match_id"] = less_than_match_id
        return await self._get("/publicMatches", params) or []

    # ── Hero Data ──────────────────────────────────────────────

    async def get_hero_benchmarks(self, hero_id: int) -> dict:
        """Get percentile benchmarks for a hero (GPM, XPM, etc.)."""
        return await self._get("/benchmarks", {"hero_id": hero_id})

    async def get_heroes(self) -> list[dict]:
        """Get list of all heroes with stats."""
        return await self._get("/heroes") or []
    
    # ── Parser Worker ────────────────────────────────────────────────
    async def get_replay_url(self, match_id: int) -> str | None:
        data = await self._get(f"/matches/{match_id}")
        if not data:
            return None
        cluster = data.get("cluster")
        salt = data.get("replay_salt")
        if not cluster or not salt:
            return None
        return f"http://replay{cluster}.valve.net/570/{match_id}_{salt}.dem.bz2"

    # ── Cleanup ────────────────────────────────────────────────

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
