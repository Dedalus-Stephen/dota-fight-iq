"""
STRATZ API Client (GraphQL)

Handles: match playback data (positions, inventory), bulk match lookup.
Rate limit: 10,000 calls/day free tier.
Docs: https://stratz.com/api
"""

import httpx
import asyncio
import logging
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.stratz.com/graphql"


class StratzClient:

    def __init__(self):
        settings = get_settings()
        self.api_token = settings.stratz_api_token
        self._semaphore = asyncio.Semaphore(5)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "STRATZ_API",
            }
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers=headers,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    )
    async def _query(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        async with self._semaphore:
            client = await self._get_client()
            payload = {"query": query}
            if variables:
                payload["variables"] = variables

            response = await client.post(GRAPHQL_URL, json=payload)

            if response.status_code == 400:
                logger.error(f"STRATZ 400 Error: {response.text}")

            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                logger.error(f"STRATZ GraphQL errors: {result['errors']}")
                raise ValueError(f"GraphQL errors: {result['errors']}")

            return result.get("data", {})

    # ── Match Playback (position data, inventory) ──────────

    async def get_match_playback(self, match_id: int) -> dict | None:
        """
        Fetch match with per-player playback data.
        Position events, gold events, purchases, inventory snapshots.
        """
        query = """
        query GetMatch($matchId: Long!) {
            match(id: $matchId) {
                id
                didRadiantWin
                durationSeconds
                playbackData {
                    buildingEvents { time indexId type isRadiant }
                    roshanEvents { time }
                }
                players {
                    steamAccountId
                    heroId
                    playerSlot
                    isRadiant
                    playbackData {
                        playerUpdatePositionEvents { time x y }
                        playerUpdateGoldEvents { time gold networth }
                        purchaseEvents { time itemId }
                        inventoryEvents {
                            time
                            item0 { itemId }
                            item1 { itemId }
                            item2 { itemId }
                            item3 { itemId }
                            item4 { itemId }
                            item5 { itemId }
                        }
                    }
                    stats {
                        networthPerMinute
                        goldPerMinute
                        experiencePerMinute
                        lastHitsPerMinute
                    }
                }
            }
        }
        """
        result = await self._query(query, {"matchId": int(match_id)})
        return result.get("match")

    # ── Bulk Match Lookup by IDs ───────────────────────────

    async def get_matches_by_ids(self, match_ids: list[int]) -> list[dict]:
        """
        Fetch multiple matches by their IDs.
        STRATZ matches() takes ids: [Long!]! — up to 100 per call.
        """
        query = """
        query GetMatches($ids: [Long]!) {
            matches(ids: $ids) {
                id
                didRadiantWin
                durationSeconds
                startDateTime
                bracket
                averageRank
                players {
                    steamAccountId
                    heroId
                    position
                    kills
                    deaths
                    assists
                    networth
                    goldPerMinute
                    experiencePerMinute
                    isRadiant
                    imp
                    role
                }
            }
        }
        """
        result = await self._query(query, {"ids": match_ids})
        return result.get("matches", [])

    # ── Player Match Discovery (for high-MMR collection) ───

    async def get_player_matches(
    self,
    steam_id: int,
    take: int = 50,
    ) -> list[dict]:
        """
        Get recent matches for a specific player.
        """
        # Change [RANKED_MATCHMAKING] to [7]
        query = """
        query GetPlayerMatches($steamId: Long!, $take: Int!) {
            player(steamAccountId: $steamId) {
                matches(request: { take: $take, lobbyTypeIds: [7] }) {
                    id
                    didRadiantWin
                    durationSeconds
                    players {
                        steamAccountId
                        heroId
                        kills
                        deaths
                        assists
                        networth
                        goldPerMinute
                        experiencePerMinute
                        isRadiant
                        imp
                        role
                    }
                }
            }
        }
        """
        result = await self._query(query, {"steamId": steam_id, "take": take})
        player = result.get("player", {})
        return player.get("matches", []) if player else []

    # ── Hero Stats ─────────────────────────────────────────

    async def get_hero_stats(self, hero_id: int) -> dict | None:
        """Get hero win rate and pick rate from STRATZ."""
        query = """
        query GetHeroStats($heroId: Short!) {
            heroStats {
                stats(heroIds: [$heroId], bracketBasicIds: [IMMORTAL]) {
                    heroId
                    matchCount
                    winCount
                }
            }
        }
        """
        result = await self._query(query, {"heroId": hero_id})
        stats = result.get("heroStats", {}).get("stats", [])
        return stats[0] if stats else None

    # ── Cleanup ────────────────────────────────────────────

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()