"""
Match Processing Service

Core pipeline: match_id → fetch APIs → store raw in S3 → extract fights → write to Postgres.
"""

import logging
from datetime import datetime, timezone

from app.clients.opendota import OpenDotaClient
from app.clients.stratz import StratzClient
from app.core.storage import get_storage
from app.core import database as db

logger = logging.getLogger(__name__)


class MatchProcessor:

    def __init__(self):
        self.opendota = OpenDotaClient()
        self.stratz = StratzClient()
        self.storage = get_storage()

    async def process_match(self, match_id: int) -> dict:
        """
        Full pipeline for a single match:
        1. Fetch from OpenDota (fight data)
        2. Fetch from STRATZ (position data)
        3. Store raw JSONs in S3
        4. Extract and store match record
        5. Extract and store player records
        6. Extract and store teamfight records
        7. Extract and store position data
        """
        logger.info(f"Processing match {match_id}")

        # Step 1: Fetch from OpenDota
        od_data = await self.opendota.get_match(match_id)
        if not od_data:
            raise ValueError(f"Match {match_id} not found on OpenDota")

        # Check if parsed
        if not self.opendota.is_parsed(od_data):
            logger.info(f"Match {match_id} not parsed. Requesting parse.")
            await self.opendota.request_parse(match_id)
            return {"status": "parse_requested", "match_id": match_id}

        # Step 2: Fetch from STRATZ (position data — optional, don't fail if unavailable)
        stratz_data = None
        try:
            stratz_data = await self.stratz.get_match_playback(match_id)
        except Exception as e:
            logger.warning(f"STRATZ fetch failed for match {match_id} (continuing without position data): {e}")

        # Step 3: Store raw data
        storage_key_od = self.storage.store_raw_match(match_id, od_data, source="opendota")
        if stratz_data:
            self.storage.store_raw_match(match_id, stratz_data, source="stratz")

        # Step 3.5: Clean up any existing data for this match (safe reprocessing)
        db.delete_match_data(match_id)

        # Step 4: Extract and store match record
        match_record = self._extract_match(od_data, storage_key_od)
        db.upsert_match(match_record)

        # Step 5: Extract and store player records
        players = self._extract_players(od_data)
        if players:
            db.upsert_match_players(players)

        # Step 6: Extract and store teamfights
        fight_count = 0
        teamfights_raw = od_data.get("teamfights", [])
        if teamfights_raw:
            teamfights, fight_stats = self._extract_teamfights(od_data, teamfights_raw)
            if teamfights:
                result = db.insert_teamfights(teamfights)
                # Get inserted IDs to link fight_player_stats
                inserted_fights = result.data
                for i, fight in enumerate(inserted_fights):
                    for stat in fight_stats[i]:
                        stat["teamfight_id"] = fight["id"]
                all_stats = [s for stats in fight_stats for s in stats]
                if all_stats:
                    db.insert_fight_player_stats(all_stats)
                fight_count = len(teamfights)

        # Step 7: Extract and store position data
        position_count = 0
        if stratz_data and stratz_data.get("players"):
            positions = self._extract_positions(match_id, stratz_data)
            if positions:
                db.insert_player_positions(positions)
                position_count = len(positions)

        # Step 8: Extract and store ward events
        ward_count = 0
        try:
            wards = self._extract_wards(od_data)
            if wards:
                from app.core.database import get_supabase
                get_supabase().table("ward_events").insert(wards).execute()
                ward_count = len(wards)
        except Exception as e:
            logger.warning(f"Ward extraction/insert failed for match {match_id}: {e}")

        logger.info(
            f"Match {match_id} processed: {fight_count} fights, "
            f"{position_count} position events, {ward_count} wards"
        )

        return {
            "status": "processed",
            "match_id": match_id,
            "fights": fight_count,
            "positions": position_count,
            "wards": ward_count,
        }

    # ── Extraction Methods ─────────────────────────────────

    def _extract_match(self, od_data: dict, storage_key: str) -> dict:
        """Extract match-level record from OpenDota response."""
        return {
            "match_id": od_data["match_id"],
            "duration": od_data.get("duration"),
            "start_time": od_data.get("start_time"),
            "game_mode": od_data.get("game_mode"),
            "lobby_type": od_data.get("lobby_type"),
            "radiant_win": od_data.get("radiant_win"),
            "patch": str(od_data.get("patch", "")),
            "avg_rank_tier": od_data.get("avg_rank_tier"),
            "radiant_score": od_data.get("radiant_score"),
            "dire_score": od_data.get("dire_score"),
            "first_blood_time": od_data.get("first_blood_time"),
            "is_parsed": True,
            "s3_key": storage_key,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _extract_players(self, od_data: dict) -> list[dict]:
        """Extract per-player records from OpenDota response."""
        players = []
        for p in od_data.get("players", []):
            players.append({
                "match_id": od_data["match_id"],
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "player_slot": p.get("player_slot"),
                "kills": p.get("kills"),
                "deaths": p.get("deaths"),
                "assists": p.get("assists"),
                "gpm": p.get("gold_per_min"),
                "xpm": p.get("xp_per_min"),
                "last_hits": p.get("last_hits"),
                "denies": p.get("denies"),
                "hero_damage": p.get("hero_damage"),
                "tower_damage": p.get("tower_damage"),
                "hero_healing": p.get("hero_healing"),
                "net_worth": p.get("net_worth"),
                "level": p.get("level"),
                "lane": p.get("lane"),
                "is_radiant": p.get("player_slot", 0) < 128,
                "items": {
                    f"item_{i}": p.get(f"item_{i}")
                    for i in range(6)
                },
                "purchase_log": p.get("purchase_log"),
                "gold_t": p.get("gold_t"),
                "xp_t": p.get("xp_t"),
            })
        return players

    def _extract_teamfights(
        self, od_data: dict, teamfights_raw: list[dict]
    ) -> tuple[list[dict], list[list[dict]]]:
        """
        Extract teamfight records and per-player fight stats.
        Returns (teamfight_records, list_of_player_stats_per_fight).
        """
        match_id = od_data["match_id"]
        players = od_data.get("players", [])

        teamfight_records = []
        all_fight_stats = []

        for idx, fight in enumerate(teamfights_raw):
            fight_players = fight.get("players", [])

            # Calculate kills per side
            radiant_kills = 0
            dire_kills = 0
            gold_swing = 0

            fight_stats = []
            for player_idx, fp in enumerate(fight_players):
                is_radiant = player_idx < 5
                killed = fp.get("killed", {})
                kills_count = sum(killed.values()) if isinstance(killed, dict) else 0

                if is_radiant:
                    radiant_kills += kills_count
                else:
                    dire_kills += kills_count

                gold_swing += fp.get("gold_delta", 0) if is_radiant else -fp.get("gold_delta", 0)

                # Get hero_id and account_id from match players
                hero_id = players[player_idx].get("hero_id") if player_idx < len(players) else None
                account_id = players[player_idx].get("account_id") if player_idx < len(players) else None

                fight_stats.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "ability_uses": fp.get("ability_uses", {}),
                    "item_uses": fp.get("item_uses", {}),
                    "damage": fp.get("damage", 0),
                    "healing": fp.get("healing", 0),
                    "gold_delta": fp.get("gold_delta", 0),
                    "xp_delta": fp.get("xp_delta", 0),
                    "deaths": fp.get("deaths", 0),
                    "killed": killed,
                    "buybacks": fp.get("buybacks", 0),
                    "xp_start": fp.get("xp_start"),
                    "xp_end": fp.get("xp_end"),
                    "is_radiant": is_radiant,
                    # items_at_fight will be enriched from purchase_log later
                })

            teamfight_records.append({
                "match_id": match_id,
                "fight_index": idx,
                "start_time": fight.get("start", 0),
                "end_time": fight.get("end", 0),
                "deaths_count": fight.get("deaths", 0),
                "radiant_kills": radiant_kills,
                "dire_kills": dire_kills,
                "gold_swing": gold_swing,
            })
            all_fight_stats.append(fight_stats)

        return teamfight_records, all_fight_stats

    def _extract_positions(self, match_id: int, stratz_data: dict) -> list[dict]:
        """Extract position events from STRATZ per-player playbackData."""
        positions = []
        players = stratz_data.get("players", [])

        for player in players:
            playback = player.get("playbackData")
            if not playback:
                continue

            account_id = player.get("steamAccountId")
            hero_id = player.get("heroId")

            pos_events = playback.get("playerUpdatePositionEvents", [])
            if not pos_events:
                continue

            for event in pos_events:
                positions.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "time": event.get("time", 0),
                    "x": event.get("x", 0),
                    "y": event.get("y", 0),
                })

        return positions

    def _extract_wards(self, od_data: dict) -> list[dict]:
        """Extract ward placement events from OpenDota data."""
        wards = []
        match_id = od_data["match_id"]

        for player in od_data.get("players", []):
            account_id = player.get("account_id")
            hero_id = player.get("hero_id")

            for ward in player.get("obs_log", []) or []:
                wards.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "time": ward.get("time", 0),
                    "x": int(float(ward.get("x", 0))),
                    "y": int(float(ward.get("y", 0))),
                    "ward_type": "observer",
                })

            for ward in player.get("sen_log", []) or []:
                wards.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "time": ward.get("time", 0),
                    "x": int(float(ward.get("x", 0))),
                    "y": int(float(ward.get("y", 0))),
                    "ward_type": "sentry",
                })

        return wards

    async def close(self):
        await self.opendota.close()
        await self.stratz.close()