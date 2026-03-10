"""
Match Processing Service

Core pipeline: match_id → fetch APIs → store raw in S3 → extract fights → write to Postgres.

v2: Extended extraction for laning, farming, itemization, objectives,
    fight context, support stats, ward details, and chat data.
"""

import logging
from datetime import datetime, timezone

from app.clients.opendota import OpenDotaClient
from app.clients.stratz import StratzClient
from app.core.storage import get_storage
from app.core import database as db

logger = logging.getLogger(__name__)

# Key items to track timing for in itemization analysis
KEY_ITEMS = frozenset({
    "black_king_bar", "radiance", "manta", "butterfly", "skadi",
    "heart", "satanic", "daedalus", "monkey_king_bar", "assault",
    "shivas_guard", "aghanims_shard", "ultimate_scepter",
    "blink", "aether_lens", "force_staff", "hurricane_pike",
    "desolator", "diffusal_blade", "orchid", "bloodthorn",
    "linken_sphere", "travel_boots", "refresher", "octarine_core",
    "ethereal_blade", "pipe", "crimson_guard", "mekansm",
    "guardian_greaves", "spirit_vessel", "glimmer_cape",
    "aghanims_scepter", "aeon_disk", "lotus_orb", "solar_crest",
    "hand_of_midas", "battle_fury", "maelstrom", "mjollnir",
    "gleipnir", "overwhelming_blink", "arcane_blink", "swift_blink",
    "wind_waker", "phylactery", "boots_of_bearing",
})


class MatchProcessor:

    def __init__(self):
        self.opendota = OpenDotaClient()
        self.stratz = StratzClient()
        self.storage = get_storage()

    async def process_match(self, match_id: int) -> dict:
        """
        Full pipeline for a single match:
        1.  Fetch from OpenDota (fight data)
        2.  Fetch from STRATZ (position data)
        3.  Store raw JSONs in S3
        3b. Clean up existing data for safe reprocessing
        4.  Extract and store match record
        5.  Extract and store player records (with support stats)
        6.  Extract and store teamfights + fight context
        7.  Extract and store position data
        8.  Extract and store ward events (with lifespan)
        9.  Extract and store objectives
        10. Extract and store laning analysis
        11. Extract and store itemization analysis
        12. Extract and store farming analysis
        13. Extract and store chat data
        """
        logger.info(f"Processing match {match_id}")

        # Step 1: Fetch from OpenDota
        od_data = await self.opendota.get_match(match_id)
        if not od_data:
            raise ValueError(f"Match {match_id} not found on OpenDota")

        # Check if parsed
        if not self.opendota.is_parsed(od_data):
            logger.info(f"Match {match_id} not parsed on OpenDota. Attempting local parse.")
            return await self.request_local_parse(match_id)

        # Step 2: Fetch from STRATZ (position data — optional)
        stratz_data = None
        try:
            stratz_data = await self.stratz.get_match_playback(match_id)
        except Exception as e:
            logger.warning(
                f"STRATZ fetch failed for match {match_id} "
                f"(continuing without position data): {e}"
            )

        # Step 3: Store raw data
        storage_key_od = self.storage.store_raw_match(
            match_id, od_data, source="opendota"
        )
        if stratz_data:
            self.storage.store_raw_match(match_id, stratz_data, source="stratz")

        # Step 3b: Clean up existing data
        db.delete_match_data(match_id)

        # Step 4: Extract and store match record
        match_record = self._extract_match(od_data, storage_key_od)
        db.upsert_match(match_record)

        # Step 5: Extract and store player records (extended with support stats)
        players = self._extract_players(od_data)
        if players:
            db.upsert_match_players(players)

        # Step 6: Teamfights + fight context
        fight_count = 0
        teamfights_raw = od_data.get("teamfights", [])
        if teamfights_raw:
            teamfights, fight_stats = self._extract_teamfights(
                od_data, teamfights_raw
            )
            if teamfights:
                result = db.insert_teamfights(teamfights)
                inserted_fights = result.data
                for i, fight in enumerate(inserted_fights):
                    for stat in fight_stats[i]:
                        stat["teamfight_id"] = fight["id"]
                all_stats = [s for stats in fight_stats for s in stats]
                if all_stats:
                    db.insert_fight_player_stats(all_stats)
                fight_count = len(teamfights)

                # Step 6b: Fight context for each teamfight
                self._store_fight_contexts(
                    od_data, teamfights, inserted_fights
                )

        # Step 7: Position data
        position_count = 0
        if stratz_data and stratz_data.get("players"):
            positions = self._extract_positions(match_id, stratz_data)
            if positions:
                db.insert_player_positions(positions)
                position_count = len(positions)

        # Step 8: Ward events (enhanced with lifespan)
        ward_count = 0
        try:
            wards = self._extract_ward_details(od_data)
            if wards:
                from app.core.database import get_supabase
                get_supabase().table("ward_events").insert(wards).execute()
                ward_count = len(wards)
        except Exception as e:
            logger.warning(
                f"Ward extraction failed for match {match_id}: {e}"
            )

        # Step 9: Objectives
        objective_count = 0
        try:
            objectives = self._extract_objectives(od_data)
            if objectives:
                from app.core.database import get_supabase
                get_supabase().table("match_objectives").insert(
                    objectives
                ).execute()
                objective_count = len(objectives)
        except Exception as e:
            logger.warning(
                f"Objective extraction failed for match {match_id}: {e}"
            )

        # Step 10: Laning analysis
        laning_count = 0
        try:
            laning = self._extract_laning(od_data)
            if laning:
                from app.core.database import get_supabase
                get_supabase().table("laning_analysis").insert(
                    laning
                ).execute()
                laning_count = len(laning)
        except Exception as e:
            logger.warning(
                f"Laning extraction failed for match {match_id}: {e}"
            )

        # Step 11: Itemization
        item_count = 0
        try:
            items = self._extract_itemization(od_data)
            if items:
                from app.core.database import get_supabase
                get_supabase().table("itemization_analysis").insert(
                    items
                ).execute()
                item_count = len(items)
        except Exception as e:
            logger.warning(
                f"Itemization extraction failed for match {match_id}: {e}"
            )

        # Step 12: Farming analysis
        farming_count = 0
        try:
            farming = self._extract_farming(od_data)
            if farming:
                from app.core.database import get_supabase
                get_supabase().table("farming_analysis").insert(
                    farming
                ).execute()
                farming_count = len(farming)
        except Exception as e:
            logger.warning(
                f"Farming extraction failed for match {match_id}: {e}"
            )

        # Step 13: Chat data
        chat_count = 0
        try:
            chat = self._extract_chat(od_data)
            if chat:
                from app.core.database import get_supabase
                get_supabase().table("chat_analysis").insert(chat).execute()
                chat_count = len(chat)
        except Exception as e:
            logger.warning(
                f"Chat extraction failed for match {match_id}: {e}"
            )
        
        build_count = {"item_snapshots": 0, "ability_builds": 0, "context_vectors": 0}
        try:
            from app.services.build_extractor import process_match_builds
            from app.core.database import get_supabase
            build_count = process_match_builds(get_supabase(), match_id, od_data)
        except Exception as e:
            logger.warning(f"Build extraction failed for match {match_id}: {e}")
            
        logger.info(
            f"Match {match_id} processed: {fight_count} fights, "
            f"{position_count} positions, {ward_count} wards, "
            f"{objective_count} objectives, {laning_count} laning, "
            f"{item_count} items, {farming_count} farming, "
            f"{chat_count} chat records, "
            f"{build_count.get('item_snapshots', 0)} item snapshots, "
            f"{build_count.get('ability_builds', 0)} ability builds"
        )

        return {
            "status": "processed",
            "match_id": match_id,
            "fights": fight_count,
            "positions": position_count,
            "wards": ward_count,
            "objectives": objective_count,
            "laning": laning_count,
            "items": item_count,
            "farming": farming_count,
            "chat": chat_count,
            "build_count": build_count
        }

    # ═══════════════════════════════════════════════════════
    # Extraction Methods
    # ═══════════════════════════════════════════════════════

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
        """Extract per-player records, including support-specific stats."""
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
                # Support-specific stats (Feature 5)
                "obs_placed": p.get("obs_placed", 0),
                "sen_placed": p.get("sen_placed", 0),
                "camps_stacked": p.get("camps_stacked", 0),
                "creeps_stacked": p.get("creeps_stacked", 0),
                "teamfight_participation": p.get("teamfight_participation"),
                "stuns": p.get("stuns", 0),
                "rune_pickups": p.get("rune_pickups", 0),
                "actions_per_min": p.get("actions_per_min", 0),
                "kills_log": p.get("kills_log"),
            })
        return players

    def _extract_teamfights(
        self, od_data: dict, teamfights_raw: list[dict]
    ) -> tuple[list[dict], list[list[dict]]]:
        """
        Extract teamfight records and per-player fight stats.
        v2: includes ability_targets, damage_targets, deaths_pos.
        """
        match_id = od_data["match_id"]
        players = od_data.get("players", [])

        teamfight_records = []
        all_fight_stats = []

        for idx, fight in enumerate(teamfights_raw):
            fight_players = fight.get("players", [])
            radiant_kills = 0
            dire_kills = 0
            gold_swing = 0

            fight_stats = []
            for player_idx, fp in enumerate(fight_players):
                is_radiant = player_idx < 5
                killed = fp.get("killed", {})
                kills_count = (
                    sum(killed.values())
                    if isinstance(killed, dict)
                    else 0
                )

                if is_radiant:
                    radiant_kills += kills_count
                else:
                    dire_kills += kills_count

                gold_swing += (
                    fp.get("gold_delta", 0)
                    if is_radiant
                    else -fp.get("gold_delta", 0)
                )

                hero_id = (
                    players[player_idx].get("hero_id")
                    if player_idx < len(players)
                    else None
                )
                account_id = (
                    players[player_idx].get("account_id")
                    if player_idx < len(players)
                    else None
                )

                fight_stats.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "ability_uses": fp.get("ability_uses", {}),
                    "ability_targets": fp.get("ability_targets", {}),
                    "damage_targets": fp.get("damage_targets", {}),
                    "deaths_pos": fp.get("deaths_pos", {}),
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
                })

            start = fight.get("start", 0)
            end = fight.get("end", 0)
            teamfight_records.append({
                "match_id": match_id,
                "fight_index": idx,
                "start_time": start,
                "end_time": end,
                # NOTE: duration is a GENERATED column (end_time - start_time) — do NOT include it
                "deaths_count": fight.get("deaths", 0),
                "radiant_kills": radiant_kills,
                "dire_kills": dire_kills,
                "gold_swing": gold_swing,
            })
            all_fight_stats.append(fight_stats)

        return teamfight_records, all_fight_stats

    # ── Fight Context (Feature 4) ─────────────────────────

    def _store_fight_contexts(
        self,
        od_data: dict,
        teamfight_records: list[dict],
        inserted_fights: list[dict],
    ):
        """Build and store team state snapshot for each fight."""
        from app.core.database import get_supabase
        sb = get_supabase()

        players = od_data.get("players", [])
        contexts = []

        for i, inserted in enumerate(inserted_fights):
            try:
                ctx = self._build_fight_context(
                    od_data,
                    teamfight_records[i],
                    inserted["id"],
                    players,
                )
                if ctx:
                    contexts.append(ctx)
            except Exception as e:
                logger.warning(
                    f"Fight context extraction failed for fight "
                    f"{teamfight_records[i].get('fight_index')}: {e}"
                )

        if contexts:
            sb.table("fight_context").insert(contexts).execute()

    def _build_fight_context(
        self,
        od_data: dict,
        teamfight_record: dict,
        teamfight_id: int,
        players: list[dict],
    ) -> dict | None:
        """Build team state snapshot at fight start time."""
        fight_time = teamfight_record["start_time"]
        minute_idx = max(0, fight_time // 60)

        def _get_at_time(arr, idx):
            if not arr:
                return 0
            return arr[idx] if idx < len(arr) else arr[-1]

        radiant_nws = []
        dire_nws = []
        radiant_xps = []
        dire_xps = []
        radiant_items = {}
        dire_items = {}

        for i, p in enumerate(players):
            is_radiant = p.get("player_slot", 0) < 128
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []

            nw = _get_at_time(gold_t, minute_idx)
            xp = _get_at_time(xp_t, minute_idx)

            # Reconstruct inventory at fight time
            items_at_time = []
            for purchase in (p.get("purchase_log") or []):
                if purchase.get("time", 0) <= fight_time:
                    items_at_time.append(purchase["key"])

            # Position label (sorted by NW later)
            pos_label = f"pos{(i % 5) + 1}"

            if is_radiant:
                radiant_nws.append(nw)
                radiant_xps.append(xp)
                radiant_items[pos_label] = items_at_time[-6:]
            else:
                dire_nws.append(nw)
                dire_xps.append(xp)
                dire_items[pos_label] = items_at_time[-6:]

        # Sort NW descending so pos1=highest farm
        radiant_nws.sort(reverse=True)
        dire_nws.sort(reverse=True)

        r_total = sum(radiant_nws)
        d_total = sum(dire_nws)

        def _safe(arr, idx):
            return arr[idx] if idx < len(arr) else 0

        return {
            "teamfight_id": teamfight_id,
            "match_id": od_data["match_id"],
            "radiant_total_nw": r_total,
            "dire_total_nw": d_total,
            "nw_difference": r_total - d_total,
            "radiant_xp_total": sum(radiant_xps),
            "dire_xp_total": sum(dire_xps),
            "radiant_pos1_nw": _safe(radiant_nws, 0),
            "radiant_pos2_nw": _safe(radiant_nws, 1),
            "radiant_pos3_nw": _safe(radiant_nws, 2),
            "radiant_pos4_nw": _safe(radiant_nws, 3),
            "radiant_pos5_nw": _safe(radiant_nws, 4),
            "dire_pos1_nw": _safe(dire_nws, 0),
            "dire_pos2_nw": _safe(dire_nws, 1),
            "dire_pos3_nw": _safe(dire_nws, 2),
            "dire_pos4_nw": _safe(dire_nws, 3),
            "dire_pos5_nw": _safe(dire_nws, 4),
            "radiant_key_items": radiant_items,
            "dire_key_items": dire_items,
            "game_time_minutes": fight_time / 60.0,
        }

    # ── Objectives (Feature 3) ────────────────────────────

    def _extract_objectives(self, od_data: dict) -> list[dict]:
        """Extract objective events from OpenDota data."""
        match_id = od_data["match_id"]
        objectives = []

        for obj in od_data.get("objectives", []):
            key = str(obj.get("key") or "")
            obj_type = obj.get("type", "")

            # Derive subtype from key
            subtype = None
            if "tower" in key:
                # npc_dota_badguys_tower1_mid → tower1_mid
                parts = key.replace("npc_dota_badguys_", "").replace(
                    "npc_dota_goodguys_", ""
                )
                subtype = parts
            elif "rax" in key:
                parts = key.replace("npc_dota_badguys_", "").replace(
                    "npc_dota_goodguys_", ""
                )
                subtype = parts
            elif obj_type == "CHAT_MESSAGE_ROSHAN_KILL":
                subtype = "roshan"
            elif obj_type == "CHAT_MESSAGE_AEGIS":
                subtype = "aegis"
            elif obj_type == "CHAT_MESSAGE_FIRSTBLOOD":
                subtype = "first_blood"
            elif obj_type == "CHAT_MESSAGE_COURIER_LOST":
                subtype = "courier"

            objectives.append({
                "match_id": match_id,
                "time": obj.get("time", 0),
                "type": obj_type,
                "subtype": subtype,
                "key": key,
                "team": obj.get("team"),
                "slot": obj.get("slot"),
                "player_slot": obj.get("player_slot"),
                "unit": obj.get("unit"),
                "value": obj.get("value"),
            })

        return objectives

    # ── Laning Phase (Feature 6) ──────────────────────────

    def _extract_laning(self, od_data: dict) -> list[dict]:
        """Extract laning phase stats (first 10 min) per player."""
        match_id = od_data["match_id"]
        results = []

        # Build a lane opponent lookup for advantage calculation
        lane_players = {}  # {(team, lane): [player_data]}
        for p in od_data.get("players", []):
            is_radiant = p.get("player_slot", 0) < 128
            lane = p.get("lane")
            if lane:
                team = "radiant" if is_radiant else "dire"
                lane_players.setdefault((team, lane), []).append(p)

        for p in od_data.get("players", []):
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []
            lh_t = p.get("lh_t") or []
            dn_t = p.get("dn_t") or []

            def _at(arr, idx):
                if not arr:
                    return 0
                return arr[idx] if idx < len(arr) else (arr[-1] if arr else 0)

            gold_5 = _at(gold_t, 5)
            gold_10 = _at(gold_t, 10)
            xp_5 = _at(xp_t, 5)
            xp_10 = _at(xp_t, 10)
            lh_5 = _at(lh_t, 5)
            lh_10 = _at(lh_t, 10)
            dn_5 = _at(dn_t, 5)
            dn_10 = _at(dn_t, 10)

            # Count lane kills/deaths from kills_log
            kills_in_lane = 0
            for kill in (p.get("kills_log") or []):
                if kill.get("time", 999) <= 600:
                    kills_in_lane += 1

            # Deaths in lane — approximate from full death count vs timing
            deaths_in_lane = 0
            # Use life_state timing if available, otherwise estimate
            # For now, count deaths from all teamfights before 600s
            for fight in od_data.get("teamfights", []):
                if fight.get("start", 0) > 600:
                    break
                slot = p.get("player_slot", 0)
                idx = slot if slot < 128 else slot - 123
                fight_players = fight.get("players", [])
                if idx < len(fight_players):
                    deaths_in_lane += fight_players[idx].get("deaths", 0)

            # Lane advantage vs opponent
            is_radiant = p.get("player_slot", 0) < 128
            lane = p.get("lane")
            lane_gold_adv = None
            lane_xp_adv = None

            if lane:
                opp_team = "dire" if is_radiant else "radiant"
                opponents = lane_players.get((opp_team, lane), [])
                if opponents:
                    opp_gold_10 = sum(
                        _at(o.get("gold_t") or [], 10) for o in opponents
                    ) / len(opponents)
                    opp_xp_10 = sum(
                        _at(o.get("xp_t") or [], 10) for o in opponents
                    ) / len(opponents)
                    lane_gold_adv = int(gold_10 - opp_gold_10)
                    lane_xp_adv = int(xp_10 - opp_xp_10)

            results.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "player_slot": p.get("player_slot"),
                "is_radiant": is_radiant,
                "lane": lane,
                "gold_at_5min": gold_5,
                "gold_at_10min": gold_10,
                "xp_at_5min": xp_5,
                "xp_at_10min": xp_10,
                "lh_at_5min": lh_5,
                "lh_at_10min": lh_10,
                "dn_at_5min": dn_5,
                "dn_at_10min": dn_10,
                "kills_in_lane": kills_in_lane,
                "deaths_in_lane": deaths_in_lane,
                "lane_gold_advantage": lane_gold_adv,
                "lane_xp_advantage": lane_xp_adv,
                "cs_per_min_5": round(lh_5 / 5.0, 2) if lh_5 else 0,
                "cs_per_min_10": round(lh_10 / 10.0, 2) if lh_10 else 0,
                "lane_pos": p.get("lane_pos"),
            })

        return results

    # ── Itemization (Feature 7) ───────────────────────────

    def _extract_itemization(self, od_data: dict) -> list[dict]:
        """Extract itemization data for each player."""
        match_id = od_data["match_id"]
        results = []

        # Build enemy lineup per side
        radiant_heroes = []
        dire_heroes = []
        for p in od_data.get("players", []):
            if p.get("player_slot", 0) < 128:
                radiant_heroes.append(p.get("hero_id"))
            else:
                dire_heroes.append(p.get("hero_id"))

        for p in od_data.get("players", []):
            is_radiant = p.get("player_slot", 0) < 128
            purchase_log = p.get("purchase_log") or []

            # Key item timings
            item_timings = {}
            for purchase in purchase_log:
                key = purchase.get("key", "")
                if key in KEY_ITEMS and key not in item_timings:
                    item_timings[key] = purchase.get("time", 0)

            # Final items
            final_items = [p.get(f"item_{i}") for i in range(6)]
            final_backpack = [p.get(f"backpack_{i}") for i in range(3)]

            results.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "player_slot": p.get("player_slot"),
                "is_radiant": is_radiant,
                "purchase_log": purchase_log,
                "item_timings": item_timings,
                "final_items": final_items,
                "final_backpack": final_backpack,
                "enemy_hero_ids": dire_heroes if is_radiant else radiant_heroes,
            })

        return results

    # ── Farming Efficiency (Feature 8) ────────────────────

    def _extract_farming(self, od_data: dict) -> list[dict]:
        """Extract farming efficiency data per player."""
        match_id = od_data["match_id"]
        results = []

        # Get teamfight timestamps for farming window detection
        fight_times = []
        for fight in od_data.get("teamfights", []):
            fight_times.append(
                (fight.get("start", 0), fight.get("end", 0))
            )

        for p in od_data.get("players", []):
            gold_t = p.get("gold_t") or []
            xp_t = p.get("xp_t") or []
            lh_t = p.get("lh_t") or []

            # Estimate idle minutes
            idle_minutes = 0.0
            idle_threshold = 150
            for i in range(1, len(gold_t)):
                gold_gain = gold_t[i] - gold_t[i - 1]
                if gold_gain < idle_threshold:
                    idle_minutes += 1.0

            # Identify farming windows (gaps between fights)
            farming_windows = []
            sorted_fights = sorted(fight_times, key=lambda x: x[0])
            # Window from laning end to first fight
            if sorted_fights:
                if sorted_fights[0][0] > 600:  # first fight after 10 min
                    farming_windows.append({
                        "start": 600,
                        "end": sorted_fights[0][0],
                    })
                # Gaps between fights
                for i in range(len(sorted_fights) - 1):
                    gap_start = sorted_fights[i][1]
                    gap_end = sorted_fights[i + 1][0]
                    if gap_end - gap_start >= 60:  # at least 1 min gap
                        farming_windows.append({
                            "start": gap_start,
                            "end": gap_end,
                        })

            # Calculate gold earned in each window
            for window in farming_windows:
                start_min = window["start"] // 60
                end_min = window["end"] // 60
                if start_min < len(gold_t) and end_min < len(gold_t):
                    window["gold_earned"] = (
                        gold_t[end_min] - gold_t[start_min]
                    )
                    if start_min < len(lh_t) and end_min < len(lh_t):
                        window["lh_count"] = (
                            lh_t[end_min] - lh_t[start_min]
                        )

            results.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "hero_id": p.get("hero_id"),
                "player_slot": p.get("player_slot"),
                "is_radiant": p.get("player_slot", 0) < 128,
                "gold_t": gold_t,
                "xp_t": xp_t,
                "lh_t": lh_t,
                "gpm": p.get("gold_per_min"),
                "xpm": p.get("xp_per_min"),
                "total_last_hits": p.get("last_hits"),
                "total_denies": p.get("denies"),
                "estimated_idle_minutes": round(idle_minutes, 1),
                "farming_heatmap": p.get("lane_pos"),
                "farming_windows": farming_windows,
            })

        return results

    # ── Chat / Toxicity (Feature 10) ──────────────────────

    def _extract_chat(self, od_data: dict) -> list[dict]:
        """Extract chat data for toxicity analysis."""
        match_id = od_data["match_id"]
        chat_messages = od_data.get("chat") or []
        players = od_data.get("players", [])

        if not chat_messages:
            return []

        # Group messages by player slot index
        messages_by_idx = {}
        for msg in chat_messages:
            idx = msg.get("slot")
            if idx is not None:
                messages_by_idx.setdefault(idx, []).append({
                    "time": msg.get("time"),
                    "key": msg.get("key", ""),
                    "type": msg.get("type", ""),
                })

        results = []
        for i, p in enumerate(players):
            player_messages = messages_by_idx.get(i, [])
            if not player_messages:
                continue

            all_chat = [
                m for m in player_messages if m.get("type") == "chat"
            ]

            results.append({
                "match_id": match_id,
                "account_id": p.get("account_id"),
                "player_slot": p.get("player_slot"),
                "hero_id": p.get("hero_id"),
                "chat_messages": player_messages,
                "total_messages": len(player_messages),
                "all_chat_messages": len(all_chat),
            })

        return results

    # ── Ward Details (Feature 5 - enhanced) ───────────────

    def _extract_ward_details(self, od_data: dict) -> list[dict]:
        """Enhanced ward extraction with lifespan and deward tracking."""
        match_id = od_data["match_id"]
        wards = []

        for player in od_data.get("players", []):
            account_id = player.get("account_id")
            hero_id = player.get("hero_id")

            # Index destruction events by approximate position
            obs_left = {}
            for w in (player.get("obs_left_log") or []):
                x = int(float(w.get("x", 0)))
                y = int(float(w.get("y", 0)))
                obs_left[(x, y)] = w

            sen_left = {}
            for w in (player.get("sen_left_log") or []):
                x = int(float(w.get("x", 0)))
                y = int(float(w.get("y", 0)))
                sen_left[(x, y)] = w

            for ward in (player.get("obs_log") or []):
                x = int(float(ward.get("x", 0)))
                y = int(float(ward.get("y", 0)))
                place_time = ward.get("time", 0)
                left = obs_left.pop((x, y), None)

                destroyed_at = left.get("time") if left else None
                destroyed_by = left.get("attackername") if left else None

                wards.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "time": place_time,
                    "x": x,
                    "y": y,
                    "ward_type": "observer",
                    "destroyed_at": destroyed_at,
                    "destroyed_by": destroyed_by,
                    "duration_alive": (
                        (destroyed_at - place_time) if destroyed_at else None
                    ),
                    "is_dewarded": bool(
                        destroyed_by
                        and "hero" in (destroyed_by or "")
                    ),
                })

            for ward in (player.get("sen_log") or []):
                x = int(float(ward.get("x", 0)))
                y = int(float(ward.get("y", 0)))
                place_time = ward.get("time", 0)
                left = sen_left.pop((x, y), None)

                destroyed_at = left.get("time") if left else None
                destroyed_by = left.get("attackername") if left else None

                wards.append({
                    "match_id": match_id,
                    "account_id": account_id,
                    "hero_id": hero_id,
                    "time": place_time,
                    "x": x,
                    "y": y,
                    "ward_type": "sentry",
                    "destroyed_at": destroyed_at,
                    "destroyed_by": destroyed_by,
                    "duration_alive": (
                        (destroyed_at - place_time) if destroyed_at else None
                    ),
                    "is_dewarded": bool(
                        destroyed_by
                        and "hero" in (destroyed_by or "")
                    ),
                })

        return wards

    # ── Positions (unchanged) ─────────────────────────────

    def _extract_positions(
        self, match_id: int, stratz_data: dict
    ) -> list[dict]:
        """Extract position events from STRATZ playbackData."""
        positions = []
        players = stratz_data.get("players", [])

        for player in players:
            playback = player.get("playbackData")
            if not playback:
                continue

            account_id = player.get("steamAccountId")
            hero_id = player.get("heroId")

            pos_events = playback.get(
                "playerUpdatePositionEvents", []
            )
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
    
    # ── Parser Worker ────────────────────────────────────────────────
    async def request_local_parse(self, match_id: int) -> dict:
        replay_url = await self.opendota.get_replay_url(match_id)
        if not replay_url:
            raise ValueError(f"Cannot resolve replay URL for match {match_id}. "
                            "Match may be too old (>7 days) or from a private lobby.")

        from app.services.parse_dispatcher import enqueue_parse_job
        job_id = await enqueue_parse_job(match_id, replay_url)
        return {"status": "parse_queued", "job_id": job_id}

    async def close(self):
        await self.opendota.close()
        await self.stratz.close()