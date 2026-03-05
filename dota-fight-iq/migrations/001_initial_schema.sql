-- ============================================================
-- Dota Fight IQ — Database Schema v1.0
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)
-- ============================================================

-- Enable pgvector extension for similarity search (MVP replacement for Elasticsearch)
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Users ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    steam_id TEXT UNIQUE NOT NULL,
    persona_name TEXT,
    avatar_url TEXT,
    mmr_estimate INT,
    rank_tier INT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_steam_id ON users(steam_id);

-- ── Matches ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS matches (
    match_id BIGINT PRIMARY KEY,
    duration INT,                    -- seconds
    start_time BIGINT,               -- unix timestamp
    game_mode INT,
    lobby_type INT,
    radiant_win BOOLEAN,
    patch TEXT,
    avg_rank_tier INT,
    radiant_score INT,
    dire_score INT,
    first_blood_time INT,
    is_parsed BOOLEAN DEFAULT FALSE,
    s3_key TEXT,                      -- raw JSON location in S3
    processed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Match Players ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS match_players (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    account_id BIGINT,
    hero_id INT NOT NULL,
    player_slot INT,
    kills INT,
    deaths INT,
    assists INT,
    gpm INT,                         -- gold per minute
    xpm INT,                         -- xp per minute
    last_hits INT,
    denies INT,
    hero_damage INT,
    tower_damage INT,
    hero_healing INT,
    net_worth INT,
    level INT,
    role TEXT,                        -- core/support/offlane etc
    lane INT,                        -- 1=safe, 2=mid, 3=off
    is_radiant BOOLEAN,
    items JSONB,                     -- item slots at end of game
    purchase_log JSONB,              -- [{time, key}] item purchase timeline
    gold_t INT[],                    -- gold at each minute
    xp_t INT[],                      -- xp at each minute
    UNIQUE(match_id, account_id)
);

CREATE INDEX idx_match_players_match ON match_players(match_id);
CREATE INDEX idx_match_players_hero ON match_players(hero_id);
CREATE INDEX idx_match_players_account ON match_players(account_id);

-- ── Teamfights ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS teamfights (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    fight_index INT NOT NULL,         -- 0-based index within the match
    start_time INT NOT NULL,          -- game time seconds
    end_time INT NOT NULL,
    duration INT GENERATED ALWAYS AS (end_time - start_time) STORED,
    deaths_count INT,
    radiant_kills INT,
    dire_kills INT,
    gold_swing INT,                   -- net gold change (radiant perspective)
    fight_archetype TEXT,             -- set by DBSCAN later: smoke_gank, highground, rosh, etc
    UNIQUE(match_id, fight_index)
);

CREATE INDEX idx_teamfights_match ON teamfights(match_id);

-- ── Fight Player Stats ────────────────────────────────────

CREATE TABLE IF NOT EXISTS fight_player_stats (
    id BIGSERIAL PRIMARY KEY,
    teamfight_id BIGINT NOT NULL REFERENCES teamfights(id) ON DELETE CASCADE,
    match_id BIGINT NOT NULL,
    account_id BIGINT,
    hero_id INT NOT NULL,
    ability_uses JSONB,              -- {"echo_slam": 1, "fissure": 2}
    item_uses JSONB,                 -- {"blink": 1, "black_king_bar": 1}
    damage INT DEFAULT 0,
    healing INT DEFAULT 0,
    gold_delta INT DEFAULT 0,
    xp_delta INT DEFAULT 0,
    deaths INT DEFAULT 0,
    killed JSONB,                    -- {"npc_dota_hero_antimage": 1}
    buybacks INT DEFAULT 0,
    items_at_fight JSONB,            -- inventory snapshot at fight start
    xp_start INT,
    xp_end INT,
    is_radiant BOOLEAN
);

CREATE INDEX idx_fight_stats_teamfight ON fight_player_stats(teamfight_id);
CREATE INDEX idx_fight_stats_hero ON fight_player_stats(hero_id);

-- ── Hero Benchmarks ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS hero_benchmarks (
    id BIGSERIAL PRIMARY KEY,
    hero_id INT NOT NULL,
    time_bucket TEXT NOT NULL,        -- '0-15', '15-25', '25-35', '35-45', '45+'
    nw_bucket TEXT NOT NULL,          -- 'below_avg', 'average', 'above_avg', 'far_ahead'
    duration_bucket TEXT NOT NULL,    -- 'short', 'medium', 'long'
    size_bucket TEXT NOT NULL,        -- 'skirmish', 'teamfight', 'bloodbath'
    metric_name TEXT NOT NULL,        -- 'ability_casts_per_sec', 'damage_per_sec', etc
    p25 FLOAT,
    median FLOAT,
    p75 FLOAT,
    p90 FLOAT,
    sample_count INT DEFAULT 0,
    patch TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(hero_id, time_bucket, nw_bucket, duration_bucket, size_bucket, metric_name)
);

CREATE INDEX idx_benchmarks_hero ON hero_benchmarks(hero_id);

-- ── Fight IQ Scores ───────────────────────────────────────

CREATE TABLE IF NOT EXISTS fight_scores (
    id BIGSERIAL PRIMARY KEY,
    teamfight_id BIGINT REFERENCES teamfights(id) ON DELETE CASCADE,
    match_id BIGINT NOT NULL,
    account_id BIGINT,
    hero_id INT NOT NULL,
    fight_iq_score FLOAT,            -- 0-100 composite score
    ability_score FLOAT,             -- sub-score: ability utilization
    damage_score FLOAT,              -- sub-score: damage efficiency
    item_score FLOAT,                -- sub-score: item activation
    survival_score FLOAT,            -- sub-score: survival
    extraction_score FLOAT,          -- sub-score: gold/xp extraction
    component_details JSONB,         -- full breakdown
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_fight_scores_match ON fight_scores(match_id);
CREATE INDEX idx_fight_scores_account ON fight_scores(account_id);

-- ── Fight Outcome Predictions ─────────────────────────────

CREATE TABLE IF NOT EXISTS fight_predictions (
    id BIGSERIAL PRIMARY KEY,
    teamfight_id BIGINT REFERENCES teamfights(id) ON DELETE CASCADE,
    predicted_radiant_win_prob FLOAT,
    actual_winner TEXT,               -- 'radiant' or 'dire'
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Death Analysis ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS death_analysis (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    account_id BIGINT,
    hero_id INT NOT NULL,
    time INT NOT NULL,                -- game time of death
    classification TEXT,              -- 'avoidable' or 'unavoidable'
    confidence FLOAT,
    context JSONB,                    -- pre-death state: position, nearby allies, wards, etc
    model_version TEXT
);

CREATE INDEX idx_deaths_match ON death_analysis(match_id);

-- ── Player Positions (from STRATZ) ────────────────────────

CREATE TABLE IF NOT EXISTS player_positions (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    account_id BIGINT,
    hero_id INT,
    time INT NOT NULL,                -- game time seconds
    x INT NOT NULL,
    y INT NOT NULL
);

-- Partition-friendly index for time-range queries per match
CREATE INDEX idx_positions_match_time ON player_positions(match_id, time);

-- ── Ward Events ───────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ward_events (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    account_id BIGINT,
    hero_id INT,
    time INT NOT NULL,
    x INT NOT NULL,
    y INT NOT NULL,
    ward_type TEXT NOT NULL           -- 'observer' or 'sentry'
);

CREATE INDEX idx_wards_match ON ward_events(match_id);

-- ── User Hero Stats (aggregated) ──────────────────────────

CREATE TABLE IF NOT EXISTS user_hero_stats (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    hero_id INT NOT NULL,
    avg_fight_iq FLOAT,
    total_fights_analyzed INT DEFAULT 0,
    best_score FLOAT,
    worst_score FLOAT,
    trend_direction TEXT,             -- 'improving', 'declining', 'stable'
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, hero_id)
);

-- ── Match Analysis Requests ───────────────────────────────

CREATE TABLE IF NOT EXISTS match_analyses (
    id BIGSERIAL PRIMARY KEY,
    match_id BIGINT NOT NULL,
    user_id UUID REFERENCES users(id),
    status TEXT DEFAULT 'pending',    -- pending → processing → complete → failed
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_analyses_match ON match_analyses(match_id);
CREATE INDEX idx_analyses_status ON match_analyses(status);

-- ── Match Collection Pool (for data pipeline) ─────────────

CREATE TABLE IF NOT EXISTS match_collection_pool (
    match_id BIGINT PRIMARY KEY,
    source TEXT DEFAULT 'stratz',     -- where we discovered this match
    hero_ids INT[],                   -- heroes in this match (for filtering)
    avg_rank INT,
    status TEXT DEFAULT 'pending',    -- pending → fetching → parsed → processed → failed
    discovered_at TIMESTAMPTZ DEFAULT NOW(),
    fetched_at TIMESTAMPTZ,
    processed_at TIMESTAMPTZ
);

CREATE INDEX idx_pool_status ON match_collection_pool(status);

-- ── Fight Vectors (for pgvector similarity search) ────────

CREATE TABLE IF NOT EXISTS fight_vectors (
    id BIGSERIAL PRIMARY KEY,
    teamfight_id BIGINT REFERENCES teamfights(id) ON DELETE CASCADE,
    match_id BIGINT NOT NULL,
    hero_id INT NOT NULL,
    embedding vector(32),             -- 32-dim feature vector for similarity search
    metadata JSONB,                   -- game_time, net_worth, fight_size, etc for filtering
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- IVFFlat index for approximate nearest neighbor search
-- Create after inserting initial data: need at least sqrt(n) lists
-- For MVP with <10k vectors, exact search is fine
CREATE INDEX idx_fight_vectors_embedding ON fight_vectors
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX idx_fight_vectors_hero ON fight_vectors(hero_id);

-- ── Row Level Security (basic) ────────────────────────────

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE match_analyses ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data" ON users
    FOR SELECT USING (auth.uid() = id);

-- Analyses are readable by their creator
CREATE POLICY "Users can read own analyses" ON match_analyses
    FOR SELECT USING (auth.uid() = user_id);

-- Public tables (match data, benchmarks) are readable by anyone
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Matches are public" ON matches FOR SELECT USING (true);

ALTER TABLE teamfights ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Teamfights are public" ON teamfights FOR SELECT USING (true);

ALTER TABLE hero_benchmarks ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Benchmarks are public" ON hero_benchmarks FOR SELECT USING (true);

ALTER TABLE fight_scores ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Fight scores are public" ON fight_scores FOR SELECT USING (true);

-- ── Helper function: auto-update updated_at ───────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER benchmarks_updated_at
    BEFORE UPDATE ON hero_benchmarks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER user_hero_stats_updated_at
    BEFORE UPDATE ON user_hero_stats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
