# Phase 2: Benchmarks & Core ML

## New Files (drop into your repo)

```
app/ml/
├── __init__.py              # Constants, score weights, bucket definitions
├── feature_engineering.py   # Extract ML features from fight data
├── benchmarks.py            # Percentile benchmarks + recommendations
├── fight_iq_model.py        # XGBoost regression: Fight IQ 0-100
├── fight_outcome_model.py   # XGBoost classification: fight win probability
├── clustering.py            # DBSCAN fight archetype discovery
└── scoring.py               # Inference service (loads models, scores fights)

scripts/
└── train_models.py          # CLI to run full training pipeline
```

## Quick Start

### 1. Run the full training pipeline

```bash
# Dry run first (computes everything, writes nothing)
python -m scripts.train_models --dry-run

# Full pipeline: benchmarks → Fight IQ → Outcome → DBSCAN → vectors
python -m scripts.train_models
```

### 2. Run individual steps

```bash
python -m scripts.train_models --benchmarks-only
python -m scripts.train_models --fight-iq-only
python -m scripts.train_models --outcome-only
python -m scripts.train_models --clustering-only
python -m scripts.train_models --vectors-only
```

### 3. Integrate scoring into FastAPI

Add to `app/main.py`:

```python
from app.ml.scoring import get_scorer, reload_models

@app.on_event("startup")
async def load_ml_models():
    """Load ML models into memory on startup."""
    scorer = get_scorer()  # triggers model loading

@app.get("/api/models/info")
async def model_info():
    """Check loaded model versions and metrics."""
    return get_scorer().get_model_info()

@app.post("/api/models/reload")
async def reload():
    """Reload models after retraining."""
    scorer = reload_models()
    return scorer.get_model_info()
```

Update the fight detail endpoint to include scores:

```python
@app.get("/api/fights/{match_id}/{fight_index}")
async def get_fight_detail(match_id: int, fight_index: int):
    sb = db.get_supabase()
    fight = (
        sb.table("teamfights")
        .select("*, fight_player_stats(*)")
        .eq("match_id", match_id)
        .eq("fight_index", fight_index)
        .execute()
    ).data

    if not fight:
        raise HTTPException(404)

    fight_data = fight[0]
    scorer = get_scorer()

    # Get match + player data for context
    match = db.get_match(match_id)
    players = db.get_match_players(match_id)

    # Score each player in the fight
    scored_players = []
    for stat in fight_data.get("fight_player_stats", []):
        score = scorer.score_player_fight(stat, fight_data, match, players)
        scored_players.append({**stat, "analysis": score})

    # Fight outcome prediction
    outcome = scorer.predict_fight_outcome(
        fight_data, fight_data.get("fight_player_stats", []), match, players
    )

    return {
        "fight": fight_data,
        "player_scores": scored_players,
        "outcome_prediction": outcome,
    }
```

## What Each Piece Does

### Feature Engineering (`feature_engineering.py`)
Transforms raw `fight_player_stats` rows into ~20 ML features:
- `ability_casts_per_sec`, `damage_per_sec`, `damage_per_nw`
- `bkb_used`, `blink_used`, `item_activations`
- `kills`, `deaths`, `survived`, `gold_delta`, `xp_delta`
- Context buckets: `time_bucket`, `nw_bucket`, `duration_bucket`, `size_bucket`

### Benchmarks (`benchmarks.py`)
Computes p25/median/p75/p90 for each metric, stratified by hero × context.
With 24K fight samples, you get ~23K possible context combinations.
Buckets with <10 samples are flagged as low-confidence.

### Fight IQ Score (`fight_iq_model.py`)
XGBoost regression → 0-100 composite score.
Since all training data is 7k+ MMR, labels are generated from relative
performance within the dataset (percentile rank across key metrics).
Sub-scores: ability (30%), damage (25%), item (20%), survival (15%), extraction (10%).

### Fight Outcome (`fight_outcome_model.py`)
XGBoost binary classifier → radiant win probability.
Features: net worth advantage, team totals, fight timing/size.
Shows users "you had a 65% chance but lost — here's the execution gap."

### DBSCAN Clustering (`clustering.py`)
Discovers fight archetypes automatically: pickoff, smoke_gank, highground_siege,
early_skirmish, open_5v5, decisive_wipe, roshan_contest, etc.
Labels written to `teamfights.fight_archetype` column.

### Scoring Service (`scoring.py`)
Singleton that loads all models into memory at FastAPI startup.
Provides `score_player_fight()` and `predict_fight_outcome()` for the API layer.

## DB Tables Used (already in your schema)

- `hero_benchmarks` — populated by benchmark pipeline
- `fight_scores` — store Fight IQ results per player per fight
- `fight_predictions` — store outcome predictions per fight
- `fight_vectors` — pgvector embeddings for similar fight search
- `teamfights.fight_archetype` — updated by DBSCAN clustering

## Expected Output (290 matches, 24K fight samples)

```
STEP 1: Computing benchmarks
  → ~15K-20K benchmark rows across ~80-100 heroes

STEP 2: Training Fight IQ model
  → MAE: ~8-12, R²: 0.3-0.6 (expected for performance scoring)
  → Top features: damage_per_sec, gold_delta, ability_casts_per_sec

STEP 3: Training Fight Outcome model
  → Accuracy: 0.55-0.65, AUC: 0.55-0.70
  → (Above 0.5 means it's learning real patterns, not random)

STEP 4: DBSCAN clustering
  → 5-8 clusters + noise
  → Auto-labeled: pickoff, smoke_gank, teamfight, etc.

STEP 5: Similarity vectors
  → 24K 32-dim vectors in fight_vectors table
```
