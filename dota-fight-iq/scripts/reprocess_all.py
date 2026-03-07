# scripts/reprocess_all.py
import asyncio
from app.services.match_processor import MatchProcessor
from app.core.database import get_supabase
from app.core.storage import get_storage

async def main():
    sb = get_supabase()
    storage = get_storage()
    processor = MatchProcessor()
    
    # Get all match IDs that have raw data in storage
    matches = sb.table("matches").select("match_id").not_.is_("s3_key", "null").execute().data
    
    print(f"Reprocessing {len(matches)} matches...")
    for i, m in enumerate(matches):
        match_id = m["match_id"]
        print(f"[{i+1}/{len(matches)}] {match_id}")
        try:
            result = await processor.process_match(match_id)
            print(f"  → {result['status']}: {result.get('fights', 0)} fights")
        except Exception as e:
            print(f"  → Error: {e}")
    
    await processor.close()

asyncio.run(main())