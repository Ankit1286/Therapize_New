"""
One-command dev setup: run migrations + scrape 10 real therapists into Neon.

Usage:
    python scripts/setup_dev.py

Scrapes 5 therapists from OpenPath and 5 from GoodTherapy, embeds them,
and stores them in the Neon database. Takes ~2 minutes to run.
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


async def main() -> None:
    # ── Step 1: Migrations ────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("Step 1/3: Running database migrations...")
    logger.info("=" * 50)
    from scripts.migrate import run_migrations
    await run_migrations()

    # ── Step 2 + 3: Scrape + ingest ───────────────────────────────────────────
    from src.storage.database import init_db, close_db
    await init_db()

    from src.pipeline.ingestion import IngestionPipeline
    pipeline = IngestionPipeline()

    logger.info("=" * 50)
    logger.info("Step 2/3: Scraping 5 therapists from OpenPath...")
    logger.info("=" * 50)
    open_path_stats = await pipeline.run_open_path(max_therapists=5)
    logger.info(
        "OpenPath: scraped=%d  stored=%d  failed=%d",
        open_path_stats.scraped, open_path_stats.stored, open_path_stats.failed,
    )

    logger.info("=" * 50)
    logger.info("Step 3/3: Scraping 5 therapists from GoodTherapy...")
    logger.info("=" * 50)
    good_therapy_stats = await pipeline.run_good_therapy(
        max_therapists=5,
        cities=["los-angeles"],  # avoid crawling all 189 CA cities for dev
    )
    logger.info(
        "GoodTherapy: scraped=%d  stored=%d  failed=%d",
        good_therapy_stats.scraped, good_therapy_stats.stored, good_therapy_stats.failed,
    )

    await close_db()

    total = open_path_stats.stored + good_therapy_stats.stored
    logger.info("=" * 50)
    logger.info("Done! %d therapists stored in Neon.", total)
    logger.info("=" * 50)
    logger.info("Next steps:")
    logger.info("  1. uvicorn src.api.main:app --port 8000")
    logger.info("  2. streamlit run frontend/app.py")


if __name__ == "__main__":
    asyncio.run(main())
