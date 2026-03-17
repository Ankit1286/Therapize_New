"""
Small ingestion run for local development — scrapes ~5 therapists from each
source to verify the full pipeline without crawling thousands of profiles.

Usage:
    python scripts/ingest_small.py

Prerequisites:
    docker-compose up -d postgres redis
    python scripts/migrate.py
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ingestion import IngestionPipeline
from src.storage.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Limits ────────────────────────────────────────────────────────────────────
# OpenPath: cap by max_therapists (Algolia pages are 200 hits each, so 5 is
# always satisfied on the very first page — no extra pages fetched).
OPEN_PATH_MAX = 5

# GoodTherapy: restrict to a single well-populated city and 1 page so we
# hit at most ~30 profile URLs, then cap the profile fetches to 5.
GOOD_THERAPY_CITIES = ["los-angeles"]
GOOD_THERAPY_MAX_PAGES = 1
GOOD_THERAPY_MAX = 5


async def main() -> None:
    logger.info("Initialising database...")
    await init_db()

    pipeline = IngestionPipeline()

    # ── OpenPath (~5 therapists via Algolia API) ──────────────────────────────
    logger.info("=== OpenPath: scraping up to %d therapists ===", OPEN_PATH_MAX)
    op_stats = await pipeline.run_open_path(max_therapists=OPEN_PATH_MAX)
    logger.info(
        "OpenPath done — scraped=%d, stored=%d, failed=%d",
        op_stats.scraped, op_stats.stored, op_stats.failed,
    )

    # ── GoodTherapy (~5 therapists via HTML scraping) ─────────────────────────
    logger.info(
        "=== GoodTherapy: scraping up to %d therapists from %s ===",
        GOOD_THERAPY_MAX, GOOD_THERAPY_CITIES,
    )
    gt_stats = await pipeline.run_good_therapy(
        max_therapists=GOOD_THERAPY_MAX,
        max_pages=GOOD_THERAPY_MAX_PAGES,
        cities=GOOD_THERAPY_CITIES,
    )
    logger.info(
        "GoodTherapy done — scraped=%d, stored=%d, failed=%d",
        gt_stats.scraped, gt_stats.stored, gt_stats.failed,
    )

    total = op_stats.stored + gt_stats.stored
    logger.info("=== Total stored: %d therapists ===", total)
    logger.info(
        "Verify with: SELECT count(*), source FROM therapists GROUP BY source;"
    )


if __name__ == "__main__":
    asyncio.run(main())
