"""
Full-scale ingestion run — scrapes therapists from OpenPath and GoodTherapy.

Dataset design:
- OpenPath: all CA therapists (~868), already geographically diverse statewide.
- GoodTherapy: 100 therapists per city across 10 major CA cities = ~1,000.
- Total target: ~1,800 therapists, balanced across platforms and cities.

Usage:
    python scripts/run_ingestion.py

Prerequisites:
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

# OpenPath: take all CA therapists (statewide, already geographically diverse)
OPEN_PATH_MAX = 10000  # Algolia returns ~5,866 CA therapists; cap well above that

# GoodTherapy: 100 per city across 10 major CA metros
GOOD_THERAPY_CITIES = [
    "los-angeles",
    "san-francisco",
    "san-diego",
    "sacramento",
    "san-jose",
    "oakland",
    "santa-barbara",
    "irvine",
    "fresno",
    "long-beach",
]
GOOD_THERAPY_MAX_PER_CITY = 100
GOOD_THERAPY_MAX = GOOD_THERAPY_MAX_PER_CITY * len(GOOD_THERAPY_CITIES)
GOOD_THERAPY_MAX_PAGES = 100


async def main() -> None:
    logger.info("Initialising database...")
    await init_db()

    pipeline = IngestionPipeline()

    # ── OpenPath ──────────────────────────────────────────────────────────────
    logger.info("=== OpenPath: scraping all CA therapists ===")
    op_stats = await pipeline.run_open_path(max_therapists=OPEN_PATH_MAX)
    logger.info(
        "OpenPath done — scraped=%d, stored=%d, failed=%d",
        op_stats.scraped, op_stats.stored, op_stats.failed,
    )

    # ── GoodTherapy ───────────────────────────────────────────────────────────
    logger.info(
        "=== GoodTherapy: %d cities × %d per city ===",
        len(GOOD_THERAPY_CITIES), GOOD_THERAPY_MAX_PER_CITY,
    )
    gt_stats = await pipeline.run_good_therapy(
        max_therapists=GOOD_THERAPY_MAX,
        max_pages=GOOD_THERAPY_MAX_PAGES,
        cities=GOOD_THERAPY_CITIES,
        max_per_city=GOOD_THERAPY_MAX_PER_CITY,
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
