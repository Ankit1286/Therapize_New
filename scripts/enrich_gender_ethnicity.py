"""
One-off enrichment script: populate missing gender/ethnicity for OpenPath therapists.

Fetches HTML profile pages for every OpenPath therapist that is missing gender
or ethnicity, parses the labeled <p> blocks, and writes the values to the DB.

Run from repo root:
    python scripts/enrich_gender_ethnicity.py
"""
import asyncio
import logging
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

load_dotenv(_repo_root / ".env")

from src.scrapers.open_path import OpenPathScraper  # noqa: E402
from src.storage.database import TherapistRepository, close_db, get_connection, init_db  # noqa: E402

logger = logging.getLogger(__name__)

_CONCURRENCY = 15
_TIMEOUT = 15.0
_USER_AGENT = "Mozilla/5.0 (compatible; TherapizeBot/1.0; research purposes)"

_scraper = OpenPathScraper()


async def _get_missing_records() -> list[dict]:
    """Return OpenPath therapists that are missing gender or ethnicity."""
    async with get_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, source_url, name, gender, ethnicity
            FROM therapists
            WHERE source = 'open_path'
              AND is_active = TRUE
              AND (gender IS NULL OR ethnicity = '{}' OR ethnicity IS NULL)
            ORDER BY name
            """
        )
    return [dict(r) for r in rows]


async def _enrich_one(
    client: httpx.AsyncClient,
    repo: TherapistRepository,
    record: dict,
) -> str:
    """
    Fetch a single OpenPath profile page and update gender/ethnicity.

    Returns: 'updated', 'partial', 'no_data', or 'error'.
    """
    url = record["source_url"]
    try:
        resp = await client.get(url, follow_redirects=True, timeout=_TIMEOUT)
    except Exception as exc:
        logger.debug("Network error for %s: %s", url, exc)
        return "error"

    if resp.status_code != 200:
        logger.debug("Non-200 (%d) for %s", resp.status_code, url)
        return "error"

    gender, ethnicity = _scraper._parse_gender_ethnicity_from_html(resp.text)

    # Only update if we got at least something new
    had_gender = record["gender"] is not None
    had_ethnicity = bool(record["ethnicity"])

    new_gender = gender if gender else record["gender"]  # keep existing if HTML has nothing
    new_ethnicity = ethnicity if ethnicity else (list(record["ethnicity"]) if record["ethnicity"] else [])

    if new_gender == record["gender"] and new_ethnicity == list(record["ethnicity"] or []):
        return "no_data"

    await repo.update_gender_ethnicity(record["id"], new_gender, new_ethnicity)

    got_gender = new_gender is not None
    got_ethnicity = bool(new_ethnicity)

    if got_gender and got_ethnicity:
        logger.info("UPDATED  %-45s gender=%-12s ethnicity=%s", record["name"], new_gender, new_ethnicity)
        return "updated"
    logger.info("PARTIAL  %-45s gender=%-12s ethnicity=%s", record["name"], new_gender, new_ethnicity)
    return "partial"


async def main() -> None:
    """Run the gender/ethnicity enrichment pass."""
    await init_db()
    repo = TherapistRepository()

    records = await _get_missing_records()
    logger.info("Found %d OpenPath therapists with missing gender or ethnicity", len(records))

    sem = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(headers={"User-Agent": _USER_AGENT}) as client:
        async def bounded(record: dict) -> str:
            async with sem:
                return await _enrich_one(client, repo, record)

        results = await asyncio.gather(*[bounded(r) for r in records])

    counts = {k: results.count(k) for k in ("updated", "partial", "no_data", "error")}
    logger.info(
        "Enrichment complete — updated=%d, partial=%d, no_data=%d, error=%d",
        counts["updated"], counts["partial"], counts["no_data"], counts["error"],
    )

    # Final coverage stats
    async with get_connection() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM therapists WHERE source='open_path' AND is_active=TRUE")
        has_gender = await conn.fetchval("SELECT COUNT(*) FROM therapists WHERE source='open_path' AND is_active=TRUE AND gender IS NOT NULL")
        has_eth = await conn.fetchval("SELECT COUNT(*) FROM therapists WHERE source='open_path' AND is_active=TRUE AND ethnicity != '{}'")
    logger.info(
        "Coverage: gender %d/%d (%.0f%%), ethnicity %d/%d (%.0f%%)",
        has_gender, total, 100 * has_gender / total,
        has_eth, total, 100 * has_eth / total,
    )

    await close_db()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(main())
