"""
Weekly profile health-check script.

For every active therapist in the database, fetches their source URL and:
- Deletes the record if the page returns 404 (profile no longer exists).
- Sets accepting_new_clients=FALSE if the page text says they are full.
- Sets accepting_new_clients=TRUE if they were marked false but now appear open.
- Skips on network errors or 5xx responses (transient — try again next week).

Run locally:
    python scripts/check_profiles.py

Run on GCP VM (add to crontab):
    0 2 * * 0 cd /opt/therapize && /opt/therapize/venv/bin/python scripts/check_profiles.py >> /var/log/therapize-audit.log 2>&1
"""
import asyncio
import logging
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load .env before importing src modules so DATABASE_URL etc. are available
load_dotenv(Path(__file__).parent.parent / ".env")

from src.storage.database import TherapistRepository, close_db, init_db  # noqa: E402

logger = logging.getLogger(__name__)

# ── "Not accepting" phrase list ────────────────────────────────────────────────
# Scanned against lowercased page text. Works for both OpenPath and GoodTherapy.
_NOT_ACCEPTING_PHRASES = [
    "not accepting new clients",
    "not accepting new patients",
    "not currently accepting",
    "no longer accepting",
    "currently not accepting",
    "closed to new clients",
]

_CONCURRENCY = 20   # simultaneous HTTP requests
_TIMEOUT = 15.0     # seconds per request
_USER_AGENT = "Mozilla/5.0 (compatible; TherapizeBot/1.0; research purposes)"


async def _check_one(
    client: httpx.AsyncClient,
    repo: TherapistRepository,
    record: dict,
) -> str:
    """
    Check a single therapist URL and update the DB accordingly.

    Returns one of: 'deleted', 'deactivated', 'activated', 'ok', 'error'.
    """
    url = record["source_url"]
    try:
        resp = await client.get(url, follow_redirects=True, timeout=_TIMEOUT)
    except Exception as exc:
        logger.debug("Network error for %s: %s", url, exc)
        return "error"

    if resp.status_code == 404:
        logger.info("DELETE (404): %s — %s", record["name"], url)
        await repo.delete_by_id(record["id"])
        return "deleted"

    if resp.status_code >= 500:
        logger.debug("Skipping %s — server error %d", url, resp.status_code)
        return "error"

    if resp.status_code == 200:
        text = resp.text.lower()
        not_accepting = any(phrase in text for phrase in _NOT_ACCEPTING_PHRASES)

        if not_accepting and record["accepting_new_clients"]:
            logger.info("DEACTIVATE: %s — %s", record["name"], url)
            await repo.set_accepting_new_clients(record["id"], False)
            return "deactivated"

        if not not_accepting and not record["accepting_new_clients"]:
            logger.info("REACTIVATE: %s — %s", record["name"], url)
            await repo.set_accepting_new_clients(record["id"], True)
            return "activated"

        return "ok"

    # Other 2xx/3xx that didn't resolve cleanly — leave unchanged
    return "ok"


async def main() -> None:
    """Run the full profile audit."""
    await init_db()
    repo = TherapistRepository()

    records = await repo.get_all_for_audit()
    logger.info("Starting audit of %d active profiles...", len(records))

    sem = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        async def bounded(record: dict) -> str:
            async with sem:
                return await _check_one(client, repo, record)

        results = await asyncio.gather(*[bounded(r) for r in records])

    counts = {
        label: results.count(label)
        for label in ("deleted", "deactivated", "activated", "ok", "error")
    }
    logger.info(
        "Audit complete — deleted=%d, deactivated=%d, activated=%d, ok=%d, error=%d",
        counts["deleted"],
        counts["deactivated"],
        counts["activated"],
        counts["ok"],
        counts["error"],
    )

    await close_db()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(main())
