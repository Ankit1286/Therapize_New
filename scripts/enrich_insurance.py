"""
Enrichment script: update insurance_providers and sliding_scale for GoodTherapy therapists.

Reads __NEXT_DATA__ JSON from each profile page (more reliable than HTML parsing).
Only updates insurance_providers and sliding_scale — all other fields are left untouched.

Usage:
    python scripts/enrich_insurance.py
"""
import asyncio
import json
import logging
import sys
from pathlib import Path

import asyncpg
import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
load_dotenv(_repo_root / ".env")

from src.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONCURRENCY = 15
TIMEOUT = 15.0

# Maps JSON insurance company name fragments → DB enum value
# Matched case-insensitively; order matters (longer/more specific first)
INSURANCE_NAME_MAP: list[tuple[str, str]] = [
    ("blue cross blue shield", "blue_cross"),
    ("anthem blue cross", "blue_cross"),
    ("bluecross", "blue_cross"),
    ("blue cross", "blue_cross"),
    ("blue shield of california", "blue_shield"),
    ("blue shield", "blue_shield"),
    ("optum/united", "united_healthcare"),
    ("united behavioral health", "united_healthcare"),
    ("unitedhealthcare", "united_healthcare"),
    ("united health care", "united_healthcare"),
    ("optum health", "optum"),
    ("optum", "optum"),
    ("aetna", "aetna"),
    ("cigna", "cigna"),
    ("kaiser", "kaiser"),
    ("magellan", "magellan"),
    ("out of network", "out_of_network"),
]


def map_insurance_name(name: str) -> str | None:
    """Map a JSON insurance company name to a DB enum value. Returns None if unrecognized."""
    lower = name.lower()
    for fragment, enum_val in INSURANCE_NAME_MAP:
        if fragment in lower:
            return enum_val
    return None


async def fetch_insurance_data(
    client: httpx.AsyncClient, url: str
) -> tuple[list[str], bool] | None:
    """
    Fetch a GoodTherapy profile and extract insurance_providers and sliding_scale
    from the __NEXT_DATA__ JSON blob.

    Returns (insurance_enum_values, sliding_scale) or None on error.
    """
    try:
        resp = await client.get(url, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script:
            return None
        profile = json.loads(script.string)["props"]["pageProps"]["profile"]
    except Exception as exc:
        logger.warning("Fetch error for %s: %s", url, exc)
        return None

    sliding_scale: bool = bool(profile.get("slidingScale", False))

    raw_list: list[str] = profile.get("insuranceCompaniesList") or []
    providers: list[str] = []
    for name in raw_list:
        mapped = map_insurance_name(name)
        if mapped and mapped not in providers:
            providers.append(mapped)

    return providers, sliding_scale


async def main() -> None:
    settings = get_settings()
    db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")

    pool = await asyncpg.create_pool(db_url, ssl="require", min_size=2, max_size=5)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, source_url FROM therapists WHERE source = 'good_therapy' AND is_active = TRUE"
        )
    logger.info("Found %d GoodTherapy therapists to enrich", len(rows))

    sem = asyncio.Semaphore(CONCURRENCY)
    counters = {"updated": 0, "no_data": 0, "error": 0}

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
    ) as client:

        async def process(row: asyncpg.Record) -> None:
            async with sem:
                result = await fetch_insurance_data(client, row["source_url"])
                if result is None:
                    counters["error"] += 1
                    return

                providers, sliding_scale = result
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE therapists
                        SET insurance_providers = $1::insurance_provider[],
                            sliding_scale = $2,
                            last_updated = NOW()
                        WHERE id = $3
                        """,
                        providers,
                        sliding_scale,
                        row["id"],
                    )
                if providers or sliding_scale:
                    counters["updated"] += 1
                    logger.debug(
                        "%s: insurance=%s sliding_scale=%s",
                        row["source_url"].split("/")[-1],
                        providers,
                        sliding_scale,
                    )
                else:
                    counters["no_data"] += 1

        await asyncio.gather(*[process(r) for r in rows])

    await pool.close()
    logger.info(
        "Done — updated=%d no_data=%d error=%d",
        counters["updated"],
        counters["no_data"],
        counters["error"],
    )


if __name__ == "__main__":
    asyncio.run(main())
