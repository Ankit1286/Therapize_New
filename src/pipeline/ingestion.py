"""
Ingestion pipeline — orchestrates scraping → cleaning → embedding → storage.

Run this on a schedule (weekly) to keep therapist data fresh.
Supports incremental updates: only re-embeds profiles that changed.

Usage:
    python scripts/ingest_small.py    # ~5 from each source (dev)
    python scripts/run_ingestion.py   # full scale
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.pipeline.cleaner import DataCleaner
from src.pipeline.embeddings import EmbeddingPipeline
from src.scrapers.open_path import OpenPathScraper
from src.scrapers.good_therapy import GoodTherapyScraper
from src.storage.database import TherapistRepository, init_db

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    source: str
    scraped: int = 0
    cleaned: int = 0
    embedded: int = 0
    stored: int = 0
    failed: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)

    def log(self) -> None:
        """Log a summary of the ingestion run: counts for each pipeline stage and elapsed time."""
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        logger.info(
            "[%s] Ingestion complete: scraped=%d, cleaned=%d, embedded=%d, "
            "stored=%d, failed=%d, elapsed=%.1fs",
            self.source, self.scraped, self.cleaned, self.embedded,
            self.stored, self.failed, elapsed,
        )


class IngestionPipeline:
    """
    Full ingestion pipeline: scrape → clean → embed → store.

    Design: each stage is independent and can be run separately.
    Failures in one profile don't stop the whole pipeline.
    """

    def __init__(self):
        self._cleaner = DataCleaner()
        self._embedder = EmbeddingPipeline()
        self._repository = TherapistRepository()

    async def run_open_path(
        self,
        max_therapists: int = 5000,
    ) -> IngestionStats:
        """
        Scrape OpenPath and ingest profiles into the database.

        Runs the full pipeline: scrape → clean → embed → upsert.
        Returns IngestionStats with counts for each stage.
        """
        stats = IngestionStats(source="open_path")

        scraper = OpenPathScraper()
        raw_profiles = await scraper.run(max_therapists=max_therapists)
        stats.scraped = len(raw_profiles)

        cleaned_profiles = []
        for profile in raw_profiles:
            cleaned = self._cleaner.clean(profile)
            if cleaned:
                cleaned_profiles.append(cleaned)
        stats.cleaned = len(cleaned_profiles)
        logger.info("Cleaned %d/%d profiles", stats.cleaned, stats.scraped)

        profile_embeddings = await self._embedder.embed_profiles(cleaned_profiles)
        stats.embedded = sum(1 for _, emb in profile_embeddings if emb)

        for profile, embedding in profile_embeddings:
            if not embedding:
                stats.failed += 1
                continue
            try:
                await self._repository.upsert(profile, embedding)
                stats.stored += 1
            except Exception as exc:
                logger.warning("Failed to store %s: %s", profile.name, exc)
                stats.failed += 1

        stats.log()
        return stats

    async def run_good_therapy(
        self,
        max_therapists: int = 5000,
        max_pages: int = 100,
        cities: list[str] | None = None,
        max_per_city: int | None = None,
    ) -> IngestionStats:
        """
        Scrape GoodTherapy and ingest into the database.

        ``cities``: optional list of city slugs (e.g. ["los-angeles", "san-francisco"])
        to restrict URL collection — useful for quick dev runs.  When None, all
        ~189 CA cities are walked.

        ``max_per_city``: cap the number of therapists fetched per city, ensuring
        no single city dominates the dataset. Ignored when cities is None.
        """
        stats = IngestionStats(source="good_therapy")

        scraper = GoodTherapyScraper()

        if cities:
            import httpx
            raw_profiles = []
            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (compatible; TherapizeBot/1.0)"},
                follow_redirects=True,
                timeout=30,
            ) as client:
                scraper._client = client
                for city_slug in cities:
                    if len(raw_profiles) >= max_therapists:
                        break
                    city_url = f"{scraper.base_url}/therapists/ca/{city_slug}"
                    city_urls = await scraper._collect_city_profile_urls(
                        city_url, max_pages_per_city=max_pages
                    )
                    # Apply per-city cap if set
                    city_limit = max_per_city if max_per_city else len(city_urls)
                    city_urls = city_urls[:city_limit]
                    logger.info(
                        "GoodTherapy: fetching %d profiles from %s",
                        len(city_urls), city_slug,
                    )
                    for url in city_urls:
                        if len(raw_profiles) >= max_therapists:
                            break
                        try:
                            r = await client.get(url)
                            profile = await scraper.parse_therapist_profile(url, r.text)
                            if profile:
                                raw_profiles.append(profile)
                        except Exception as exc:
                            logger.warning("Failed to fetch %s: %s", url, exc)
        else:
            # Full run: walk all cities via the standard scraper.run() flow
            raw_profiles = await scraper.run(
                max_therapists=max_therapists,
                max_pages=max_pages,
            )

        stats.scraped = len(raw_profiles)

        cleaned_profiles = []
        for profile in raw_profiles:
            cleaned = self._cleaner.clean(profile)
            if cleaned:
                cleaned_profiles.append(cleaned)
        stats.cleaned = len(cleaned_profiles)
        logger.info("Cleaned %d/%d profiles", stats.cleaned, stats.scraped)

        profile_embeddings = await self._embedder.embed_profiles(cleaned_profiles)
        stats.embedded = sum(1 for _, emb in profile_embeddings if emb)

        for profile, embedding in profile_embeddings:
            if not embedding:
                stats.failed += 1
                continue
            try:
                await self._repository.upsert(profile, embedding)
                stats.stored += 1
            except Exception as exc:
                logger.warning("Failed to store %s: %s", profile.name, exc)
                stats.failed += 1

        stats.log()
        return stats
