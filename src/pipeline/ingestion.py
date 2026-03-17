"""
Ingestion pipeline — orchestrates scraping → cleaning → embedding → storage.

Run this on a schedule (weekly) to keep therapist data fresh.
Supports incremental updates: only re-embeds profiles that changed.

Usage:
    python scripts/run_scraper.py --source psychology_today --max 500
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from src.pipeline.cleaner import DataCleaner
from src.pipeline.embeddings import EmbeddingPipeline
from src.scrapers.psychology_today import PsychologyTodayScraper
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

    async def run_psychology_today(
        self,
        max_therapists: int = 500,
        max_pages: int = 25,
    ) -> IngestionStats:
        stats = IngestionStats(source="psychology_today")

        # Stage 1: Scrape
        scraper = PsychologyTodayScraper()
        raw_profiles = await scraper.run(
            max_therapists=max_therapists,
            max_pages=max_pages,
        )
        stats.scraped = len(raw_profiles)

        # Stage 2: Clean and validate
        cleaned_profiles = []
        for profile in raw_profiles:
            cleaned = self._cleaner.clean(profile)
            if cleaned:
                cleaned_profiles.append(cleaned)
        stats.cleaned = len(cleaned_profiles)
        logger.info("Cleaned %d/%d profiles", stats.cleaned, stats.scraped)

        # Stage 3: Generate embeddings (batched)
        profile_embeddings = await self._embedder.embed_profiles(cleaned_profiles)
        stats.embedded = sum(1 for _, emb in profile_embeddings if emb)

        # Stage 4: Store in database
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

    async def run_open_path(
        self,
        max_therapists: int = 5000,
    ) -> IngestionStats:
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
    ) -> IngestionStats:
        """
        Scrape GoodTherapy and ingest into the database.

        ``cities``: optional list of city slugs (e.g. ["los-angeles", "san-francisco"])
        to restrict URL collection — useful for quick dev runs.  When None, all
        ~189 CA cities are walked.
        """
        stats = IngestionStats(source="good_therapy")

        # Use the scraper's run() so the httpx client is properly initialized.
        # For small batches we restrict to specific cities to avoid crawling all 189.
        scraper = GoodTherapyScraper()

        if cities:
            # Fast path: collect URLs only from the specified cities
            import httpx
            raw_profiles = []
            async with httpx.AsyncClient(
                headers={"User-Agent": "Mozilla/5.0 (compatible; TherapizeBot/1.0)"},
                follow_redirects=True,
                timeout=30,
            ) as client:
                scraper._client = client
                profile_urls: list[str] = []
                for city_slug in cities:
                    city_url = f"{scraper.base_url}/therapists/ca/{city_slug}"
                    city_urls = await scraper._collect_city_profile_urls(
                        city_url, max_pages_per_city=max_pages
                    )
                    profile_urls.extend(city_urls)
                    if len(profile_urls) >= max_therapists:
                        break

                for url in profile_urls[:max_therapists]:
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
