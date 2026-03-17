"""
Abstract base scraper with production-grade reliability features.

Every scraper inherits from here and gets:
- Exponential backoff retry
- Rate limiting (polite crawling)
- Robots.txt compliance
- Structured error handling
- Progress tracking
"""
import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from src.config import get_settings
from src.models.therapist import TherapistProfile

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class ScraperStats:
    """Tracks scraping run statistics."""
    started_at: datetime = field(default_factory=datetime.utcnow)
    pages_scraped: int = 0
    therapists_found: int = 0
    therapists_saved: int = 0
    errors: int = 0
    skipped: int = 0

    def log_summary(self) -> None:
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        logger.info(
            "Scraper summary: pages=%d, found=%d, saved=%d, errors=%d, "
            "skipped=%d, elapsed=%.1fs",
            self.pages_scraped, self.therapists_found, self.therapists_saved,
            self.errors, self.skipped, elapsed,
        )


class BaseScraper(ABC):
    """
    Abstract base for all platform scrapers.

    Key features:
    1. Robots.txt compliance — check before scraping
    2. Exponential backoff — retry transient failures
    3. Rate limiting — polite delay between requests
    4. Structured output — always returns TherapistProfile objects
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._robots_parser: RobotFileParser | None = None
        self._stats = ScraperStats()
        self._last_request_time = 0.0

    @property
    @abstractmethod
    def base_url(self) -> str:
        """Root URL of the platform."""
        ...

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Identifier for the data source (stored in therapist.source)."""
        ...

    @abstractmethod
    async def scrape_therapist_urls(self, max_pages: int = 100) -> list[str]:
        """Collect all therapist profile URLs to scrape."""
        ...

    @abstractmethod
    async def parse_therapist_profile(self, url: str, html: str) -> TherapistProfile | None:
        """Parse a single therapist profile page into a TherapistProfile."""
        ...

    async def run(
        self,
        max_therapists: int = 1000,
        max_pages: int = 50,
    ) -> list[TherapistProfile]:
        """
        Main scraping loop. Returns list of parsed therapist profiles.

        Flow:
        1. Check robots.txt
        2. Collect profile URLs (paginated listing pages)
        3. For each URL: fetch + parse with retry
        4. Return validated profiles
        """
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.scraper_user_agent},
            timeout=settings.scraper_timeout_seconds,
            follow_redirects=True,
        ) as client:
            self._client = client

            # Step 1: Robots.txt compliance
            if not await self._check_robots():
                logger.warning(
                    "%s: robots.txt disallows scraping. Skipping.", self.source_name
                )
                return []

            # Step 2: Collect profile URLs
            logger.info("%s: collecting profile URLs...", self.source_name)
            profile_urls = await self.scrape_therapist_urls(max_pages=max_pages)
            profile_urls = profile_urls[:max_therapists]
            logger.info("%s: found %d profile URLs", self.source_name, len(profile_urls))

            # Step 3: Fetch and parse each profile
            profiles = []
            for url in profile_urls:
                try:
                    html = await self._fetch_with_retry(url)
                    if html is None:
                        continue
                    profile = await self.parse_therapist_profile(url, html)
                    if profile:
                        profiles.append(profile)
                        self._stats.therapists_found += 1
                        if self._stats.therapists_found % 10 == 0:
                            logger.info(
                                "%s: scraped %d/%d profiles",
                                self.source_name,
                                self._stats.therapists_found,
                                len(profile_urls),
                            )
                    else:
                        self._stats.skipped += 1
                except Exception as exc:
                    self._stats.errors += 1
                    logger.warning(
                        "%s: failed to parse %s: %s", self.source_name, url, exc
                    )

            self._stats.log_summary()
            return profiles

    async def _fetch_with_retry(self, url: str) -> str | None:
        """
        Fetch a URL with exponential backoff retry.

        Retry on: 429 (rate limited), 503 (overloaded), connection errors
        Don't retry: 404 (not found), 403 (forbidden)
        """
        await self._rate_limit()

        for attempt in range(settings.scraper_max_retries):
            try:
                response = await self._client.get(url)
                self._stats.pages_scraped += 1

                if response.status_code == 200:
                    return response.text

                if response.status_code in (404, 410):
                    logger.debug("URL not found: %s", url)
                    return None

                if response.status_code in (403, 401):
                    logger.warning("Access denied: %s", url)
                    return None

                if response.status_code in (429, 503, 502):
                    wait_time = self._backoff(attempt)
                    logger.warning(
                        "Rate limited (%d) on %s, waiting %.1fs (attempt %d/%d)",
                        response.status_code, url, wait_time,
                        attempt + 1, settings.scraper_max_retries,
                    )
                    await asyncio.sleep(wait_time)

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                wait_time = self._backoff(attempt)
                logger.warning(
                    "Network error on %s: %s. Retrying in %.1fs",
                    url, exc, wait_time
                )
                await asyncio.sleep(wait_time)

        logger.error("All retries exhausted for %s", url)
        return None

    async def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.monotonic() - self._last_request_time
        min_delay = settings.scraper_delay_seconds
        if elapsed < min_delay:
            # Add small jitter to avoid synchronized requests
            jitter = random.uniform(0, 0.5)
            await asyncio.sleep(min_delay - elapsed + jitter)
        self._last_request_time = time.monotonic()

    async def _check_robots(self) -> bool:
        """Returns True if robots.txt allows scraping."""
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            response = await self._client.get(robots_url)
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
                return parser.can_fetch(settings.scraper_user_agent, self.base_url)
        except Exception as exc:
            logger.warning("Could not check robots.txt: %s — proceeding", exc)
        return True  # If robots.txt unreachable, assume allowed

    @staticmethod
    def _backoff(attempt: int, base: float = 2.0, max_wait: float = 60.0) -> float:
        """Exponential backoff: 2, 4, 8, ... seconds with jitter."""
        wait = min(base ** (attempt + 1), max_wait)
        jitter = random.uniform(0, wait * 0.1)
        return wait + jitter
