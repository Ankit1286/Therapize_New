"""
OpenPath Collective scraper for California therapist profiles.

OpenPath exposes therapist data through Algolia search rather than HTML pages.
This scraper queries the Algolia API directly — faster and more reliable than
HTML scraping, and uses structured JSON responses so no HTML parsing is needed.

Key design decisions:
- Does NOT extend BaseScraper: different transport pattern (JSON API, not HTML).
  Shares only the TherapistProfile output contract with other scrapers.
- Fetches the taxonomy ID→label map once from the OpenPath page JS, then caches
  it for the lifetime of the scraper run. Taxonomy IDs are stable across pages.
- Insurance: OpenPath is a sliding-scale platform ($50–$80/session flat).
  Therapists do not accept insurance. We record SELF_PAY and set sliding_scale=False
  because the fee is fixed (not sliding) — just affordable.
- Bio: profiles have no free-text bio. We construct a descriptive sentence from
  structured fields and store it in approach_description, not bio.

Algolia config sourced from openpathcollective.org page JS (tmAlgolia object):
  Application ID : TYXHARJ0OS
  Search API key : 05c557da6f2dc9d5685af66709f0e1cd
  Index          : prod_searchable_posts
"""
import asyncio
import json
import logging
import random
import re
import time
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.config import get_settings
from src.models.therapist import (
    InsuranceProvider,
    SessionFormat,
    TherapistLocation,
    TherapistProfile,
    TherapistSpecialization,
    TherapyModality,
)
from src.scrapers.base import ScraperStats

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Algolia connection constants ───────────────────────────────────────────────
_ALGOLIA_APP_ID = "TYXHARJ0OS"
_ALGOLIA_API_KEY = "05c557da6f2dc9d5685af66709f0e1cd"
_ALGOLIA_INDEX = "prod_searchable_posts"
_ALGOLIA_ENDPOINT = (
    f"https://{_ALGOLIA_APP_ID}-dsn.algolia.net/1/indexes/*/queries"
)
_ALGOLIA_HEADERS = {
    "X-Algolia-Application-Id": _ALGOLIA_APP_ID,
    "X-Algolia-API-Key": _ALGOLIA_API_KEY,
    "Referer": "https://openpathcollective.org/",
    "Origin": "https://openpathcollective.org",
    "Content-Type": "application/json",
}
_OPENPATH_TAXONOMY_URL = "https://openpathcollective.org/find-a-therapist/"
_HITS_PER_PAGE = 200
_CA_FILTER = "post_type:therapist"
_CA_FACET_FILTER = '[["state:California"]]'

# ── Orientation (therapy approach) label → TherapyModality enum ───────────────
# Keys are lowercase substrings to match against Algolia taxonomy labels.
# Ordered from most-specific to least-specific to avoid false partial matches.
ORIENTATION_LABEL_MAP: dict[str, TherapyModality] = {
    "acceptance commitment": TherapyModality.ACT,
    "cognitive behavioral": TherapyModality.CBT,
    "cbt": TherapyModality.CBT,
    "dialectical behavior": TherapyModality.DBT,
    "dbt": TherapyModality.DBT,
    "emdr": TherapyModality.EMDR,
    "cognitive processing": TherapyModality.CPT,
    "emotionally focused": TherapyModality.EFT,
    "gottman": TherapyModality.GOTTMAN,
    "exposure": TherapyModality.EXPOSURE,
    "narrative": TherapyModality.NARRATIVE,
    "solution-focused": TherapyModality.SOLUTION_FOCUSED,
    "solution focused": TherapyModality.SOLUTION_FOCUSED,
    "motivational interviewing": TherapyModality.MOTIVATIONAL,
    "family systems": TherapyModality.FAMILY_SYSTEMS,
    "trauma-informed": TherapyModality.TRAUMA_INFORMED,
    "trauma informed": TherapyModality.TRAUMA_INFORMED,
    "psychoanalytic": TherapyModality.PSYCHOANALYTIC,
    "psychodynamic": TherapyModality.PSYCHODYNAMIC,
    "humanistic": TherapyModality.HUMANISTIC,
    "mindfulness": TherapyModality.MINDFULNESS,
    "somatic": TherapyModality.SOMATIC,
    "integrative": TherapyModality.INTEGRATIVE,
    "play therapy": TherapyModality.PLAY_THERAPY,
    "art therapy": TherapyModality.ART_THERAPY,
}

# ── Specialty label → TherapistSpecialization enum ────────────────────────────
SPECIALTY_LABEL_MAP: dict[str, TherapistSpecialization] = {
    "addiction": TherapistSpecialization.ADDICTION,
    "anxiety": TherapistSpecialization.ANXIETY,
    "depression": TherapistSpecialization.DEPRESSION,
    "trauma": TherapistSpecialization.TRAUMA,
    "ptsd": TherapistSpecialization.PTSD,
    "grief": TherapistSpecialization.GRIEF,
    "adhd": TherapistSpecialization.ADHD,
    "ocd": TherapistSpecialization.OCD,
    "eating disorder": TherapistSpecialization.EATING_DISORDERS,
    "eating": TherapistSpecialization.EATING_DISORDERS,
    "relationship": TherapistSpecialization.RELATIONSHIP,
    "couples": TherapistSpecialization.COUPLES,
    "family": TherapistSpecialization.FAMILY,
    "lgbtq": TherapistSpecialization.LGBTQ,
    "life transitions": TherapistSpecialization.LIFE_TRANSITIONS,
    "life transition": TherapistSpecialization.LIFE_TRANSITIONS,
    "stress": TherapistSpecialization.STRESS,
    "self-esteem": TherapistSpecialization.SELF_ESTEEM,
    "self esteem": TherapistSpecialization.SELF_ESTEEM,
    "anger": TherapistSpecialization.ANGER,
    "sleep": TherapistSpecialization.SLEEP,
    "bipolar": TherapistSpecialization.BIPOLAR,
    "chronic pain": TherapistSpecialization.CHRONIC_PAIN,
    "career": TherapistSpecialization.CAREER,
    "children": TherapistSpecialization.CHILDREN,
    "adolescent": TherapistSpecialization.ADOLESCENTS,
    "personality disorder": TherapistSpecialization.PERSONALITY_DISORDERS,
}

# ── therapist_modality IDs (session types, not therapy approaches) ─────────────
# 734 = Individuals, 735 = Couples/Partners, 736 = Families
_SESSION_TYPE_IDS: dict[int, str] = {
    734: "individuals",
    735: "couples",
    736: "families",
}


class OpenPathScraper:
    """
    Scraper for OpenPath Collective therapist directory.

    Uses the Algolia search API that backs openpathcollective.org, so requests
    are JSON POST calls rather than HTML fetches. The taxonomy mapping (IDs to
    human-readable labels) is fetched once from the OpenPath page and cached.

    Usage::

        scraper = OpenPathScraper()
        profiles = await scraper.run(max_therapists=5000)
    """

    source_name: str = "open_path"

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._stats = ScraperStats()
        self._last_request_time = 0.0
        # Maps taxonomy_type → {id: label}
        self._taxonomy_cache: dict[str, dict[int, str]] = {}

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, max_therapists: int = 5000) -> list[TherapistProfile]:
        """
        Full scrape run. Returns up to *max_therapists* TherapistProfile objects.

        Steps:
        1. Fetch taxonomy map from OpenPath JS (once).
        2. Paginate through Algolia, building profiles from each hit.
        3. Stop when Algolia returns no more results or max_therapists reached.
        """
        async with httpx.AsyncClient(
            headers={
                "User-Agent": settings.scraper_user_agent,
                **_ALGOLIA_HEADERS,
            },
            timeout=settings.scraper_timeout_seconds,
            follow_redirects=True,
        ) as client:
            self._client = client

            logger.info("%s: fetching taxonomy map...", self.source_name)
            await self._load_taxonomy()

            logger.info(
                "%s: starting Algolia pagination (max=%d)...",
                self.source_name, max_therapists,
            )
            profiles = await self._paginate_algolia(max_therapists)

        self._stats.log_summary()
        return profiles

    # ── Algolia pagination ─────────────────────────────────────────────────────

    async def _paginate_algolia(
        self, max_therapists: int
    ) -> list[TherapistProfile]:
        """Paginate through all Algolia pages, converting hits to profiles."""
        profiles: list[TherapistProfile] = []
        page = 0

        while len(profiles) < max_therapists:
            await self._rate_limit()
            hits, nb_pages = await self._fetch_algolia_page(page)

            if not hits:
                logger.info(
                    "%s: no hits on page %d — stopping", self.source_name, page
                )
                break

            for hit in hits:
                if len(profiles) >= max_therapists:
                    break
                try:
                    profile = self._parse_hit(hit)
                    if profile:
                        profiles.append(profile)
                        self._stats.therapists_found += 1
                except Exception as exc:
                    self._stats.errors += 1
                    logger.warning(
                        "%s: failed to parse hit %s: %s",
                        self.source_name,
                        hit.get("post_id", "?"),
                        exc,
                    )

            self._stats.pages_scraped += 1
            if self._stats.therapists_found % 200 == 0 and self._stats.therapists_found:
                logger.info(
                    "%s: processed %d therapists (page %d/%d)",
                    self.source_name,
                    self._stats.therapists_found,
                    page + 1,
                    nb_pages,
                )

            page += 1
            if page >= nb_pages:
                logger.info(
                    "%s: reached last Algolia page (%d)", self.source_name, nb_pages
                )
                break

        logger.info(
            "%s: finished pagination — %d profiles collected",
            self.source_name, len(profiles),
        )
        return profiles

    async def _fetch_algolia_page(
        self, page: int
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Fetch one page of Algolia results for CA therapists.

        Returns (hits, nb_pages). On error, returns ([], 0).
        """
        payload = {
            "requests": [
                {
                    "indexName": _ALGOLIA_INDEX,
                    "params": (
                        f"filters={_CA_FILTER}"
                        f"&facetFilters={_CA_FACET_FILTER}"
                        f"&page={page}"
                        f"&hitsPerPage={_HITS_PER_PAGE}"
                    ),
                }
            ]
        }

        for attempt in range(settings.scraper_max_retries):
            try:
                resp = await self._client.post(
                    _ALGOLIA_ENDPOINT,
                    content=json.dumps(payload),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    result = data["results"][0]
                    return result.get("hits", []), result.get("nbPages", 0)

                if resp.status_code in (429, 503, 502):
                    wait = self._backoff(attempt)
                    logger.warning(
                        "%s: Algolia rate limited (%d), page %d — waiting %.1fs",
                        self.source_name, resp.status_code, page, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                logger.warning(
                    "%s: Algolia unexpected status %d on page %d",
                    self.source_name, resp.status_code, page,
                )
                return [], 0

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                wait = self._backoff(attempt)
                logger.warning(
                    "%s: network error on page %d: %s — retrying in %.1fs",
                    self.source_name, page, exc, wait,
                )
                await asyncio.sleep(wait)

        logger.error(
            "%s: all retries exhausted for Algolia page %d", self.source_name, page
        )
        return [], 0

    # ── Hit → TherapistProfile ─────────────────────────────────────────────────

    def _parse_hit(self, hit: dict[str, Any]) -> TherapistProfile | None:
        """Convert a single Algolia hit dict to a TherapistProfile."""
        # Required fields — skip record if absent
        post_id = hit.get("post_id")
        name = (hit.get("post_title") or "").strip()
        permalink = hit.get("permalink", "")
        if not post_id or not name or not permalink:
            return None

        # Credentials — comma-separated string, e.g. "AMFT, APCC"
        raw_creds = hit.get("credentials") or ""
        credentials = [c.strip() for c in raw_creds.split(",") if c.strip()]

        # Location — use first location entry; OpenPath provides structured locs
        locations: list[dict] = hit.get("locations") or []
        location = self._parse_location(locations)
        if not location:
            return None  # Require at least a city

        # Session formats
        session_formats = self._parse_session_formats(hit, locations)

        # Fees — OpenPath platform fee: $50–$80 per session
        cost = hit.get("cost") or {}
        def _safe_int(v, default):
            try:
                return int(v) if v not in (None, "") else default
            except (ValueError, TypeError):
                return default

        fee_min = _safe_int(cost.get("individual_min"), 50)
        fee_max = _safe_int(cost.get("individual_max"), 80)

        # Taxonomy-driven fields
        tax_ids: dict[str, list[int]] = hit.get("tax_ids") or {}
        modalities = self._resolve_modalities(tax_ids.get("orientation") or [])
        specializations = self._resolve_specializations(
            tax_ids.get("specialty") or []
        )
        languages = self._resolve_labels("language", tax_ids.get("language") or [])
        populations = self._resolve_populations(
            tax_ids.get("age") or [],
            tax_ids.get("therapist_modality") or [],
            tax_ids.get("client_gender") or [],
        )

        # Accepting new clients
        accepting_new = bool(hit.get("new_clients", 1))

        # Constructed approach description (no free-text bio on OpenPath)
        approach_description = self._build_approach_description(
            name=name,
            credentials=credentials,
            city=location.city,
            populations=populations,
            specializations=specializations,
            modalities=modalities,
        )

        # Profile completeness heuristic
        completeness = self._compute_completeness(
            modalities, specializations, credentials, languages, fee_min
        )

        return TherapistProfile(
            source=self.source_name,
            source_url=permalink,  # type: ignore[arg-type]
            source_id=str(post_id),
            name=name,
            credentials=credentials,
            modalities=modalities,
            specializations=specializations,
            populations_served=populations,
            languages=languages,
            location=location,
            session_formats=session_formats,
            # OpenPath therapists do not accept insurance — platform is self-pay
            accepts_insurance=[InsuranceProvider.SELF_PAY],
            sliding_scale=False,  # Fixed-fee platform, not sliding scale
            fee_min=int(fee_min) if fee_min is not None else None,
            fee_max=int(fee_max) if fee_max is not None else None,
            accepting_new_clients=accepting_new,
            bio="",
            approach_description=approach_description,
            profile_completeness=completeness,
        )

    # ── Location helpers ───────────────────────────────────────────────────────

    def _parse_location(
        self, locations: list[dict[str, Any]]
    ) -> TherapistLocation | None:
        """Extract the primary location from the Algolia locations list."""
        if not locations:
            return None

        # Prefer an in-person CA location; fall back to the first entry
        primary = next(
            (loc for loc in locations if loc.get("in_person") and loc.get("state")),
            locations[0],
        )
        city = (primary.get("city") or "").strip()
        if not city:
            return None

        return TherapistLocation(
            city=city,
            state="CA",
            zip_code=str(primary["zip"]) if primary.get("zip") else None,
            latitude=primary.get("lat"),
            longitude=primary.get("lng"),
        )

    def _parse_session_formats(
        self, hit: dict[str, Any], locations: list[dict[str, Any]]
    ) -> list[SessionFormat]:
        """
        Determine session formats from the hit.

        OpenPath uses two fields:
        - ``online``: list of states where online sessions are available
        - ``in_person``: list of states where in-person sessions are offered
        Also check individual location entries for granular online/in_person flags.
        """
        has_online = bool(hit.get("online"))
        has_in_person = bool(hit.get("in_person"))

        # Cross-check with location-level flags if top-level fields are absent
        if not has_online and not has_in_person and locations:
            has_online = any(loc.get("online") for loc in locations)
            has_in_person = any(loc.get("in_person") for loc in locations)

        if has_online and has_in_person:
            return [SessionFormat.BOTH]
        if has_online:
            return [SessionFormat.TELEHEALTH]
        return [SessionFormat.IN_PERSON]

    # ── Taxonomy resolution ────────────────────────────────────────────────────

    def _resolve_modalities(self, orientation_ids: list[int]) -> list[TherapyModality]:
        """Map orientation taxonomy IDs to TherapyModality enum values."""
        result: list[TherapyModality] = []
        orientation_map = self._taxonomy_cache.get("orientation", {})
        for tid in orientation_ids:
            entry = orientation_map.get(tid, {})
            label = (entry.get("label", "") if isinstance(entry, dict) else str(entry)).lower()
            if not label:
                continue
            for keyword, modality in ORIENTATION_LABEL_MAP.items():
                if keyword in label:
                    if modality not in result:
                        result.append(modality)
                    break
        return result

    def _resolve_specializations(
        self, specialty_ids: list[int]
    ) -> list[TherapistSpecialization]:
        """Map specialty taxonomy IDs to TherapistSpecialization enum values."""
        result: list[TherapistSpecialization] = []
        specialty_map = self._taxonomy_cache.get("specialty", {})
        for tid in specialty_ids:
            entry = specialty_map.get(tid, {})
            label = (entry.get("label", "") if isinstance(entry, dict) else str(entry)).lower()
            if not label:
                continue
            for keyword, spec in SPECIALTY_LABEL_MAP.items():
                if keyword in label:
                    if spec not in result:
                        result.append(spec)
                    break
        return result

    def _resolve_labels(
        self, taxonomy_type: str, ids: list[int]
    ) -> list[str]:
        """Return human-readable labels for a list of taxonomy IDs."""
        tax_map = self._taxonomy_cache.get(taxonomy_type, {})
        result = []
        for tid in ids:
            entry = tax_map.get(tid)
            if entry is None:
                continue
            result.append(entry.get("label", str(entry)) if isinstance(entry, dict) else str(entry))
        return result

    def _resolve_populations(
        self,
        age_ids: list[int],
        modality_ids: list[int],
        gender_ids: list[int],
    ) -> list[str]:
        """
        Build populations_served list from age, session-type, and gender IDs.

        Note: ``therapist_modality`` in OpenPath refers to WHO they work with
        (Individuals / Couples / Families), not the therapy approach used.
        """
        populations: list[str] = []

        # Session type populations
        for tid in modality_ids:
            label = _SESSION_TYPE_IDS.get(tid)
            if label and label not in populations:
                populations.append(label)

        # Age groups
        age_map = self._taxonomy_cache.get("age", {})
        for tid in age_ids:
            entry = age_map.get(tid)
            label = (entry.get("label") if isinstance(entry, dict) else entry) if entry else None
            if label and label not in populations:
                populations.append(label)

        # Gender identities served
        gender_map = self._taxonomy_cache.get("client_gender", {})
        for tid in gender_ids:
            entry = gender_map.get(tid)
            label = (entry.get("label") if isinstance(entry, dict) else entry) if entry else None
            if label and label not in populations:
                populations.append(label)

        return populations

    # ── Taxonomy loading ──────────────────────────────────────────────────────

    async def _load_taxonomy(self) -> None:
        """
        Fetch and cache the OpenPath taxonomy map.

        The taxonomy is embedded as a JS object ``tmAlgolia`` in the
        find-a-therapist page. We extract it with BeautifulSoup + regex,
        then parse the JSON. Falls back to an empty cache on failure — the
        scraper will still produce records, just with empty mapped fields.
        """
        try:
            resp = await self._client.get(
                _OPENPATH_TAXONOMY_URL,
                headers={"User-Agent": settings.scraper_user_agent},
            )
            if resp.status_code != 200:
                logger.warning(
                    "%s: taxonomy page returned %d — continuing without taxonomy",
                    self.source_name, resp.status_code,
                )
                return

            soup = BeautifulSoup(resp.text, "html.parser")
            self._taxonomy_cache = self._parse_taxonomy_from_html(soup)
            logger.info(
                "%s: taxonomy loaded — %d types",
                self.source_name, len(self._taxonomy_cache),
            )

        except Exception as exc:
            logger.warning(
                "%s: failed to load taxonomy: %s — continuing without it",
                self.source_name, exc,
            )

    def _parse_taxonomy_from_html(
        self, soup: BeautifulSoup
    ) -> dict[str, dict[int, str]]:
        """
        Extract the tmAlgolia taxonomy JSON from inline page script tags.

        The page embeds something like:
            var tmAlgolia = {..., "taxonomies": {"orientation": {328: "Psychodynamic", ...}}}

        We find the script block containing ``tmAlgolia``, extract the JSON
        object with a balanced-brace approach, then parse it.
        """
        taxonomy: dict[str, dict[int, str]] = {}

        for script in soup.find_all("script"):
            src = script.string or ""
            if "tmAlgolia" not in src:
                continue

            # Try to pull out the full JS object assigned to tmAlgolia
            match = re.search(r"var\s+tmAlgolia\s*=\s*(\{)", src)
            if not match:
                # Also try window.tmAlgolia or just tmAlgolia = {
                match = re.search(r"tmAlgolia\s*=\s*(\{)", src)
            if not match:
                continue

            start = match.start(1)
            # Walk forward counting braces to find the closing }
            json_str = self._extract_balanced_braces(src, start)
            if not json_str:
                continue

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # The JS object may use single quotes or unquoted keys;
                # attempt a lenient extraction of just the taxonomies sub-key
                data = self._extract_taxonomies_lenient(src)

            # Navigate to the taxonomies map: may be at top-level or nested
            raw_taxonomies: dict | None = None
            if isinstance(data, dict):
                raw_taxonomies = (
                    data.get("taxonomies")
                    or data.get("tax")
                    or data.get("facets")
                    or data  # The whole object might be the taxonomy map
                )

            if not raw_taxonomies:
                continue

            # Convert: {"orientation": {"328": "Psychodynamic", ...}, ...}
            for tax_type, id_label_map in raw_taxonomies.items():
                if not isinstance(id_label_map, dict):
                    continue
                taxonomy[tax_type] = {
                    int(k): v
                    for k, v in id_label_map.items()
                    if str(k).isdigit()
                }

            if taxonomy:
                break  # Found and parsed successfully

        if not taxonomy:
            logger.warning(
                "%s: could not parse tmAlgolia from page — taxonomy will be empty",
                self.source_name,
            )
        return taxonomy

    @staticmethod
    def _extract_balanced_braces(src: str, start: int) -> str | None:
        """
        Extract a JSON object starting at ``start`` in ``src`` by counting
        balanced braces. Returns the raw JSON string or None on failure.
        """
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(src[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return src[start : i + 1]
        return None

    @staticmethod
    def _extract_taxonomies_lenient(src: str) -> dict:
        """
        Fallback parser: look for individual taxonomy arrays in the JS source
        using simple regex. This covers cases where the JS is not valid JSON
        (e.g., unquoted keys, trailing commas).
        """
        result: dict[str, dict[str, str]] = {}
        # Match patterns like: "orientation":{123:"Label",456:"Label2"}
        for tax_match in re.finditer(
            r'"(\w+)"\s*:\s*\{([^}]+)\}', src
        ):
            tax_type = tax_match.group(1)
            entries_str = tax_match.group(2)
            entries: dict[str, str] = {}
            for entry in re.finditer(r'(\d+)\s*:\s*"([^"]*)"', entries_str):
                entries[entry.group(1)] = entry.group(2)
            if entries:
                result[tax_type] = entries
        return result

    # ── Constructed bio ───────────────────────────────────────────────────────

    @staticmethod
    def _build_approach_description(
        name: str,
        credentials: list[str],
        city: str,
        populations: list[str],
        specializations: list[TherapistSpecialization],
        modalities: list[TherapyModality],
    ) -> str:
        """
        Build a descriptive sentence from structured fields.

        OpenPath profiles have no free-text bio, so we construct a natural
        description from the available structured data for use in embeddings
        and BM25 indexing.
        """
        parts: list[str] = []

        cred_str = ", ".join(credentials) if credentials else "therapist"
        parts.append(f"{name} is a {cred_str} in {city}, California.")

        if populations:
            pop_str = ", ".join(populations[:4])  # Limit to avoid run-on sentences
            parts.append(f"They work with {pop_str}.")

        if specializations:
            spec_labels = [s.value.replace("_", " ") for s in specializations[:5]]
            parts.append(f"Areas of focus include {', '.join(spec_labels)}.")

        if modalities:
            mod_labels = [m.value.replace("_", " ") for m in modalities[:5]]
            parts.append(
                f"Therapeutic approaches include {', '.join(mod_labels)}."
            )

        return " ".join(parts)

    # ── Completeness scoring ──────────────────────────────────────────────────

    @staticmethod
    def _compute_completeness(
        modalities: list,
        specializations: list,
        credentials: list,
        languages: list,
        fee_min: Any,
    ) -> float:
        """
        Heuristic completeness score in [0, 1].

        OpenPath profiles are highly structured so we weigh structured fields
        rather than free-text length.
        """
        score = 0.0
        if modalities:
            score += 0.30
        if specializations:
            score += 0.30
        if credentials:
            score += 0.20
        if languages:
            score += 0.10
        if fee_min is not None:
            score += 0.10
        return round(score, 2)

    # ── Rate limiting + backoff ───────────────────────────────────────────────

    async def _rate_limit(self) -> None:
        """Enforce a short but polite delay between Algolia API calls."""
        # OpenPath's Algolia is an API endpoint so delay is minimal (0.1s),
        # but we still respect it to avoid hammering the service.
        min_delay = 0.1
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < min_delay:
            jitter = random.uniform(0, 0.05)
            await asyncio.sleep(min_delay - elapsed + jitter)
        self._last_request_time = time.monotonic()

    @staticmethod
    def _backoff(attempt: int, base: float = 2.0, max_wait: float = 60.0) -> float:
        """Exponential backoff: 2, 4, 8, ... with jitter."""
        wait = min(base ** (attempt + 1), max_wait)
        return wait + random.uniform(0, wait * 0.1)
