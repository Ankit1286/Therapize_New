"""
GoodTherapy scraper for California therapist profiles.

GoodTherapy.org is a large therapist directory with ~189 California city pages.
This scraper walks the CA city index, paginates each city listing, deduplicates
profile URLs (therapists appear under multiple cities), then scrapes each profile.

Scraping strategy:
1. Fetch CA state index page → extract all /therapists/ca/{city} links.
2. For each city page, paginate with ?page=N until no new profile links found.
3. Deduplicate profile URLs across all cities.
4. Fetch and parse each individual profile page.

HTML structure notes:
- GoodTherapy restructures its markup periodically. All CSS selectors are
  centralized in the SELECTORS dict at the top of this file for easy updates.
- Profile data is primarily in ``div.mb-6`` sections. Each section is identified
  by a heading within it; we scan section headings to find the right one.
- Name + credentials: page title contains the full name + credential string.
- Bio: largest paragraph block in the page content area.
- City: inferred from the URL path segment (/therapists/ca/{city}/...).
"""
import json
import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from src.models.therapist import (
    InsuranceProvider,
    SessionFormat,
    TherapistLocation,
    TherapistProfile,
    TherapistSpecialization,
    TherapyModality,
)
from src.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# ── CSS selectors — update here first when GoodTherapy changes markup ──────────
SELECTORS: dict[str, str] = {
    # State index page
    "city_links": "a[href*='/therapists/ca/']",
    # City listing page
    "profile_links": "a[href*='/therapists/profile/']",
    "next_page": "a[rel='next'], a.pagination-next",
    # Profile page — primary content area
    "content_sections": "div.mb-6",
    "page_title": "title",
    "h1_name": "h1",
    # Profile page — bio prose divs (primary bio source)
    "bio_prose": "div[class*='whitespace-pre-line']",
    # Profile page — direct selectors (used as fallback)
    "fee_div": "div.profileTour_fitToContent, div.fee, div[class*='fee']",
    "accepting_clients": "div[class*='accepting'], span[class*='accepting']",
    "telehealth_badge": "div[class*='telehealth'], span[class*='telehealth'], div[class*='online']",
    # Breadcrumb for city extraction
    "breadcrumb": "nav[aria-label*='readcrumb'], ol.breadcrumb, ul.breadcrumb",
    "breadcrumb_links": "a[href*='/therapists/CA/'], a[href*='/therapists/ca/']",
}

# ── Insurance name → InsuranceProvider enum ────────────────────────────────────
INSURANCE_MAP: dict[str, InsuranceProvider] = {
    "blue shield": InsuranceProvider.BLUE_SHIELD,
    "blue cross": InsuranceProvider.BLUE_CROSS,
    "bluecross": InsuranceProvider.BLUE_CROSS,
    "blueshield": InsuranceProvider.BLUE_SHIELD,
    "aetna": InsuranceProvider.AETNA,
    "united": InsuranceProvider.UNITED,
    "unitedhealthcare": InsuranceProvider.UNITED,
    "cigna": InsuranceProvider.CIGNA,
    "kaiser": InsuranceProvider.KAISER,
    "magellan": InsuranceProvider.MAGELLAN,
    "optum": InsuranceProvider.OPTUM,
    "sliding scale": InsuranceProvider.SLIDING_SCALE,
    "sliding": InsuranceProvider.SLIDING_SCALE,
}

# ── Therapy approach label → TherapyModality enum ─────────────────────────────
# Keys are lowercase substrings matched against section text from profile pages.
MODALITY_MAP: dict[str, TherapyModality] = {
    "acceptance and commitment": TherapyModality.ACT,
    "acceptance commitment": TherapyModality.ACT,
    " act ": TherapyModality.ACT,
    "cognitive behavioral": TherapyModality.CBT,
    " cbt ": TherapyModality.CBT,
    "cognitive behaviour": TherapyModality.CBT,
    "dialectical behavior": TherapyModality.DBT,
    " dbt ": TherapyModality.DBT,
    "dialectical behaviour": TherapyModality.DBT,
    "emdr": TherapyModality.EMDR,
    "eye movement": TherapyModality.EMDR,
    "cognitive processing": TherapyModality.CPT,
    "emotionally focused": TherapyModality.EFT,
    " eft ": TherapyModality.EFT,
    "gottman": TherapyModality.GOTTMAN,
    "exposure therapy": TherapyModality.EXPOSURE,
    "exposure and response": TherapyModality.EXPOSURE,
    "narrative therapy": TherapyModality.NARRATIVE,
    "narrative": TherapyModality.NARRATIVE,
    "solution-focused": TherapyModality.SOLUTION_FOCUSED,
    "solution focused": TherapyModality.SOLUTION_FOCUSED,
    "motivational interviewing": TherapyModality.MOTIVATIONAL,
    "family systems": TherapyModality.FAMILY_SYSTEMS,
    "internal family": TherapyModality.FAMILY_SYSTEMS,
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

# ── Specialization label → TherapistSpecialization enum ───────────────────────
SPECIALIZATION_MAP: dict[str, TherapistSpecialization] = {
    "addiction": TherapistSpecialization.ADDICTION,
    "substance": TherapistSpecialization.ADDICTION,
    "anxiety": TherapistSpecialization.ANXIETY,
    "depression": TherapistSpecialization.DEPRESSION,
    "trauma": TherapistSpecialization.TRAUMA,
    "ptsd": TherapistSpecialization.PTSD,
    "post-traumatic": TherapistSpecialization.PTSD,
    "post traumatic": TherapistSpecialization.PTSD,
    "grief": TherapistSpecialization.GRIEF,
    "loss": TherapistSpecialization.GRIEF,
    "adhd": TherapistSpecialization.ADHD,
    "attention deficit": TherapistSpecialization.ADHD,
    "ocd": TherapistSpecialization.OCD,
    "obsessive": TherapistSpecialization.OCD,
    "eating disorder": TherapistSpecialization.EATING_DISORDERS,
    "eating": TherapistSpecialization.EATING_DISORDERS,
    "anorexia": TherapistSpecialization.EATING_DISORDERS,
    "bulimia": TherapistSpecialization.EATING_DISORDERS,
    "relationship": TherapistSpecialization.RELATIONSHIP,
    "couples": TherapistSpecialization.COUPLES,
    "marriage": TherapistSpecialization.COUPLES,
    "family": TherapistSpecialization.FAMILY,
    "lgbtq": TherapistSpecialization.LGBTQ,
    "lgbt": TherapistSpecialization.LGBTQ,
    "queer": TherapistSpecialization.LGBTQ,
    "life transitions": TherapistSpecialization.LIFE_TRANSITIONS,
    "life transition": TherapistSpecialization.LIFE_TRANSITIONS,
    "stress": TherapistSpecialization.STRESS,
    "burnout": TherapistSpecialization.STRESS,
    "self-esteem": TherapistSpecialization.SELF_ESTEEM,
    "self esteem": TherapistSpecialization.SELF_ESTEEM,
    "anger": TherapistSpecialization.ANGER,
    "sleep": TherapistSpecialization.SLEEP,
    "insomnia": TherapistSpecialization.SLEEP,
    "bipolar": TherapistSpecialization.BIPOLAR,
    "chronic pain": TherapistSpecialization.CHRONIC_PAIN,
    "chronic illness": TherapistSpecialization.CHRONIC_PAIN,
    "career": TherapistSpecialization.CAREER,
    "work": TherapistSpecialization.CAREER,
    "children": TherapistSpecialization.CHILDREN,
    "adolescent": TherapistSpecialization.ADOLESCENTS,
    "teen": TherapistSpecialization.ADOLESCENTS,
    "personality disorder": TherapistSpecialization.PERSONALITY_DISORDERS,
    "borderline": TherapistSpecialization.PERSONALITY_DISORDERS,
}

# ── Section heading keywords used to locate div.mb-6 content blocks ───────────
_SECTION_CHALLENGES = "challenges"           # "Client Challenges & Concerns"
_SECTION_APPROACHES = "approaches"           # "Therapeutic Approaches"
_SECTION_INSURANCE = "insurance"             # insurance / payment section
_SECTION_GROUPS = "groups"                   # "Groups I Work With"

# ── Credential abbreviation pattern ───────────────────────────────────────────
# Matches uppercase abbreviations at end of name string: LMFT, LCSW, PhD, etc.
_CRED_RE = re.compile(
    r"\b(LMFT|LCSW|MFT|LPCC|APCC|AMFT|PhD|PsyD|EdD|MD|MA|MS|MSW|BCBA|NCC|LPC"
    r"|LPCC-A|ACSW|CSW|LICSW|LISW|LPC-A|PMHNP|LADC|CADC|CATC|CADAC|MHC|CMHC"
    r"|CCMHC|LAC|LMC|LCPC|LMHC|LCMHC|LMHCA|BCPCC|RPT|ATR|CEDS|CSAT|CAADC)\b"
)

# ── Fee extraction pattern ─────────────────────────────────────────────────────
_FEE_RE = re.compile(r"\$\s*(\d{2,4})(?:\s*[-–]\s*\$?\s*(\d{2,4}))?")


class GoodTherapyScraper(BaseScraper):
    """
    Scraper for GoodTherapy.org California therapist directory.

    Extends BaseScraper for robots.txt compliance, retry logic, and rate limiting.
    Follows the two-phase pattern:
    1. ``scrape_therapist_urls`` — walk CA city pages to collect profile URLs.
    2. ``parse_therapist_profile`` — parse each individual profile page.

    The SELECTORS dict at the top of this file centralises all CSS selectors
    so that markup changes can be patched in one place.
    """

    @property
    def base_url(self) -> str:
        return "https://www.goodtherapy.org"

    @property
    def source_name(self) -> str:
        return "good_therapy"

    # ── Phase 1: URL collection ───────────────────────────────────────────────

    async def scrape_therapist_urls(self, max_pages: int = 100) -> list[str]:
        """
        Collect all CA therapist profile URLs from GoodTherapy.

        Steps:
        1. Fetch the CA state index to get all city listing URLs.
        2. For each city, paginate until no new profile links appear.
        3. Deduplicate and return the full URL list.

        ``max_pages`` caps the pagination depth per city (not total pages).
        """
        city_urls = await self._collect_city_urls()
        logger.info(
            "%s: found %d CA city pages", self.source_name, len(city_urls)
        )

        profile_urls: set[str] = set()
        for city_url in city_urls:
            city_profiles = await self._collect_city_profile_urls(
                city_url, max_pages_per_city=max_pages
            )
            before = len(profile_urls)
            profile_urls.update(city_profiles)
            added = len(profile_urls) - before
            logger.debug(
                "%s: %s → %d profiles (+%d new)",
                self.source_name, city_url, len(city_profiles), added,
            )

        logger.info(
            "%s: collected %d unique profile URLs",
            self.source_name, len(profile_urls),
        )
        return list(profile_urls)

    async def _collect_city_urls(self) -> list[str]:
        """
        Fetch the CA state index page and extract all city listing URLs.

        Returns URLs like https://www.goodtherapy.org/therapists/ca/los-angeles
        """
        state_url = f"{self.base_url}/therapists/ca"
        html = await self._fetch_with_retry(state_url)
        if not html:
            logger.error("%s: could not fetch CA state index", self.source_name)
            return []

        soup = BeautifulSoup(html, "html.parser")
        city_urls: list[str] = []
        seen: set[str] = set()

        for link in soup.select(SELECTORS["city_links"]):
            href = link.get("href", "")
            if not href:
                continue
            full = urljoin(self.base_url, href)
            parsed = urlparse(full)
            # Only accept paths like /therapists/ca/{city} (no profile links)
            path_parts = parsed.path.strip("/").split("/")
            if (
                len(path_parts) == 3
                and path_parts[0] == "therapists"
                and path_parts[1].lower() == "ca"
                and path_parts[2]  # non-empty city slug
                and "profile" not in path_parts[2]
                and full not in seen
            ):
                city_urls.append(full)
                seen.add(full)

        return city_urls

    async def _collect_city_profile_urls(
        self, city_url: str, max_pages_per_city: int = 100
    ) -> list[str]:
        """
        Paginate through a city listing page, collecting therapist profile URLs.

        GoodTherapy paginates with ``?page=N`` (1-indexed). Stop when a page
        returns no new profile links or we reach max_pages_per_city.
        """
        profile_urls: list[str] = []
        seen: set[str] = set()

        for page_num in range(1, max_pages_per_city + 1):
            page_url = f"{city_url}?page={page_num}" if page_num > 1 else city_url
            html = await self._fetch_with_retry(page_url)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            page_links = self._extract_profile_links(soup)

            if not page_links:
                break  # No profiles on this page → end of pagination

            new_links = [u for u in page_links if u not in seen]
            if not new_links:
                break  # All links already seen (duplicate page or loop guard)

            profile_urls.extend(new_links)
            seen.update(new_links)

        return profile_urls

    def _extract_profile_links(self, soup: BeautifulSoup) -> list[str]:
        """Extract therapist profile URLs from a listing page."""
        urls: list[str] = []
        for link in soup.select(SELECTORS["profile_links"]):
            href = link.get("href", "")
            if not href:
                continue
            full = urljoin(self.base_url, href)
            # Validate: must be a profile URL, not a category or search page
            if "/therapists/profile/" in full and full not in urls:
                urls.append(full)
        return urls

    # ── Phase 2: Profile parsing ──────────────────────────────────────────────

    async def parse_therapist_profile(
        self, url: str, html: str
    ) -> TherapistProfile | None:
        """Parse a single GoodTherapy therapist profile page."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            return self._extract_profile(url, soup)
        except Exception as exc:
            logger.warning("%s: parse error for %s: %s", self.source_name, url, exc)
            return None

    def _extract_profile(
        self, url: str, soup: BeautifulSoup
    ) -> TherapistProfile | None:
        """Extract a TherapistProfile from a parsed GoodTherapy profile page."""
        # ── Name + credentials ────────────────────────────────────────────────
        name, credentials = self._extract_name_and_credentials(soup)
        if not name:
            logger.debug("%s: no name found at %s — skipping", self.source_name, url)
            return None

        # ── Source ID from URL slug ───────────────────────────────────────────
        # GoodTherapy profile URLs: /therapists/profile/{slug}
        source_id = url.rstrip("/").split("/")[-1]

        # ── Content sections (div.mb-6) ───────────────────────────────────────
        sections = soup.select(SELECTORS["content_sections"])

        # ── Bio ───────────────────────────────────────────────────────────────
        bio = self._extract_bio(soup, sections)

        # ── Modalities ────────────────────────────────────────────────────────
        modalities = self._extract_modalities(sections)

        # ── Specializations ───────────────────────────────────────────────────
        specializations = self._extract_specializations(sections)

        # ── Insurance ─────────────────────────────────────────────────────────
        next_data = self._get_next_data(soup)
        insurance, sliding_scale = self._extract_insurance(soup, sections, next_data)

        # ── Location ─────────────────────────────────────────────────────────
        location = self._extract_location(url, soup)
        if not location:
            logger.debug(
                "%s: no location found at %s — skipping", self.source_name, url
            )
            return None

        # ── Session formats ───────────────────────────────────────────────────
        session_formats = self._extract_session_formats(soup, sections)

        # ── Fees ─────────────────────────────────────────────────────────────
        fee_min, fee_max = self._extract_fees(soup, sections)

        # ── Accepting new clients ─────────────────────────────────────────────
        accepting_new = self._extract_accepting_new_clients(soup)

        # ── Populations served (from "Groups I Work With" section) ────────────
        populations = self._extract_populations(sections)

        # ── Profile completeness ──────────────────────────────────────────────
        completeness = self._compute_completeness(
            bio, modalities, specializations, insurance
        )

        return TherapistProfile(
            source=self.source_name,
            source_url=url,  # type: ignore[arg-type]
            source_id=source_id,
            name=name,
            credentials=credentials,
            modalities=modalities,
            specializations=specializations,
            populations_served=populations,
            location=location,
            session_formats=session_formats,
            accepts_insurance=insurance,
            sliding_scale=sliding_scale,
            fee_min=fee_min,
            fee_max=fee_max,
            accepting_new_clients=accepting_new,
            bio=bio,
            profile_completeness=completeness,
        )

    # ── Name + credentials extraction ─────────────────────────────────────────

    def _extract_name_and_credentials(
        self, soup: BeautifulSoup
    ) -> tuple[str, list[str]]:
        """
        Extract therapist name and credential list.

        Primary source: page title — "Firstname Lastname, Title, CRED | GoodTherapy"
        Fallback: h1 element.

        Credential detection uses a regex for known abbreviations so that full
        license titles (e.g., "Marriage and Family Therapist") are not confused
        with the credential abbreviation.
        """
        # --- Page title approach ---
        title_el = soup.select_one(SELECTORS["page_title"])
        if title_el:
            title_text = title_el.get_text(strip=True)
            # Strip site suffix
            profile_part = title_text.split(" | ")[0].strip()
            if profile_part:
                return self._parse_name_credentials_from_string(profile_part)

        # --- h1 fallback ---
        h1_el = soup.select_one(SELECTORS["h1_name"])
        if h1_el:
            return self._parse_name_credentials_from_string(
                h1_el.get_text(strip=True)
            )

        return "", []

    @staticmethod
    def _parse_name_credentials_from_string(
        text: str,
    ) -> tuple[str, list[str]]:
        """
        Split a string like "Jane Smith, LMFT, PhD" into name and credentials.

        Strategy:
        1. Find all credential abbreviations at the end of the comma-separated list.
        2. Everything before the first recognized credential (or just the first
           two comma-parts) is the name.
        """
        parts = [p.strip() for p in text.split(",")]
        if not parts:
            return "", []

        # Identify which parts are credentials vs. name tokens
        creds: list[str] = []
        name_parts: list[str] = []
        found_cred = False

        for i, part in enumerate(parts):
            if _CRED_RE.search(part):
                found_cred = True
                # Extract all credentials in this part (e.g., "LMFT PhD" → 2)
                matched = _CRED_RE.findall(part)
                creds.extend(matched)
            else:
                if not found_cred:
                    name_parts.append(part)
                # After a cred is found, skip non-cred parts (full license titles)

        name = ", ".join(name_parts).strip() if name_parts else parts[0].strip()
        # Clean up any residual credential tokens from the name
        name = _CRED_RE.sub("", name).strip().strip(",").strip()

        return name, creds

    # ── Bio extraction ─────────────────────────────────────────────────────────

    def _extract_bio(
        self, soup: BeautifulSoup, sections: list[Tag]
    ) -> str:
        """
        Extract the therapist's bio text.

        Primary: GoodTherapy renders bio paragraphs in ``div`` elements with the
        Tailwind class ``whitespace-pre-line`` (class ``whitespace-pre-line text-sm
        leading-relaxed text-[#333]``). Each titled bio subsection gets its own such
        div. We concatenate all of them for the full bio.

        Fallback: If that selector yields nothing (older profile layouts), we fall
        back to the longest ``div.mb-6`` section that is *not* a structured-data
        block (challenges / approaches / insurance / services / breadcrumb).
        """
        import html as _html

        # ── Primary: whitespace-pre-line prose divs ────────────────────────────
        prose_divs = soup.select("div[class*='whitespace-pre-line']")
        if prose_divs:
            parts = [
                _html.unescape(d.get_text(separator=" ", strip=True))
                for d in prose_divs
                if len(d.get_text(strip=True)) > 30  # skip trivially short fragments
            ]
            if parts:
                return " ".join(parts)

        # ── Fallback: div.mb-6 heuristic ──────────────────────────────────────
        _SKIP_HEADINGS = (
            _SECTION_CHALLENGES, _SECTION_APPROACHES, _SECTION_INSURANCE,
            "services", "languages", "age groups", "industries", "communities",
        )
        best_text = ""
        for section in sections:
            heading = self._section_heading(section)
            if any(kw in heading for kw in _SKIP_HEADINGS):
                continue

            # Skip the breadcrumb section (no heading, starts with "< Back")
            raw_start = section.get_text(strip=True)[:20]
            if raw_start.startswith("< Back") or raw_start.startswith("> Find"):
                continue

            # Prefer paragraphs to avoid picking up enumeration sections
            paragraphs = section.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
            else:
                heading_el = section.find(["h2", "h3", "h4", "strong", "b"])
                raw = section.get_text(separator=" ", strip=True)
                if heading_el:
                    heading_text = heading_el.get_text(strip=True)
                    if raw.startswith(heading_text):
                        raw = raw[len(heading_text):].strip()
                text = raw

            text = _html.unescape(text)
            if len(text) > len(best_text):
                best_text = text

        return best_text

    # ── Modalities extraction ─────────────────────────────────────────────────

    def _extract_modalities(self, sections: list[Tag]) -> list[TherapyModality]:
        """
        Extract therapeutic modalities from the approaches section.

        GoodTherapy labels this section "Therapeutic Approaches & Evidence-Based Methods"
        (or similar). We find it by heading keyword, then match against MODALITY_MAP.
        """
        approaches_section = self._find_section(sections, _SECTION_APPROACHES)
        if not approaches_section:
            # Fallback: search all sections
            return self._scan_text_for_modalities(
                " ".join(s.get_text(separator=" ", strip=True) for s in sections)
            )

        text = approaches_section.get_text(separator=" ", strip=True).lower()
        return self._scan_text_for_modalities(text)

    def _scan_text_for_modalities(self, text: str) -> list[TherapyModality]:
        """Match MODALITY_MAP keywords against arbitrary text."""
        text_lower = f" {text.lower()} "  # Pad so word-boundary checks work
        result: list[TherapyModality] = []
        for keyword, modality in MODALITY_MAP.items():
            if keyword in text_lower:
                if modality not in result:
                    result.append(modality)
        return result

    # ── Specializations extraction ────────────────────────────────────────────

    def _extract_specializations(
        self, sections: list[Tag]
    ) -> list[TherapistSpecialization]:
        """
        Extract specializations from the "Client Challenges & Concerns" section.
        """
        challenges_section = self._find_section(sections, _SECTION_CHALLENGES)
        if not challenges_section:
            return []

        text = challenges_section.get_text(separator=" ", strip=True).lower()
        result: list[TherapistSpecialization] = []
        for keyword, spec in SPECIALIZATION_MAP.items():
            if keyword in text:
                if spec not in result:
                    result.append(spec)
        return result

    # ── Insurance extraction ──────────────────────────────────────────────────

    @staticmethod
    def _get_next_data(soup: BeautifulSoup) -> dict:
        """Extract the __NEXT_DATA__ JSON profile blob embedded by Next.js, or empty dict."""
        try:
            script = soup.find("script", {"id": "__NEXT_DATA__"})
            if script:
                return json.loads(script.string)["props"]["pageProps"]["profile"]
        except Exception:
            pass
        return {}

    def _extract_insurance(
        self, soup: BeautifulSoup, sections: list[Tag], next_data: dict | None = None
    ) -> tuple[list[InsuranceProvider], bool]:
        """
        Extract accepted insurance providers and sliding scale flag.

        Reads from __NEXT_DATA__ JSON first (authoritative source since GoodTherapy
        migrated to Next.js). Falls back to HTML parsing if JSON is unavailable.

        Returns (insurance_list, sliding_scale_flag).
        """
        if next_data:
            return self._extract_insurance_from_json(next_data)
        return self._extract_insurance_from_html(soup, sections)

    def _extract_insurance_from_json(
        self, profile: dict
    ) -> tuple[list[InsuranceProvider], bool]:
        """Parse insurance from the __NEXT_DATA__ JSON profile object."""
        sliding = bool(profile.get("slidingScale", False))
        raw_list: list[str] = profile.get("insuranceCompaniesList") or []
        providers: list[InsuranceProvider] = []
        for name in raw_list:
            mapped = self._map_insurance_name(name)
            if mapped and mapped not in providers:
                providers.append(mapped)
        return providers, sliding

    @staticmethod
    def _map_insurance_name(name: str) -> InsuranceProvider | None:
        """Map a JSON insurance company name to an InsuranceProvider enum value."""
        lower = name.lower()
        mapping = [
            ("blue cross blue shield", InsuranceProvider.BLUE_CROSS),
            ("anthem blue cross", InsuranceProvider.BLUE_CROSS),
            ("bluecross", InsuranceProvider.BLUE_CROSS),
            ("blue cross", InsuranceProvider.BLUE_CROSS),
            ("blue shield of california", InsuranceProvider.BLUE_SHIELD),
            ("blue shield", InsuranceProvider.BLUE_SHIELD),
            ("optum/united", InsuranceProvider.UNITED),
            ("united behavioral health", InsuranceProvider.UNITED),
            ("unitedhealthcare", InsuranceProvider.UNITED),
            ("united health care", InsuranceProvider.UNITED),
            ("optum health", InsuranceProvider.OPTUM),
            ("optum", InsuranceProvider.OPTUM),
            ("aetna", InsuranceProvider.AETNA),
            ("cigna", InsuranceProvider.CIGNA),
            ("kaiser", InsuranceProvider.KAISER),
            ("magellan", InsuranceProvider.MAGELLAN),
            ("out of network", InsuranceProvider.OUT_OF_NETWORK),
        ]
        for fragment, provider in mapping:
            if fragment in lower:
                return provider
        return None

    def _extract_insurance_from_html(
        self, soup: BeautifulSoup, sections: list[Tag]
    ) -> tuple[list[InsuranceProvider], bool]:
        """HTML fallback for insurance extraction (used when __NEXT_DATA__ is absent)."""
        insurance_section = self._find_section(sections, _SECTION_INSURANCE)
        insurance_text = ""
        if insurance_section:
            insurance_text = insurance_section.get_text(separator=" ", strip=True).lower()
        else:
            fee_div = soup.select_one(SELECTORS["fee_div"])
            if fee_div:
                insurance_text = fee_div.get_text(separator=" ", strip=True).lower()

        if not insurance_text:
            return [], False

        providers: list[InsuranceProvider] = []
        sliding = False

        if (
            "don't accept insurance" in insurance_text
            or "do not accept insurance" in insurance_text
            or "does not accept insurance" in insurance_text
            or "not accept insurance" in insurance_text
        ):
            return [InsuranceProvider.SELF_PAY], False

        for keyword, provider in INSURANCE_MAP.items():
            if keyword in insurance_text:
                if provider == InsuranceProvider.SLIDING_SCALE:
                    sliding = True
                elif provider not in providers:
                    providers.append(provider)

        if not providers and not sliding:
            if "out of network" in insurance_text or "out-of-network" in insurance_text:
                providers.append(InsuranceProvider.OUT_OF_NETWORK)
            elif "self" in insurance_text or "private pay" in insurance_text:
                providers.append(InsuranceProvider.SELF_PAY)

        return providers, sliding

    # ── Location extraction ────────────────────────────────────────────────────

    def _extract_location(
        self, url: str, soup: BeautifulSoup
    ) -> TherapistLocation | None:
        """
        Extract city from the profile URL or breadcrumb navigation.

        GoodTherapy profile URLs don't always embed city, but breadcrumb links
        back to /therapists/ca/{city} do. Fall back to URL-based extraction
        if breadcrumbs aren't present.

        We don't trust city inferred from which city listing the profile appeared
        in, because the same therapist can appear under many cities. Instead we
        read from the profile page itself.
        """
        # --- Breadcrumb approach ---
        breadcrumb_links = soup.select(SELECTORS["breadcrumb_links"])
        for link in breadcrumb_links:
            href = link.get("href", "")
            # Match /therapists/ca/{city} breadcrumb
            m = re.search(r"/therapists/[Cc][Aa]/([^/?#]+)", href)
            if m:
                city = self._slug_to_city(m.group(1))
                return TherapistLocation(city=city, state="CA")

        # --- Structured address div fallback ---
        # Look for city/state pattern in page text
        page_text = soup.get_text(separator=" ")
        m = re.search(r"\b([A-Za-z][A-Za-z\s]{2,30}),\s*CA\s*(\d{5})?\b", page_text)
        if m:
            return TherapistLocation(
                city=m.group(1).strip(),
                state="CA",
                zip_code=m.group(2),
            )

        # --- URL slug fallback: sometimes the profile URL contains city ---
        # e.g. .../profile/jane-smith-los-angeles — heuristic at best
        # We require at least one better signal, so return None if we get here
        return None

    @staticmethod
    def _slug_to_city(slug: str) -> str:
        """Convert a GoodTherapy city slug to a display name. e.g. 'los-angeles' → 'Los Angeles'."""
        return slug.replace("-", " ").title()

    # ── Session formats ────────────────────────────────────────────────────────

    def _extract_session_formats(
        self, soup: BeautifulSoup, sections: list[Tag]
    ) -> list[SessionFormat]:
        """
        Determine session format from telehealth badges or section text.
        """
        page_text = soup.get_text(separator=" ", strip=True).lower()
        has_online = (
            bool(soup.select_one(SELECTORS["telehealth_badge"]))
            or "telehealth" in page_text
            or "online therapy" in page_text
            or "video sessions" in page_text
            or "virtual sessions" in page_text
        )
        has_in_person = (
            "in-person" in page_text
            or "in person" in page_text
            or "office" in page_text
        )

        if has_online and has_in_person:
            return [SessionFormat.BOTH]
        if has_online:
            return [SessionFormat.TELEHEALTH]
        return [SessionFormat.IN_PERSON]

    # ── Fee extraction ─────────────────────────────────────────────────────────

    def _extract_fees(
        self, soup: BeautifulSoup, sections: list[Tag]
    ) -> tuple[int | None, int | None]:
        """
        Extract fee range from fee-related divs or insurance sections.

        Looks for patterns like "$120", "$120-$180", "$80 - $200".
        """
        # Search fee-specific element first
        fee_el = soup.select_one(SELECTORS["fee_div"])
        search_text = fee_el.get_text(strip=True) if fee_el else ""

        # Also search the insurance section (fees often listed there)
        if not search_text:
            insurance_section = self._find_section(sections, _SECTION_INSURANCE)
            if insurance_section:
                search_text = insurance_section.get_text(strip=True)

        if search_text:
            match = _FEE_RE.search(search_text)
            if match:
                fee_min = int(match.group(1))
                fee_max = int(match.group(2)) if match.group(2) else fee_min
                return fee_min, fee_max

        # Scan full page as last resort
        page_text = soup.get_text(separator=" ", strip=True)
        for match in _FEE_RE.finditer(page_text):
            fee_min_val = int(match.group(1))
            fee_max_val = int(match.group(2)) if match.group(2) else fee_min_val
            # Sanity check: realistic therapy fee range
            if 20 <= fee_min_val <= 1000:
                return fee_min_val, fee_max_val

        return None, None

    # ── Accepting new clients ─────────────────────────────────────────────────

    def _extract_accepting_new_clients(self, soup: BeautifulSoup) -> bool:
        """
        Determine if the therapist is accepting new clients.

        Defaults to True (most listed therapists are active/available).
        """
        el = soup.select_one(SELECTORS["accepting_clients"])
        if el:
            text = el.get_text(strip=True).lower()
            if "not accepting" in text or "unavailable" in text:
                return False
            if "accepting" in text:
                return True

        # Search page text for explicit signals
        page_text = soup.get_text(separator=" ", strip=True).lower()
        if "not accepting new clients" in page_text or "not accepting new patients" in page_text:
            return False
        return True

    # ── Populations ───────────────────────────────────────────────────────────

    def _extract_populations(self, sections: list[Tag]) -> list[str]:
        """
        Extract populations from the "Groups I Work With" section.

        This section lists items like "Adults", "Children", "Couples", etc.
        We return them as raw strings (not mapped to an enum) since
        TherapistProfile.populations_served is a list[str].
        """
        groups_section = self._find_section(sections, _SECTION_GROUPS)
        if not groups_section:
            return []

        populations: list[str] = []
        # Try list items first
        for li in groups_section.find_all("li"):
            text = li.get_text(strip=True)
            if text and len(text) < 60:  # Skip overly long items
                populations.append(text)

        if not populations:
            # Fall back to comma-separated span/text content
            raw = groups_section.get_text(separator=",", strip=True)
            for item in raw.split(","):
                item = item.strip()
                if item and len(item) < 60:
                    populations.append(item)

        # Remove duplicates while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for p in populations:
            if p.lower() not in seen:
                seen.add(p.lower())
                result.append(p)

        return result

    # ── Section helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _section_heading(section: Tag) -> str:
        """Return the lowercased heading text of a div.mb-6 section, or ''."""
        heading_el = section.find(["h2", "h3", "h4", "strong", "b"])
        if heading_el:
            return heading_el.get_text(strip=True).lower()
        return ""

    def _find_section(
        self, sections: list[Tag], heading_keyword: str
    ) -> Tag | None:
        """Find the first div.mb-6 whose heading contains *heading_keyword*."""
        for section in sections:
            if heading_keyword in self._section_heading(section):
                return section
        return None

    # ── Completeness score ────────────────────────────────────────────────────

    @staticmethod
    def _compute_completeness(
        bio: str,
        modalities: list,
        specializations: list,
        insurance: list,
    ) -> float:
        """Heuristic profile completeness score in [0, 1]."""
        score = 0.0
        if bio and len(bio) > 50:
            score += 0.35
        if modalities:
            score += 0.25
        if specializations:
            score += 0.25
        if insurance:
            score += 0.15
        return round(score, 2)
