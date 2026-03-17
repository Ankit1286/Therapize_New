"""
Psychology Today scraper for California therapist profiles.

Psychology Today is the largest therapist directory in the US.
California URL: https://www.psychologytoday.com/us/therapists/ca

Data extraction:
- Listing pages → therapist profile URLs
- Profile pages → structured therapist data

Note on ethics:
- We respect robots.txt, rate limits, and Terms of Service
- We only collect publicly visible information
- Data is used for research/educational purposes
- We identify our bot in User-Agent

Note on parsing:
- Psychology Today's HTML structure changes. BeautifulSoup selectors
  may break. The SELECTORS dict centralizes all CSS selectors for easy updates.
- If scraping breaks, check selectors first — they're the most brittle part.
"""
import logging
import re
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup

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

# CSS selectors — centralized for easy maintenance when the site updates
SELECTORS = {
    "therapist_cards": "div.results-row div.profile-summary",
    "profile_link": "a.profile-title",
    "name": "h1.profile-title",
    "credentials": "div.credentials",
    "bio": "div.profile-statement",
    "approach": "div.treatment-orientation",
    "specializations": "div.specialty-list span.spec",
    "modalities": "div.modalities-list span.modality",
    "insurance": "div.insurance-list span.insurance-name",
    "location": "div.profile-address",
    "city": "span.city",
    "zip": "span.zip",
    "fee": "div.fee-range",
    "session_format": "div.telehealth-badge",
    "accepting_new": "div.accepting-clients",
    "rating": "span.rating-value",
    "review_count": "span.review-count",
}

# Map PT insurance names to our enum values
INSURANCE_MAP = {
    "blue shield": InsuranceProvider.BLUE_SHIELD,
    "blue cross": InsuranceProvider.BLUE_CROSS,
    "aetna": InsuranceProvider.AETNA,
    "united": InsuranceProvider.UNITED,
    "cigna": InsuranceProvider.CIGNA,
    "kaiser": InsuranceProvider.KAISER,
    "magellan": InsuranceProvider.MAGELLAN,
    "optum": InsuranceProvider.OPTUM,
    "sliding scale": InsuranceProvider.SLIDING_SCALE,
}

# Map PT modality names to our enum values
MODALITY_MAP = {
    "cognitive behavioral": TherapyModality.CBT,
    "cbt": TherapyModality.CBT,
    "dialectical behavior": TherapyModality.DBT,
    "dbt": TherapyModality.DBT,
    "emdr": TherapyModality.EMDR,
    "acceptance and commitment": TherapyModality.ACT,
    "act": TherapyModality.ACT,
    "psychodynamic": TherapyModality.PSYCHODYNAMIC,
    "humanistic": TherapyModality.HUMANISTIC,
    "mindfulness": TherapyModality.MINDFULNESS,
    "somatic": TherapyModality.SOMATIC,
    "gottman": TherapyModality.GOTTMAN,
    "emotionally focused": TherapyModality.EFT,
    "eft": TherapyModality.EFT,
    "exposure": TherapyModality.EXPOSURE,
    "narrative": TherapyModality.NARRATIVE,
    "solution-focused": TherapyModality.SOLUTION_FOCUSED,
    "motivational interviewing": TherapyModality.MOTIVATIONAL,
    "family systems": TherapyModality.FAMILY_SYSTEMS,
    "trauma-informed": TherapyModality.TRAUMA_INFORMED,
    "integrative": TherapyModality.INTEGRATIVE,
}

SPECIALIZATION_MAP = {
    "anxiety": TherapistSpecialization.ANXIETY,
    "depression": TherapistSpecialization.DEPRESSION,
    "trauma": TherapistSpecialization.TRAUMA,
    "ptsd": TherapistSpecialization.PTSD,
    "relationship": TherapistSpecialization.RELATIONSHIP,
    "grief": TherapistSpecialization.GRIEF,
    "adhd": TherapistSpecialization.ADHD,
    "addiction": TherapistSpecialization.ADDICTION,
    "eating": TherapistSpecialization.EATING_DISORDERS,
    "ocd": TherapistSpecialization.OCD,
    "bipolar": TherapistSpecialization.BIPOLAR,
    "chronic pain": TherapistSpecialization.CHRONIC_PAIN,
    "life transitions": TherapistSpecialization.LIFE_TRANSITIONS,
    "lgbtq": TherapistSpecialization.LGBTQ,
    "couples": TherapistSpecialization.COUPLES,
    "family": TherapistSpecialization.FAMILY,
    "stress": TherapistSpecialization.STRESS,
    "sleep": TherapistSpecialization.SLEEP,
    "self-esteem": TherapistSpecialization.SELF_ESTEEM,
    "anger": TherapistSpecialization.ANGER,
}


class PsychologyTodayScraper(BaseScraper):

    @property
    def base_url(self) -> str:
        return "https://www.psychologytoday.com"

    @property
    def source_name(self) -> str:
        return "psychology_today"

    async def scrape_therapist_urls(self, max_pages: int = 100) -> list[str]:
        """
        Scrape listing pages to collect individual therapist profile URLs.
        PT's CA listing: /us/therapists/ca?page=N
        """
        urls = []
        for page_num in range(1, max_pages + 1):
            listing_url = (
                f"{self.base_url}/us/therapists/ca"
                f"?{urlencode({'page': page_num})}"
            )
            html = await self._fetch_with_retry(listing_url)
            if not html:
                break

            soup = BeautifulSoup(html, "html.parser")
            cards = soup.select(SELECTORS["therapist_cards"])

            if not cards:
                logger.info("No more therapist cards on page %d — stopping", page_num)
                break

            for card in cards:
                link = card.select_one(SELECTORS["profile_link"])
                if link and link.get("href"):
                    full_url = urljoin(self.base_url, link["href"])
                    urls.append(full_url)

            logger.debug("Page %d: collected %d URLs", page_num, len(cards))

        return urls

    async def parse_therapist_profile(
        self, url: str, html: str
    ) -> TherapistProfile | None:
        """Parse a single therapist profile page."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            return self._extract_profile(url, soup)
        except Exception as exc:
            logger.warning("Parse error for %s: %s", url, exc)
            return None

    def _extract_profile(self, url: str, soup: BeautifulSoup) -> TherapistProfile | None:
        """Extract structured data from parsed HTML."""
        # Name (required — skip if missing)
        name_el = soup.select_one(SELECTORS["name"])
        if not name_el:
            return None
        name = name_el.get_text(strip=True)
        if not name:
            return None

        # Extract source ID from URL
        # PT URLs: /us/therapists/ca/john-smith/123456
        source_id = url.rstrip("/").split("/")[-1]

        # Credentials
        creds_el = soup.select_one(SELECTORS["credentials"])
        credentials = []
        if creds_el:
            # e.g., "LMFT, PhD" → ["LMFT", "PhD"]
            credentials = [c.strip() for c in creds_el.get_text().split(",")]

        # Bio
        bio_el = soup.select_one(SELECTORS["bio"])
        bio = bio_el.get_text(strip=True) if bio_el else ""

        # Approach
        approach_el = soup.select_one(SELECTORS["approach"])
        approach = approach_el.get_text(strip=True) if approach_el else ""

        # Modalities
        modalities = self._extract_modalities(soup)

        # Specializations
        specializations = self._extract_specializations(soup)

        # Insurance
        insurance = self._extract_insurance(soup)

        # Location
        location = self._extract_location(soup)
        if not location:
            return None  # Skip profiles without location

        # Session format
        session_formats = self._extract_session_formats(soup)

        # Fee
        fee_min, fee_max = self._extract_fees(soup)

        # Accepting new clients
        accepting_el = soup.select_one(SELECTORS["accepting_new"])
        accepting_new = True  # default
        if accepting_el:
            accepting_new = "accepting" in accepting_el.get_text().lower()

        # Rating
        rating, review_count = self._extract_rating(soup)

        # Sliding scale
        sliding_scale = any(
            i == InsuranceProvider.SLIDING_SCALE for i in insurance
        )

        # Profile completeness score (heuristic)
        completeness = self._compute_completeness(
            bio, approach, modalities, specializations, insurance
        )

        profile = TherapistProfile(
            source=self.source_name,
            source_url=url,  # type: ignore[arg-type]
            source_id=source_id,
            name=name,
            credentials=credentials,
            modalities=modalities,
            specializations=specializations,
            location=location,
            session_formats=session_formats,
            accepts_insurance=insurance,
            sliding_scale=sliding_scale,
            fee_min=fee_min,
            fee_max=fee_max,
            accepting_new_clients=accepting_new,
            bio=bio,
            approach_description=approach,
            rating=rating,
            review_count=review_count,
            profile_completeness=completeness,
        )
        return profile

    def _extract_modalities(self, soup: BeautifulSoup) -> list[TherapyModality]:
        modalities = []
        for el in soup.select(SELECTORS["modalities"]):
            text = el.get_text(strip=True).lower()
            for keyword, modality in MODALITY_MAP.items():
                if keyword in text:
                    if modality not in modalities:
                        modalities.append(modality)
                    break
        return modalities

    def _extract_specializations(
        self, soup: BeautifulSoup
    ) -> list[TherapistSpecialization]:
        specs = []
        for el in soup.select(SELECTORS["specializations"]):
            text = el.get_text(strip=True).lower()
            for keyword, spec in SPECIALIZATION_MAP.items():
                if keyword in text:
                    if spec not in specs:
                        specs.append(spec)
                    break
        return specs

    def _extract_insurance(self, soup: BeautifulSoup) -> list[InsuranceProvider]:
        providers = []
        for el in soup.select(SELECTORS["insurance"]):
            text = el.get_text(strip=True).lower()
            for keyword, provider in INSURANCE_MAP.items():
                if keyword in text:
                    if provider not in providers:
                        providers.append(provider)
                    break
        return providers

    def _extract_location(self, soup: BeautifulSoup) -> TherapistLocation | None:
        city_el = soup.select_one(SELECTORS["city"])
        if not city_el:
            # Try structured address div
            addr_el = soup.select_one(SELECTORS["location"])
            if not addr_el:
                return None
            # Parse from text: "San Francisco, CA 94102"
            text = addr_el.get_text(strip=True)
            match = re.search(r"([A-Za-z\s]+),\s*CA\s*(\d{5})?", text)
            if match:
                return TherapistLocation(
                    city=match.group(1).strip(),
                    state="CA",
                    zip_code=match.group(2),
                )
            return None

        city = city_el.get_text(strip=True)
        zip_el = soup.select_one(SELECTORS["zip"])
        zip_code = zip_el.get_text(strip=True) if zip_el else None

        return TherapistLocation(city=city, state="CA", zip_code=zip_code)

    def _extract_session_formats(self, soup: BeautifulSoup) -> list[SessionFormat]:
        formats = []
        telehealth_el = soup.select_one(SELECTORS["session_format"])
        if telehealth_el:
            formats.append(SessionFormat.TELEHEALTH)
        # Assume in-person if they have a location
        if soup.select_one(SELECTORS["location"]):
            if SessionFormat.TELEHEALTH in formats:
                # Replace with BOTH
                return [SessionFormat.BOTH]
            formats.append(SessionFormat.IN_PERSON)
        return formats or [SessionFormat.IN_PERSON]

    def _extract_fees(self, soup: BeautifulSoup) -> tuple[int | None, int | None]:
        fee_el = soup.select_one(SELECTORS["fee"])
        if not fee_el:
            return None, None
        text = fee_el.get_text(strip=True)
        # Match patterns like "$120-$180" or "120-180"
        match = re.search(r"\$?(\d+)(?:\s*[-–]\s*\$?(\d+))?", text)
        if match:
            fee_min = int(match.group(1))
            fee_max = int(match.group(2)) if match.group(2) else fee_min
            return fee_min, fee_max
        return None, None

    def _extract_rating(
        self, soup: BeautifulSoup
    ) -> tuple[float | None, int]:
        rating_el = soup.select_one(SELECTORS["rating"])
        count_el = soup.select_one(SELECTORS["review_count"])
        rating = None
        count = 0
        if rating_el:
            try:
                rating = float(rating_el.get_text(strip=True))
            except ValueError:
                pass
        if count_el:
            try:
                count = int(re.sub(r"\D", "", count_el.get_text()))
            except ValueError:
                pass
        return rating, count

    @staticmethod
    def _compute_completeness(bio, approach, modalities, specializations, insurance) -> float:
        """Simple completeness score: fraction of key fields populated."""
        score = 0.0
        if bio and len(bio) > 50:
            score += 0.3
        if approach and len(approach) > 20:
            score += 0.2
        if modalities:
            score += 0.2
        if specializations:
            score += 0.2
        if insurance:
            score += 0.1
        return round(score, 2)
