"""
Data cleaning and normalization for scraped therapist profiles.

Cleaning prevents garbage from entering the DB and degrading search quality.
Validates and normalizes at the pipeline boundary — not at query time.

Principle: be lenient on input, strict on output.
"""
import logging
import re

from src.models.therapist import TherapistProfile

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and validates scraped therapist profiles."""

    MIN_BIO_LENGTH = 20   # reject empty/stub bios
    MAX_BIO_LENGTH = 5000  # truncate absurdly long bios
    MAX_FEE = 1000         # sanity check: no $1000/session records

    def clean(self, profile: TherapistProfile) -> TherapistProfile | None:
        """
        Clean a raw scraped profile. Returns None if the profile is invalid.

        Validation rules:
        - Name must be non-empty
        - Must have a California location
        - Bio is cleaned but not required
        """
        # Required fields
        if not profile.name or len(profile.name.strip()) < 2:
            return None
        if not profile.location or profile.location.state != "CA":
            return None
        if not profile.source_url:
            return None

        # Normalize name (strip extra whitespace, title case)
        name = " ".join(profile.name.split())
        name = name.title() if name.isupper() else name

        # Clean bio
        bio = self._clean_text(profile.bio)
        if len(bio) < self.MIN_BIO_LENGTH:
            bio = ""
        bio = bio[:self.MAX_BIO_LENGTH]

        # Sanity-check fees
        fee_min = profile.fee_min
        fee_max = profile.fee_max
        if fee_min and (fee_min < 0 or fee_min > self.MAX_FEE):
            fee_min = None
            fee_max = None
        if fee_max and fee_max > self.MAX_FEE:
            fee_max = fee_min

        # Normalize city — strip whitespace, control chars, trailing punctuation
        if profile.location.city:
            import re
            city = re.sub(r'[\x00-\x1f\u200b-\u200f]', '', profile.location.city)
            city = city.strip().rstrip(',').strip().title() or None
        else:
            city = None

        return profile.model_copy(update={
            "name": name,
            "bio": bio,
            "fee_min": fee_min,
            "fee_max": fee_max,
            "location": profile.location.model_copy(update={"city": city}),
            "languages": [lang.lower().strip() for lang in profile.languages],
            "credentials": [c.strip().upper() for c in profile.credentials if c.strip()],
        })

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove HTML artifacts, excessive whitespace, special chars."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove non-printable characters
        text = "".join(c for c in text if c.isprintable())
        return text
