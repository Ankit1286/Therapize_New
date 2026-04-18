"""
Filter Engine — hard constraint filtering before ranking.

Rule-based, deterministic, zero-LLM.
Applied FIRST to reduce the candidate set before expensive operations.

Performance: SQL-level filtering via WHERE clauses (not Python-level)
so this scales to millions of therapist records.

Key insight: Apply the cheapest, most discriminating filters first.
Order: accepting_new_clients → location → insurance → budget → format
"""
import logging
from dataclasses import dataclass

from src.models.query import ExtractedQueryIntent, UserQuestionnaire
from src.models.therapist import InsuranceProvider, SessionFormat

logger = logging.getLogger(__name__)

# Maps age_group values from UserQuestionnaire to ILIKE patterns matched against
# the populations_served TEXT[] column.  Patterns are intentionally broad to
# handle the free-text variants scraped from different platforms.
_AGE_GROUP_PATTERNS: dict[str, list[str]] = {
    "child":      ["%child%", "%toddler%", "%preteen%", "%tween%"],
    "adolescent": ["%adolescent%", "%teen%", "%preteen%", "%tween%"],
    "adult":      ["%adult%", "%elder%"],
}


@dataclass
class FilterCriteria:
    """
    Compiled filter criteria ready for SQL generation.
    Keeping this separate from the query model makes SQL generation clean.
    """
    # Geographic
    state: str = "CA"
    city: str | None = None
    zip_code: str | None = None
    max_distance_miles: int | None = None

    # Logistics
    insurance: InsuranceProvider | None = None
    session_format: SessionFormat | None = None
    max_budget: int | None = None
    accepting_new_clients: bool = True

    # Preferences
    preferred_gender: str | None = None
    preferred_language: str | None = None
    preferred_ethnicity: str | None = None

    # Requirement — never relaxed in progressive fallback
    age_group: str | None = None


class FilterEngine:
    """
    Compiles search intent into SQL WHERE clauses for PostgreSQL.

    Design: generate parameterized queries, never f-string SQL
    (prevents SQL injection even though these come from an LLM).
    """

    def compile_filters(
        self,
        intent: ExtractedQueryIntent,
    ) -> FilterCriteria:
        """Merge explicit questionnaire + LLM-inferred filters into FilterCriteria."""
        explicit = intent.inferred_filters

        return FilterCriteria(
            state="CA",  # always California
            city=explicit.city,
            zip_code=explicit.zip_code,
            max_distance_miles=explicit.max_distance_miles,
            insurance=explicit.insurance,
            session_format=explicit.session_format,
            max_budget=explicit.max_budget_per_session,
            accepting_new_clients=True,  # always filter for active therapists
            preferred_gender=explicit.preferred_gender,
            preferred_language=explicit.preferred_language,
            preferred_ethnicity=explicit.preferred_ethnicity,
            age_group=explicit.age_group,
        )

    def build_sql_where(self, criteria: FilterCriteria) -> tuple[str, list]:
        """
        Build parameterized SQL WHERE clause and parameter list.

        Returns: (where_clause_string, params_list)
        Always returns valid SQL even if no filters specified.

        Example output:
            WHERE state = $1 AND accepting_new_clients = $2
                  AND insurance_providers && $3::insurance_provider[]
        """
        conditions = []
        params = []
        idx = 1

        # Always filter to California and active therapists
        conditions.append(f"state = ${idx}")
        params.append(criteria.state)
        idx += 1

        conditions.append(f"accepting_new_clients = ${idx}")
        params.append(criteria.accepting_new_clients)
        idx += 1

        # City filter: always applied as a hard filter when set.
        # Matches therapists physically based in the selected city, regardless of
        # session format. Telehealth users may still want a therapist from their city.
        if criteria.city:
            conditions.append(f"LOWER(city) = LOWER(${idx})")
            params.append(criteria.city)
            idx += 1

        # Zip code filter
        if criteria.zip_code:
            conditions.append(f"zip_code = ${idx}")
            params.append(criteria.zip_code)
            idx += 1

        # Insurance filter — uses PostgreSQL array overlap operator (&&)
        if criteria.insurance and criteria.insurance != InsuranceProvider.SELF_PAY:
            # Accept therapists who take the specified insurance OR sliding scale
            conditions.append(
                f"(insurance_providers @> ARRAY[${idx}::insurance_provider] "
                f"OR ${idx + 1}::insurance_provider = ANY(insurance_providers))"
            )
            params.extend([criteria.insurance.value, InsuranceProvider.SLIDING_SCALE.value])
            idx += 2

        # Session format filter
        if criteria.session_format and criteria.session_format != SessionFormat.BOTH:
            conditions.append(
                f"(${idx}::session_format = ANY(session_formats) "
                f"OR 'both'::session_format = ANY(session_formats))"
            )
            params.append(criteria.session_format.value)
            idx += 1

        # Budget filter (use fee_min — if their minimum fee is within budget)
        if criteria.max_budget:
            conditions.append(
                f"(fee_min IS NULL OR fee_min <= ${idx})"
            )
            params.append(criteria.max_budget)
            idx += 1

        # Gender preference
        if criteria.preferred_gender:
            conditions.append(f"LOWER(gender) = LOWER(${idx})")
            params.append(criteria.preferred_gender)
            idx += 1

        # Ethnicity preference
        if criteria.preferred_ethnicity:
            conditions.append(
                f"EXISTS ("
                f"SELECT 1 FROM unnest(ethnicity) AS eth "
                f"WHERE LOWER(eth) = LOWER(${idx})"
                f")"
            )
            params.append(criteria.preferred_ethnicity)
            idx += 1

        # Language preference
        if criteria.preferred_language:
            conditions.append(f"${idx} = ANY(languages)")
            params.append(criteria.preferred_language.lower())
            idx += 1

        # Age group — matched against populations_served free-text array.
        # Uses ILIKE patterns because scraped values are not normalised
        # (e.g. "Children (6-10)", "Adolescents / Teenagers (14 - 19)").
        if criteria.age_group:
            patterns = _AGE_GROUP_PATTERNS.get(criteria.age_group.lower(), [])
            if patterns:
                conditions.append(
                    f"EXISTS ("
                    f"SELECT 1 FROM unnest(populations_served) AS pop "
                    f"WHERE pop ILIKE ANY(${idx}::text[])"
                    f")"
                )
                params.append(patterns)
                idx += 1

        where_clause = " AND ".join(conditions)
        logger.debug(
            "Filter SQL: WHERE %s | params=%s", where_clause, params
        )
        return where_clause, params

    def explain_filters(self, criteria: FilterCriteria) -> list[str]:
        """Human-readable explanation of applied filters (for debugging & UI)."""
        explanations = []
        if criteria.city:
            explanations.append(f"Location: {criteria.city}, CA")
        if criteria.insurance:
            explanations.append(f"Insurance: {criteria.insurance.value}")
        if criteria.max_budget:
            explanations.append(f"Budget: up to ${criteria.max_budget}/session")
        if criteria.session_format:
            explanations.append(f"Format: {criteria.session_format.value}")
        if criteria.preferred_language:
            explanations.append(f"Language: {criteria.preferred_language}")
        return explanations
