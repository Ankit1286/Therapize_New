"""
DB correctness verification script.

Runs SQL checks against the therapists table after ingestion and prints a
pass/fail report. Exits non-zero on failures so it can be used in CI.

Usage:
    python scripts/verify_db.py           # full report
    python scripts/verify_db.py --fail-only   # only show non-passing checks
"""
import asyncio
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.storage.database import close_db, get_connection, init_db

logging.basicConfig(level=logging.WARNING)


class Severity(Enum):
    """Severity levels for DB checks."""
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class CheckResult:
    """Result of a single DB verification check."""
    name: str
    category: str
    severity: Severity
    passed: bool
    detail: str


class DatabaseVerifier:
    """
    Runs all DB integrity checks and collects results.

    Checks are grouped into four categories:
      A - Schema / Constraint Validation
      B - Data Quality
      C - Business Rules
      D - Distribution / Sanity
    """

    def __init__(self) -> None:
        """Initialize with an empty results list."""
        self.results: list[CheckResult] = []

    def _record(
        self,
        name: str,
        category: str,
        severity: Severity,
        passed: bool,
        detail: str,
    ) -> None:
        """Append a CheckResult to self.results."""
        self.results.append(CheckResult(name, category, severity, passed, detail))

    # ── A: Schema / Constraint Validation ─────────────────────────────────

    async def _check_not_null_columns(self, conn) -> None:
        """Check that required columns have no NULL values."""
        required_cols = [
            "name", "state", "source_url", "source",
            "accepting_new_clients", "is_active",
        ]
        for col in required_cols:
            count = await conn.fetchval(
                f"SELECT COUNT(*) FROM therapists WHERE {col} IS NULL"
            )
            passed = count == 0
            self._record(
                name=f"A1_not_null_{col}",
                category="Schema / Constraint Validation",
                severity=Severity.FAIL,
                passed=passed,
                detail=f"{count} NULL violation(s)" if not passed else "0 NULL violations",
            )

    async def _check_unique_source_url(self, conn) -> None:
        """Check that source_url has no duplicate values."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM (
                SELECT source_url FROM therapists
                GROUP BY source_url HAVING COUNT(*) > 1
            ) dups
            """
        )
        passed = count == 0
        self._record(
            name="A2_unique_source_url",
            category="Schema / Constraint Validation",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} duplicate source_url(s)" if not passed else "no duplicates",
        )

    async def _check_fee_range(self, conn) -> None:
        """Check that fee_min <= fee_max wherever both are set."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM therapists
            WHERE fee_min IS NOT NULL AND fee_max IS NOT NULL AND fee_min > fee_max
            """
        )
        passed = count == 0
        self._record(
            name="A3_fee_min_lte_fee_max",
            category="Schema / Constraint Validation",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with fee_min > fee_max" if not passed else "all fee ranges valid",
        )

    async def _check_rating_range(self, conn) -> None:
        """Check that rating is between 0 and 5."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM therapists
            WHERE rating IS NOT NULL AND (rating < 0 OR rating > 5)
            """
        )
        passed = count == 0
        self._record(
            name="A4_rating_range",
            category="Schema / Constraint Validation",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with rating outside [0,5]" if not passed else "all ratings in range",
        )

    async def _check_required_indexes(self, conn) -> None:
        """Check that all 6 required indexes exist on the therapists table."""
        required = [
            "therapists_embedding_hnsw_idx",
            "therapists_modalities_gin_idx",
            "therapists_insurance_gin_idx",
            "therapists_session_formats_gin_idx",
            "therapists_active_accepting_idx",
            "therapists_location_idx",
        ]
        rows = await conn.fetch(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'therapists' AND indexname = ANY($1)
            """,
            required,
        )
        found = {r["indexname"] for r in rows}
        missing = [idx for idx in required if idx not in found]
        passed = len(missing) == 0
        self._record(
            name="A5_required_indexes",
            category="Schema / Constraint Validation",
            severity=Severity.FAIL,
            passed=passed,
            detail=(
                f"{len(found)}/{len(required)} indexes found (missing: {', '.join(missing)})"
                if not passed
                else f"all {len(required)} required indexes present"
            ),
        )

    async def _check_enum_types(self, conn) -> None:
        """Check that all 3 required DB enum types are registered."""
        required_enums = ["session_format", "insurance_provider", "therapy_modality"]
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM pg_type
            WHERE typname = ANY($1) AND typtype = 'e'
            """,
            required_enums,
        )
        passed = count == len(required_enums)
        self._record(
            name="A6_enum_types",
            category="Schema / Constraint Validation",
            severity=Severity.FAIL,
            passed=passed,
            detail=(
                f"{count}/{len(required_enums)} enum types registered"
                if not passed
                else f"all {len(required_enums)} enum types registered"
            ),
        )

    # ── B: Data Quality ────────────────────────────────────────────────────

    async def _check_bio_length(self, conn) -> None:
        """Check that non-NULL bios have at least 20 characters."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE bio IS NOT NULL AND LENGTH(TRIM(bio)) < 20"
        )
        passed = count == 0
        self._record(
            name="B1_bio_min_length",
            category="Data Quality",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with bio shorter than 20 chars" if not passed else "all bios meet minimum length",
        )

    async def _check_profile_completeness_range(self, conn) -> None:
        """Check that profile_completeness is in [0.0, 1.0]."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM therapists
            WHERE profile_completeness < 0.0 OR profile_completeness > 1.0
            """
        )
        passed = count == 0
        self._record(
            name="B2_profile_completeness_range",
            category="Data Quality",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with profile_completeness outside [0,1]" if not passed else "all in range",
        )

    async def _check_no_negative_fees(self, conn) -> None:
        """Check that no fee values are negative."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE fee_min < 0 OR fee_max < 0"
        )
        passed = count == 0
        self._record(
            name="B3_no_negative_fees",
            category="Data Quality",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with negative fees" if not passed else "no negative fees",
        )

    async def _check_fees_not_excessive(self, conn) -> None:
        """Warn if any fees exceed $1000."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE fee_min > 1000 OR fee_max > 1000"
        )
        passed = count == 0
        self._record(
            name="B4_fees_not_excessive",
            category="Data Quality",
            severity=Severity.WARN,
            passed=passed,
            detail=f"{count} row(s) with fee > $1000" if not passed else "no fees over $1000",
        )

    async def _check_source_url_format(self, conn) -> None:
        """Check that all source_urls start with 'http'."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE source_url NOT LIKE 'http%'"
        )
        passed = count == 0
        self._record(
            name="B5_source_url_format",
            category="Data Quality",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} row(s) with invalid source_url format" if not passed else "all source_urls valid",
        )

    async def _check_no_future_timestamps(self, conn) -> None:
        """Warn if any scraped_at timestamps are in the future."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE scraped_at > NOW() + INTERVAL '1 minute'"
        )
        passed = count == 0
        self._record(
            name="B6_no_future_timestamps",
            category="Data Quality",
            severity=Severity.WARN,
            passed=passed,
            detail=f"{count} row(s) with future scraped_at" if not passed else "no future timestamps",
        )

    # ── C: Business Rules ──────────────────────────────────────────────────

    async def _check_all_state_ca(self, conn) -> None:
        """Check that all therapists are in California (state = 'CA')."""
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE state != 'CA'"
        )
        passed = count == 0
        self._record(
            name="C1_all_state_ca",
            category="Business Rules",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} non-CA therapist(s)" if not passed else "all therapists in CA",
        )

    async def _check_embedding_presence(self, conn) -> None:
        """Check that at least 95% of therapists have embeddings."""
        total = await conn.fetchval("SELECT COUNT(*) FROM therapists")
        if total == 0:
            self._record(
                name="C2_embedding_presence",
                category="Business Rules",
                severity=Severity.FAIL,
                passed=True,
                detail="no records (skipped)",
            )
            return
        embedded = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE embedding IS NOT NULL"
        )
        pct = embedded / total * 100
        passed = pct >= 95.0
        self._record(
            name="C2_embedding_presence",
            category="Business Rules",
            severity=Severity.FAIL,
            passed=passed,
            detail=(
                f"{pct:.1f}% embedded — {total - embedded} missing (threshold: 95%)"
                if not passed
                else f"{pct:.1f}% embedded ({embedded}/{total})"
            ),
        )

    async def _check_embedding_dimensions(self, conn) -> None:
        """Check that all non-NULL embeddings have exactly 384 dimensions."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM therapists
            WHERE embedding IS NOT NULL AND vector_dims(embedding) != 384
            """
        )
        passed = count == 0
        self._record(
            name="C3_embedding_dimensions",
            category="Business Rules",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} embedding(s) with wrong dimensions" if not passed else "all embeddings are 384-dim",
        )

    async def _check_no_zero_embeddings(self, conn) -> None:
        """Check that no embeddings are zero/near-zero vectors."""
        zero_vec = "[" + ",".join(["0"] * 384) + "]"
        count = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM therapists
            WHERE embedding IS NOT NULL
              AND embedding <-> '{zero_vec}'::vector < 0.01
            """
        )
        passed = count == 0
        self._record(
            name="C4_no_zero_embeddings",
            category="Business Rules",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} near-zero embedding(s) detected" if not passed else "no zero embeddings",
        )

    async def _check_active_therapists_have_modalities(self, conn) -> None:
        """Warn if more than 10% of active therapists have no modalities."""
        total_active = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE is_active"
        )
        if total_active == 0:
            self._record(
                name="C5_active_therapists_have_modalities",
                category="Business Rules",
                severity=Severity.WARN,
                passed=True,
                detail="no active records (skipped)",
            )
            return
        no_modalities = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE is_active AND (modalities IS NULL OR modalities = '{}')"
        )
        pct = no_modalities / total_active * 100
        passed = pct < 10.0
        self._record(
            name="C5_active_therapists_have_modalities",
            category="Business Rules",
            severity=Severity.WARN,
            passed=passed,
            detail=(
                f"{pct:.1f}% of active therapists have no modalities ({no_modalities}/{total_active})"
                if not passed
                else f"{pct:.1f}% active therapists without modalities (under 10% threshold)"
            ),
        )

    async def _check_lat_lon_ca_bounds(self, conn) -> None:
        """Warn if any lat/lon coordinates are outside California bounding box."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM therapists
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
              AND NOT (latitude BETWEEN 32.5 AND 42.0 AND longitude BETWEEN -124.5 AND -114.1)
            """
        )
        passed = count == 0
        self._record(
            name="C6_lat_lon_ca_bounds",
            category="Business Rules",
            severity=Severity.WARN,
            passed=passed,
            detail=f"{count} coordinate(s) outside CA bounding box" if not passed else "all coordinates within CA",
        )

    async def _check_no_orphaned_feedback(self, conn) -> None:
        """Check that all feedback rows have a valid therapist_id FK."""
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM feedback f
            WHERE NOT EXISTS (
                SELECT 1 FROM therapists WHERE id = f.therapist_id
            )
            """
        )
        passed = count == 0
        self._record(
            name="C7_no_orphaned_feedback",
            category="Business Rules",
            severity=Severity.FAIL,
            passed=passed,
            detail=f"{count} orphaned feedback row(s)" if not passed else "no orphaned feedback",
        )

    # ── D: Distribution / Sanity ───────────────────────────────────────────

    async def _check_totals_summary(self, conn) -> None:
        """Report total / active / accepting / embedded counts."""
        total = await conn.fetchval("SELECT COUNT(*) FROM therapists")
        active = await conn.fetchval("SELECT COUNT(*) FROM therapists WHERE is_active")
        accepting = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE is_active AND accepting_new_clients"
        )
        embedded = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE embedding IS NOT NULL"
        )
        self._record(
            name="D1_totals_summary",
            category="Distribution / Sanity",
            severity=Severity.INFO,
            passed=True,
            detail=f"total={total}, active={active}, accepting={accepting}, embedded={embedded}",
        )

    async def _check_source_distribution(self, conn) -> None:
        """Report counts by source; warn if any source exceeds 80%."""
        total = await conn.fetchval("SELECT COUNT(*) FROM therapists")
        rows = await conn.fetch(
            "SELECT source, COUNT(*) AS cnt FROM therapists GROUP BY source ORDER BY cnt DESC"
        )
        dist = {r["source"]: r["cnt"] for r in rows}
        detail = ", ".join(f"{src}={cnt}" for src, cnt in dist.items())
        dominant = total > 0 and any(cnt / total > 0.80 for cnt in dist.values())
        self._record(
            name="D2_source_distribution",
            category="Distribution / Sanity",
            severity=Severity.WARN if dominant else Severity.INFO,
            passed=not dominant,
            detail=detail + (" — WARNING: one source exceeds 80%" if dominant else ""),
        )

    async def _check_top_cities(self, conn) -> None:
        """Report the top 10 cities by therapist count."""
        rows = await conn.fetch(
            "SELECT city, COUNT(*) AS cnt FROM therapists GROUP BY city ORDER BY cnt DESC LIMIT 10"
        )
        detail = ", ".join(f"{r['city']}={r['cnt']}" for r in rows)
        self._record(
            name="D3_top_cities",
            category="Distribution / Sanity",
            severity=Severity.INFO,
            passed=True,
            detail=detail or "no data",
        )

    async def _check_embedding_coverage_by_source(self, conn) -> None:
        """Warn if any source has less than 90% embedding coverage."""
        rows = await conn.fetch(
            """
            SELECT source,
                   COUNT(*) AS total,
                   COUNT(embedding) AS embedded
            FROM therapists
            GROUP BY source
            """
        )
        low_coverage = []
        details = []
        for r in rows:
            pct = r["embedded"] / r["total"] * 100 if r["total"] > 0 else 100.0
            details.append(f"{r['source']}={pct:.1f}%")
            if pct < 90.0:
                low_coverage.append(r["source"])
        passed = len(low_coverage) == 0
        self._record(
            name="D4_embedding_coverage_by_source",
            category="Distribution / Sanity",
            severity=Severity.WARN,
            passed=passed,
            detail=(
                ", ".join(details)
                + (f" — low coverage: {', '.join(low_coverage)}" if low_coverage else "")
            ),
        )

    async def _check_data_recency(self, conn) -> None:
        """Warn if the most recently scraped record is more than 7 days old."""
        age = await conn.fetchval(
            "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(scraped_at)))/86400 FROM therapists"
        )
        if age is None:
            self._record(
                name="D5_data_recency",
                category="Distribution / Sanity",
                severity=Severity.WARN,
                passed=True,
                detail="no records (skipped)",
            )
            return
        passed = age <= 7.0
        self._record(
            name="D5_data_recency",
            category="Distribution / Sanity",
            severity=Severity.WARN,
            passed=passed,
            detail=f"newest record is {age:.1f} day(s) old" + (" — stale!" if not passed else ""),
        )

    async def _check_avg_profile_completeness(self, conn) -> None:
        """Warn if average profile_completeness is below 0.6."""
        avg = await conn.fetchval(
            "SELECT AVG(profile_completeness) FROM therapists"
        )
        if avg is None:
            self._record(
                name="D6_avg_profile_completeness",
                category="Distribution / Sanity",
                severity=Severity.WARN,
                passed=True,
                detail="no records (skipped)",
            )
            return
        passed = float(avg) >= 0.6
        self._record(
            name="D6_avg_profile_completeness",
            category="Distribution / Sanity",
            severity=Severity.WARN,
            passed=passed,
            detail=f"avg profile_completeness = {float(avg):.3f}" + (" — below 0.6 threshold" if not passed else ""),
        )

    # ── Entry point ────────────────────────────────────────────────────────

    async def run_all(self) -> list[CheckResult]:
        """Run all checks and return results."""
        async with get_connection() as conn:
            # A — Schema / Constraint
            await self._check_not_null_columns(conn)
            await self._check_unique_source_url(conn)
            await self._check_fee_range(conn)
            await self._check_rating_range(conn)
            await self._check_required_indexes(conn)
            await self._check_enum_types(conn)
            # B — Data Quality
            await self._check_bio_length(conn)
            await self._check_profile_completeness_range(conn)
            await self._check_no_negative_fees(conn)
            await self._check_fees_not_excessive(conn)
            await self._check_source_url_format(conn)
            await self._check_no_future_timestamps(conn)
            # C — Business Rules
            await self._check_all_state_ca(conn)
            await self._check_embedding_presence(conn)
            await self._check_embedding_dimensions(conn)
            await self._check_no_zero_embeddings(conn)
            await self._check_active_therapists_have_modalities(conn)
            await self._check_lat_lon_ca_bounds(conn)
            await self._check_no_orphaned_feedback(conn)
            # D — Distribution / Sanity
            await self._check_totals_summary(conn)
            await self._check_source_distribution(conn)
            await self._check_top_cities(conn)
            await self._check_embedding_coverage_by_source(conn)
            await self._check_data_recency(conn)
            await self._check_avg_profile_completeness(conn)
        return self.results


def print_report(results: list[CheckResult], fail_only: bool = False) -> None:
    """Print a formatted pass/fail report to stdout."""
    settings = get_settings()
    db_url = str(settings.database_url)
    # Redact password from display
    try:
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        display_url = db_url.replace(parsed.password or "", "****") if parsed.password else db_url
    except Exception:
        display_url = db_url

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 65)
    print(" THERAPIZE DB VERIFICATION REPORT")
    print(f" Timestamp : {now}")
    print(f" Database  : {display_url}")
    print("=" * 65)

    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for cat, checks in categories.items():
        visible = [c for c in checks if not fail_only or not c.passed]
        if not visible:
            continue
        print(f"\n[{cat}]")
        for c in visible:
            status = "PASS" if c.passed else c.severity.value
            print(f"  {status:<4}  {c.name:<45} {c.detail}")

    total = len(results)
    passes = sum(1 for r in results if r.passed)
    warns = sum(1 for r in results if not r.passed and r.severity == Severity.WARN)
    fails = sum(1 for r in results if not r.passed and r.severity == Severity.FAIL)

    print()
    print("=" * 65)
    print(f" SUMMARY: {total} checks | {passes} PASS | {warns} WARN | {fails} FAIL")
    status_line = "PASSED" if fails == 0 else f"FAILED (exit code 1)"
    print(f" STATUS: {status_line}")
    print("=" * 65)


async def main() -> None:
    """Connect, run all checks, print report, and exit with appropriate code."""
    import argparse
    parser = argparse.ArgumentParser(description="Verify Therapize DB integrity after ingestion")
    parser.add_argument("--fail-only", action="store_true", help="Only show non-passing checks")
    args = parser.parse_args()

    try:
        await init_db()
    except Exception as exc:
        print(f"ERROR: Cannot connect to database — {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        verifier = DatabaseVerifier()
        results = await verifier.run_all()
    finally:
        await close_db()

    print_report(results, fail_only=args.fail_only)

    has_fails = any(not r.passed and r.severity == Severity.FAIL for r in results)
    sys.exit(1 if has_fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
