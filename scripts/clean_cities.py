"""
City name cleanup script.

Fixes two categories of dirty data in the therapists.city column:

  1. Typos / formatting variants  → corrected city name
  2. Non-CA locations / garbage   → NULL (treated as telehealth-only)

Dry-run by default. Pass --apply to commit changes to the database.

Usage:
    python scripts/clean_cities.py            # preview changes
    python scripts/clean_cities.py --apply    # apply changes
"""
import argparse
import asyncio
import sys

import asyncpg

sys.path.insert(0, ".")
from src.config import get_settings
from src.storage.database import _clean_dsn

# ── Correction map ─────────────────────────────────────────────────────────────
# None = set city to NULL (becomes telehealth-only, not deactivated)
CORRECTIONS: dict[str, str | None] = {
    # Typos
    "Los Angeles, Ca":  "Los Angeles",
    "Los Angles":       "Los Angeles",
    "Los Alomitos":     "Los Alamitos",
    "Murrietta":        "Murrieta",
    "San Clamente":     "San Clemente",
    "San Diego, Ca":    "San Diego",
    "Pasadena Ca":      "Pasadena",
    "Truckee, Ca":      "Truckee",
    "N. Hollywood":     "North Hollywood",
    "Mogan Hill":       "Morgan Hill",

    # Not a specific city — scraping artifact
    "Re":               None,
    "Sf Bay Area":      None,
    "California":       None,

    # Non-California locations
    "New York":         None,
    "Seattle":          None,
    "Paris":            None,
    "Gurnee":           None,   # Gurnee, IL
    "White Plains":     None,   # White Plains, NY
    "Woodstock":        None,   # not a CA city
}


async def run(apply: bool) -> None:
    """Preview or apply all city corrections."""
    conn = await asyncpg.connect(_clean_dsn(str(get_settings().database_url)))

    try:
        # Count affected rows per correction
        print(f"\n{'='*60}")
        print(f"  City cleanup -- {'APPLYING' if apply else 'DRY RUN (pass --apply to commit)'}")
        print(f"{'='*60}\n")

        total_affected = 0
        corrections_with_counts = []

        for bad, good in CORRECTIONS.items():
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM therapists WHERE city = $1", bad
            )
            if count > 0:
                label = f'"{bad}" -> "{good}"' if good else f'"{bad}" -> NULL (telehealth-only)'
                corrections_with_counts.append((bad, good, count, label))
                total_affected += count

        if not corrections_with_counts:
            print("  No dirty city values found -- data is already clean.")
            return

        # Print preview
        print(f"  {'Change':<50} {'Rows':>5}")
        print(f"  {'-'*50} {'-'*5}")
        for _, _, count, label in corrections_with_counts:
            print(f"  {label:<50} {count:>5}")
        print(f"\n  Total rows affected: {total_affected}\n")

        if not apply:
            print("  Run with --apply to commit these changes.\n")
            return

        # Apply corrections
        for bad, good, count, label in corrections_with_counts:
            await conn.execute(
                "UPDATE therapists SET city = $1 WHERE city = $2",
                good,  # None becomes SQL NULL automatically
                bad,
            )
            print(f"  Applied: {label} ({count} rows)")

        # Verify
        remaining = await conn.fetchval(
            "SELECT COUNT(*) FROM therapists WHERE city = ANY($1::text[])",
            list(CORRECTIONS.keys()),
        )
        print(f"\n  Verification: {remaining} dirty rows remaining (expected 0)")
        print(f"\n  Done. {total_affected} rows updated.\n")

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up dirty city values in therapists table.")
    parser.add_argument("--apply", action="store_true", help="Commit changes (default: dry run)")
    args = parser.parse_args()
    asyncio.run(run(apply=args.apply))
