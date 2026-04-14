"""
Online metrics dashboard for Therapize.

Queries live feedback and search_queries data to produce a real-time
picture of user satisfaction and ranking health.

Usage:
    python scripts/online_metrics.py              # last 7 days
    python scripts/online_metrics.py --days 30    # last 30 days
    python scripts/online_metrics.py --days 1     # yesterday
"""
import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

import asyncpg

sys.path.insert(0, ".")
from src.config import get_settings
from src.storage.database import _clean_dsn

settings = get_settings()


def pct(value: float | None) -> str:
    """Format a 0–1 float as a percentage string."""
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def bar(value: float | None, width: int = 20) -> str:
    """Render a simple ASCII progress bar for a 0–1 value."""
    if value is None:
        return " " * width
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


async def run(days: int) -> None:
    """Fetch and print all online metrics for the given lookback window."""
    dsn = _clean_dsn(str(settings.database_url))
    conn = await asyncpg.connect(dsn)

    since = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        # ── Volume ────────────────────────────────────────────────────────────
        total_searches = await conn.fetchval(
            "SELECT COUNT(*) FROM search_queries WHERE created_at > $1",
            since,
        )
        total_sessions = await conn.fetchval(
            "SELECT COUNT(DISTINCT session_id) FROM search_queries "
            "WHERE created_at > $1 AND session_id IS NOT NULL",
            since,
        )
        cache_hit_rate = await conn.fetchval(
            "SELECT AVG(cache_hit::int) FROM search_queries "
            "WHERE created_at > $1",
            since,
        )
        avg_latency_ms = await conn.fetchval(
            "SELECT AVG(latency_ms) FROM search_queries "
            "WHERE created_at > $1 AND latency_ms > 0",
            since,
        )

        # ── Feedback engagement ───────────────────────────────────────────────
        searches_with_feedback = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT sq.id)
            FROM search_queries sq
            JOIN feedback f ON f.query_id = sq.id
            WHERE sq.created_at > $1
            """,
            since,
        )
        feedback_rate = (
            searches_with_feedback / total_searches if total_searches else None
        )

        thumbs_up_rate = await conn.fetchval(
            """
            SELECT AVG(CASE WHEN rating = 5 THEN 1.0 ELSE 0.0 END)
            FROM feedback
            WHERE event_type = 'explicit'
              AND created_at > $1
            """,
            since,
        )

        searches_with_view = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT query_id)
            FROM feedback
            WHERE event_type = 'profile_view'
              AND created_at > $1
            """,
            since,
        )
        profile_view_rate = (
            searches_with_view / total_searches if total_searches else None
        )

        # ── Pogo-sticking (reformulation) ─────────────────────────────────────
        session_stats = await conn.fetchrow(
            """
            SELECT
                AVG(query_count)        AS avg_queries_per_session,
                COUNT(CASE WHEN query_count > 1 THEN 1 END) AS multi_query_sessions,
                COUNT(*)                AS total_sessions
            FROM (
                SELECT session_id, COUNT(*) AS query_count
                FROM search_queries
                WHERE created_at > $1
                  AND session_id IS NOT NULL
                GROUP BY session_id
            ) s
            """,
            since,
        )

        # ── Position bias ─────────────────────────────────────────────────────
        position_rows = await conn.fetch(
            """
            SELECT
                rank_position,
                COUNT(*)                                            AS total,
                AVG(CASE WHEN rating = 5 THEN 1.0 ELSE 0.0 END)   AS thumbs_up_rate,
                AVG(CASE WHEN event_type = 'profile_view'
                         THEN 1.0 ELSE 0.0 END)                    AS view_rate
            FROM feedback
            WHERE rank_position IS NOT NULL
              AND created_at > $1
            GROUP BY rank_position
            ORDER BY rank_position
            LIMIT 10
            """,
            since,
        )

        # ── Top reformulated queries (pogo candidates) ────────────────────────
        reformulation_rows = await conn.fetch(
            """
            SELECT session_id, COUNT(*) AS query_count, array_agg(free_text) AS queries
            FROM search_queries
            WHERE created_at > $1
              AND session_id IS NOT NULL
            GROUP BY session_id
            HAVING COUNT(*) > 1
            ORDER BY query_count DESC
            LIMIT 5
            """,
            since,
        )

        # ── Zero-result queries ───────────────────────────────────────────────
        zero_result_rate = await conn.fetchval(
            """
            SELECT AVG(CASE WHEN filtered_count = 0 THEN 1.0 ELSE 0.0 END)
            FROM search_queries
            WHERE created_at > $1
            """,
            since,
        )

    finally:
        await conn.close()

    # ── Print report ──────────────────────────────────────────────────────────
    w = 55
    print(f"\n{'='*w}")
    print(f"  Therapize Online Metrics -- last {days} day{'s' if days != 1 else ''}")
    print(f"{'='*w}")

    print(f"\n  VOLUME")
    print(f"  {'Total searches':<30} {total_searches or 0:>8}")
    print(f"  {'Unique sessions':<30} {total_sessions or 0:>8}")
    print(f"  {'Cache hit rate':<30} {pct(float(cache_hit_rate) if cache_hit_rate else None):>8}")
    print(f"  {'Avg latency':<30} {f'{avg_latency_ms:.0f}ms' if avg_latency_ms else 'n/a':>8}")
    print(f"  {'Zero-result rate':<30} {pct(float(zero_result_rate) if zero_result_rate else None):>8}")

    print(f"\n  ENGAGEMENT")
    print(f"  {'Feedback rate':<30} {pct(float(feedback_rate) if feedback_rate else None):>8}  {bar(float(feedback_rate) if feedback_rate else None)}")
    print(f"  {'Thumbs-up rate (explicit)':<30} {pct(float(thumbs_up_rate) if thumbs_up_rate else None):>8}  {bar(float(thumbs_up_rate) if thumbs_up_rate else None)}")
    print(f"  {'Profile view rate':<30} {pct(float(profile_view_rate) if profile_view_rate else None):>8}  {bar(float(profile_view_rate) if profile_view_rate else None)}")

    print(f"\n  SESSION QUALITY")
    avg_q = session_stats["avg_queries_per_session"]
    multi = session_stats["multi_query_sessions"] or 0
    total_s = session_stats["total_sessions"] or 1
    print(f"  {'Avg queries / session':<30} {f'{float(avg_q):.2f}' if avg_q else 'n/a':>8}")
    print(f"  {'Multi-query sessions':<30} {multi:>8}  ({pct(multi/total_s)} of sessions)")

    if reformulation_rows:
        print(f"\n  TOP REFORMULATING SESSIONS (potential dissatisfaction)")
        for row in reformulation_rows:
            queries_preview = " -> ".join(
                (q[:30] + "..." if q and len(q) > 30 else q or "(empty)")
                .encode("ascii", errors="replace").decode("ascii")
                for q in (row["queries"] or [])[:3]
            )
            print(f"  [{row['query_count']} queries] {queries_preview}")

    if position_rows:
        print(f"\n  POSITION BIAS  (👍 rate and profile view rate by rank)")
        print(f"  {'Rank':<6} {'Count':>6}  {'Thumbs-up rate':<28} {'View rate':<28}")
        print(f"  {'-'*6} {'-'*6}  {'-'*28} {'-'*28}")
        for row in position_rows:
            tup = float(row["thumbs_up_rate"]) if row["thumbs_up_rate"] else None
            vr  = float(row["view_rate"]) if row["view_rate"] else None
            print(
                f"  {row['rank_position']:<6} {row['total']:>6}  "
                f"{bar(tup, 16)} {pct(tup):<8}  "
                f"{bar(vr, 16)} {pct(vr):<8}"
            )
    else:
        print(f"\n  POSITION BIAS  -- no feedback with rank_position recorded yet")

    print(f"\n{'='*w}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print Therapize online metrics.")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days (default: 7)")
    args = parser.parse_args()
    asyncio.run(run(args.days))
