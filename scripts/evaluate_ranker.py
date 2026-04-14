"""
Offline ranker evaluation script.

Computes NDCG@5, MRR, and Precision@5 from stored feedback data,
then writes results to the evaluation_runs table for trend tracking.

Usage:
    python scripts/evaluate_ranker.py              # evaluate + save to DB
    python scripts/evaluate_ranker.py --dry-run    # evaluate only, no DB write
    python scripts/evaluate_ranker.py --history    # print past evaluation runs

Relevance grade mapping:
    5 (👍 good match)   → 3  explicit positive
    2 (profile view)    → 2  implicit positive
    1 (👎 poor match)   → 0  explicit negative
    no feedback         → 1  shown but ignored (inferred from result_ids)

Requires: DATABASE_URL env var pointing to a live Postgres instance.
"""
import argparse
import asyncio
import json
import math
import sys
from datetime import datetime
from statistics import mean
from uuid import uuid4

import asyncpg

sys.path.insert(0, ".")
from src.config import get_settings
from src.storage.database import _clean_dsn

settings = get_settings()


# ── Grade mapping ──────────────────────────────────────────────────────────────

def rating_to_grade(rating: int, event_type: str) -> int:
    """Convert a feedback row into a relevance grade (0–3)."""
    if event_type == "profile_view":
        return 2
    return {5: 3, 1: 0}.get(rating, 1)


# ── Metric functions ───────────────────────────────────────────────────────────

def dcg(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    return sum(
        r / math.log2(i + 2)
        for i, r in enumerate(relevances[:k])
    )


def ndcg(relevances: list[int], k: int) -> float:
    """Normalized DCG at rank k. Returns 0.0 if no relevant results exist."""
    ideal = dcg(sorted(relevances, reverse=True), k)
    return dcg(relevances, k) / ideal if ideal > 0 else 0.0


def mrr(relevance_lists: list[list[int]], threshold: int = 2) -> float:
    """
    Mean Reciprocal Rank.
    threshold: minimum grade to count as a relevant result (default 2 = profile view or better).
    """
    scores = []
    for relevances in relevance_lists:
        for rank, grade in enumerate(relevances, start=1):
            if grade >= threshold:
                scores.append(1.0 / rank)
                break
        else:
            scores.append(0.0)
    return mean(scores) if scores else 0.0


def precision_at_k(relevances: list[int], k: int, threshold: int = 2) -> float:
    """Fraction of top-k results with grade >= threshold."""
    top_k = relevances[:k]
    return sum(1 for r in top_k if r >= threshold) / k if top_k else 0.0


# ── Data loading ───────────────────────────────────────────────────────────────

async def load_evaluation_data(conn: asyncpg.Connection) -> list[dict]:
    """
    Load all queries that have at least one feedback event.

    For each query, reconstructs the full relevance list by:
    - Starting every ranked result at grade 1 (shown, no action)
    - Overwriting with the actual feedback grade where present

    Returns list of dicts: {query_id, free_text, relevances: list[int]}
    """
    # Pull queries with feedback
    rows = await conn.fetch(
        """
        SELECT
            sq.id          AS query_id,
            sq.free_text,
            sq.result_ids,
            f.therapist_id AS fb_therapist_id,
            f.rating,
            f.rank_position,
            f.event_type
        FROM search_queries sq
        JOIN feedback f ON f.query_id = sq.id
        WHERE sq.result_ids IS NOT NULL
          AND array_length(sq.result_ids, 1) > 0
        ORDER BY sq.id, f.rank_position NULLS LAST
        """
    )

    # Group by query
    queries: dict[str, dict] = {}
    for row in rows:
        qid = str(row["query_id"])
        if qid not in queries:
            result_ids = list(row["result_ids"] or [])
            queries[qid] = {
                "query_id": qid,
                "free_text": row["free_text"] or "",
                "result_ids": result_ids,
                # Start everyone at grade 1 (shown, no feedback)
                "feedback": {str(rid): 1 for rid in result_ids},
            }

        # Overwrite with actual feedback grade
        therapist_id = str(row["fb_therapist_id"])
        grade = rating_to_grade(row["rating"], row["event_type"] or "explicit")
        # Take the highest grade if multiple feedback events exist for the same therapist
        existing = queries[qid]["feedback"].get(therapist_id, 0)
        queries[qid]["feedback"][therapist_id] = max(existing, grade)

    # Flatten into ordered relevance lists
    results = []
    for q in queries.values():
        relevances = [
            q["feedback"].get(str(rid), 1)
            for rid in q["result_ids"]
        ]
        results.append({
            "query_id": q["query_id"],
            "free_text": q["free_text"],
            "relevances": relevances,
        })

    return results


# ── Evaluation run ─────────────────────────────────────────────────────────────

async def run_evaluation(dry_run: bool = False) -> None:
    """Load feedback data, compute metrics, print report, optionally save to DB."""
    dsn = _clean_dsn(str(settings.database_url))
    conn = await asyncpg.connect(dsn)

    try:
        data = await load_evaluation_data(conn)

        if not data:
            print("No feedback data found. Run some searches and submit feedback first.")
            return

        relevance_lists = [d["relevances"] for d in data]

        ndcg5_scores  = [ndcg(r, k=5) for r in relevance_lists]
        ndcg10_scores = [ndcg(r, k=10) for r in relevance_lists]
        p5_scores     = [precision_at_k(r, k=5) for r in relevance_lists]
        mrr_score     = mrr(relevance_lists)

        ndcg5_mean  = mean(ndcg5_scores)
        ndcg10_mean = mean(ndcg10_scores)
        p5_mean     = mean(p5_scores)

        # ── Report ────────────────────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Therapize Ranker Evaluation — {datetime.now():%Y-%m-%d %H:%M}")
        print(f"{'='*50}")
        print(f"  Queries evaluated : {len(data)}")
        print(f"  NDCG@5            : {ndcg5_mean:.4f}")
        print(f"  NDCG@10           : {ndcg10_mean:.4f}")
        print(f"  MRR               : {mrr_score:.4f}")
        print(f"  Precision@5       : {p5_mean:.4f}")
        print(f"{'='*50}\n")

        # ── Per-query breakdown ───────────────────────────────────────────────
        print("Per-query breakdown:")
        print(f"  {'Query':<45} {'NDCG@5':>7} {'P@5':>6}")
        print(f"  {'-'*45} {'-'*7} {'-'*6}")
        for d, n5, p5 in sorted(
            zip(data, ndcg5_scores, p5_scores),
            key=lambda x: x[1],  # sort by NDCG@5 ascending (worst first)
        ):
            label = d["free_text"][:44] or "(no text)"
            print(f"  {label:<45} {n5:>7.4f} {p5:>6.4f}")

        print()

        # ── Save to DB ────────────────────────────────────────────────────────
        if not dry_run:
            config_snapshot = {
                "weight_modality": settings.weight_modality,
                "weight_semantic": settings.weight_semantic,
                "weight_bm25":     settings.weight_bm25,
                "weight_recency":  settings.weight_recency,
                "weight_rating":   settings.weight_rating,
                "llm_model":       settings.llm_model,
                "embedding_model": settings.embedding_model,
            }
            await conn.execute(
                """
                INSERT INTO evaluation_runs
                    (id, run_name, ndcg_at_10, mrr, precision_at_5, num_queries, config_snapshot)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                uuid4(),
                f"eval_{datetime.now():%Y%m%d_%H%M}",
                round(ndcg10_mean, 4),
                round(mrr_score, 4),
                round(p5_mean, 4),
                len(data),
                json.dumps(config_snapshot),
            )
            print("Results saved to evaluation_runs table.")
        else:
            print("Dry run — results not saved.")

    finally:
        await conn.close()


async def print_history() -> None:
    """Print past evaluation runs from the evaluation_runs table."""
    dsn = _clean_dsn(str(settings.database_url))
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT run_name, ndcg_at_10, mrr, precision_at_5, num_queries, created_at
            FROM evaluation_runs
            ORDER BY created_at DESC
            LIMIT 20
            """
        )
        if not rows:
            print("No evaluation runs recorded yet.")
            return

        print(f"\n{'Run':<25} {'NDCG@10':>8} {'MRR':>7} {'P@5':>6} {'Queries':>8} {'Date'}")
        print(f"{'-'*25} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*20}")
        for r in rows:
            print(
                f"{r['run_name']:<25} "
                f"{r['ndcg_at_10']:>8.4f} "
                f"{r['mrr']:>7.4f} "
                f"{r['precision_at_5']:>6.4f} "
                f"{r['num_queries']:>8} "
                f"{r['created_at']:%Y-%m-%d %H:%M}"
            )
        print()
    finally:
        await conn.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Therapize ranker offline.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics but don't save to DB")
    parser.add_argument("--history", action="store_true", help="Print past evaluation run history")
    args = parser.parse_args()

    if args.history:
        asyncio.run(print_history())
    else:
        asyncio.run(run_evaluation(dry_run=args.dry_run))
