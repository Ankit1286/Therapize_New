"""
Offline evaluation pipeline.

Tracks ranking quality over time using NDCG and MRR.
This is the feedback loop: user ratings → quality metrics → prompt/weight tuning.

Why offline evaluation matters:
- Online metrics (clicks, bookings) are slow signals
- NDCG lets you evaluate ranking quality with any labeled dataset
- Run after every prompt change to catch regressions

Evaluation workflow:
1. Collect queries with user feedback (stored in `feedback` table)
2. Convert ratings to relevance labels (1-5 star → 0-3 relevance)
3. Compute NDCG@10 and MRR across all labeled queries
4. Store results in evaluation_runs table
5. Alert if NDCG drops by > 5% from baseline

NDCG (Normalized Discounted Cumulative Gain):
- Measures ranking quality when you know which results are relevant
- 1.0 = perfect ranking, 0.0 = worst possible ranking
- @10 means we only care about the top 10 results (user behavior cutoff)
"""
import logging
import math
from dataclasses import dataclass
from uuid import UUID

import numpy as np

from src.monitoring.metrics import ndcg_score as ndcg_metric
from src.monitoring.metrics import mrr_score as mrr_metric

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    query_id: UUID
    result_ids: list[UUID]          # ordered: position 0 = rank 1
    relevance_labels: dict[UUID, float]  # therapist_id → relevance (0-3)


class RankingEvaluator:
    """
    Computes NDCG@k and MRR for the search system.

    Run periodically (daily/weekly) or after prompt changes.
    """

    @staticmethod
    def ndcg_at_k(
        result_ids: list[UUID],
        relevance_labels: dict[UUID, float],
        k: int = 10,
    ) -> float:
        """
        Compute NDCG@k for a single query.

        Args:
            result_ids: Ordered list of returned therapist IDs (rank 1 first)
            relevance_labels: Ground truth relevance (0=irrelevant, 3=highly relevant)
            k: Cutoff position

        Returns: NDCG@k score in [0, 1]
        """
        # Discounted Cumulative Gain
        dcg = 0.0
        for i, result_id in enumerate(result_ids[:k]):
            rel = relevance_labels.get(result_id, 0.0)
            # Position discount: log2(rank+1), rank starts at 1
            discount = math.log2(i + 2)
            dcg += (2**rel - 1) / discount

        # Ideal DCG (best possible ordering)
        ideal_rels = sorted(relevance_labels.values(), reverse=True)[:k]
        idcg = sum(
            (2**rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
        )

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def mean_reciprocal_rank(
        result_ids: list[UUID],
        relevant_ids: set[UUID],
    ) -> float:
        """
        MRR: 1/rank of first relevant result.
        Measures "how quickly do we surface a good therapist?"
        """
        for i, result_id in enumerate(result_ids):
            if result_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def rating_to_relevance(rating: int) -> float:
        """
        Convert 1-5 star rating to relevance label for NDCG.
        Scale: 0 (irrelevant) to 3 (highly relevant)
        """
        mapping = {1: 0.0, 2: 0.5, 3: 1.0, 4: 2.0, 5: 3.0}
        return mapping.get(rating, 0.0)

    def evaluate_batch(
        self,
        queries: list[EvaluationQuery],
        k: int = 10,
    ) -> dict[str, float]:
        """
        Compute aggregate metrics across all queries.

        Returns: {ndcg@k, mrr, precision@5, coverage}
        """
        if not queries:
            return {"ndcg_at_k": 0.0, "mrr": 0.0, "precision_at_5": 0.0, "num_queries": 0}

        ndcg_scores = []
        mrr_scores = []
        p5_scores = []

        for query in queries:
            # NDCG@k
            ndcg = self.ndcg_at_k(query.result_ids, query.relevance_labels, k=k)
            ndcg_scores.append(ndcg)

            # MRR (any result with relevance > 1.0 counts as "relevant")
            relevant_ids = {
                tid for tid, rel in query.relevance_labels.items() if rel >= 1.0
            }
            mrr = self.mean_reciprocal_rank(query.result_ids, relevant_ids)
            mrr_scores.append(mrr)

            # Precision@5: fraction of top-5 results that are relevant
            top5 = set(query.result_ids[:5])
            p5 = len(top5 & relevant_ids) / 5 if top5 else 0.0
            p5_scores.append(p5)

        results = {
            "ndcg_at_k": float(np.mean(ndcg_scores)),
            "mrr": float(np.mean(mrr_scores)),
            "precision_at_5": float(np.mean(p5_scores)),
            "num_queries": len(queries),
            "ndcg_std": float(np.std(ndcg_scores)),
        }

        # Update Prometheus gauges for dashboarding
        ndcg_metric.set(results["ndcg_at_k"])
        mrr_metric.set(results["mrr"])

        logger.info(
            "Evaluation results: NDCG@%d=%.4f, MRR=%.4f, P@5=%.4f (n=%d queries)",
            k,
            results["ndcg_at_k"],
            results["mrr"],
            results["precision_at_5"],
            results["num_queries"],
        )

        return results

    def detect_regression(
        self,
        current_ndcg: float,
        baseline_ndcg: float,
        threshold: float = 0.05,
    ) -> bool:
        """
        Returns True if current NDCG has dropped significantly vs baseline.
        Use to gate prompt/weight changes in CI.
        """
        drop = baseline_ndcg - current_ndcg
        if drop > threshold:
            logger.warning(
                "NDCG regression detected: %.4f → %.4f (drop=%.4f, threshold=%.4f)",
                baseline_ndcg, current_ndcg, drop, threshold,
            )
            return True
        return False
