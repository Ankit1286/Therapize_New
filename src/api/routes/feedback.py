"""Feedback collection routes — closes the evaluation feedback loop."""
import logging

from fastapi import APIRouter, HTTPException

from src.models.query import FeedbackRequest
from src.monitoring.metrics import feedback_rating
from src.monitoring.tracing import StructuredLogger
from src.storage.database import TherapistRepository

logger = logging.getLogger(__name__)
structured_log = StructuredLogger(__name__)
router = APIRouter()
_repository = TherapistRepository()


@router.post("/feedback", status_code=201)
async def submit_feedback(feedback: FeedbackRequest) -> dict:
    """
    Submit user feedback on search results.

    Feedback is used in the offline evaluation pipeline to:
    1. Compute NDCG@10 and MRR scores
    2. Identify poorly-ranked therapists
    3. Guide prompt and weight tuning

    Rating scale: 1=poor match, 5=excellent match
    """
    try:
        # Track in Prometheus for real-time monitoring
        feedback_rating.observe(feedback.rating)

        # Log for offline analysis
        structured_log.feedback_received(
            str(feedback.query_id),
            str(feedback.therapist_id),
            feedback.rating,
            feedback.booked_appointment,
        )

        # Store in DB for evaluation pipeline
        from src.storage.database import get_connection
        async with get_connection() as conn:
            await conn.execute(
                """
                INSERT INTO feedback
                    (query_id, therapist_id, rating, rank_position, event_type, booked, feedback_text)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                feedback.query_id,
                feedback.therapist_id,
                feedback.rating,
                feedback.rank_position,
                feedback.event_type,
                feedback.booked_appointment,
                feedback.feedback_text,
            )

        return {"status": "recorded", "query_id": str(feedback.query_id)}

    except Exception as exc:
        logger.exception("Feedback submission failed")
        raise HTTPException(status_code=500, detail="Failed to record feedback") from exc
