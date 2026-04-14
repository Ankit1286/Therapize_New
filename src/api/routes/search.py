"""Search API routes."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from src.models.query import SearchRequest, SearchResponse
from src.storage.database import TherapistRepository
from src.workflow.search_graph import SearchWorkflow

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate once — the workflow builds indexes on first use
_workflow = SearchWorkflow()
_repository = TherapistRepository()


@router.get("/cities", response_model=list[str])
async def list_cities() -> list[str]:
    """Return all distinct cities that have at least one active therapist."""
    return await _repository.get_cities()


@router.get("/languages", response_model=list[str])
async def list_languages() -> list[str]:
    """Return all distinct languages spoken by at least one active therapist."""
    return await _repository.get_languages()


@router.get("/stats", response_model=dict)
async def get_stats() -> dict:
    """Return aggregate stats about the therapist database."""
    total = await _repository.get_total_count()
    return {"total_therapists": total}


@router.post("/search", response_model=SearchResponse)
async def search_therapists(
    request_body: SearchRequest,
    request: Request,
) -> SearchResponse:
    """
    Search for therapists matching the user's needs.

    Accepts:
    - free_text: Natural language description ("I have anxiety and work stress")
    - questionnaire: Structured filters (location, insurance, budget)

    Returns ranked list of therapists with scoring breakdown.
    """
    if not request_body.has_content():
        raise HTTPException(
            status_code=400,
            detail="Please provide either a search query or fill in the questionnaire.",
        )

    try:
        result = await _workflow.run(request_body)
        return result
    except Exception as exc:
        logger.exception("Search failed for query %s", request_body.query_id)
        raise HTTPException(status_code=500, detail="Search failed. Please try again.") from exc
