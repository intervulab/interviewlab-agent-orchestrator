"""
API Routes — Canvas Analysis Endpoint
----------------------------------------
Called by Component 2 (Canvas) when the interviewer needs to
review the candidate's system design drawing.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import CanvasStateRequest, CanvasAnalysisResponse
from app.agents.canvas_agent import analyse_canvas

router = APIRouter(prefix="/canvas", tags=["System Design Canvas"])


@router.post(
    "/analyse",
    response_model=CanvasAnalysisResponse,
    summary="Analyse a system design canvas",
    description="""
    Receives the JSON graph (nodes + edges) from Component 2's canvas.
    
    **What happens internally:**
    The Canvas Agent (backed by an LLM) reviews the architecture topology and:
    - Identifies specific flaws linked to component IDs
    - Suggests improvements
    - Gives an overall design assessment

    The `flaws[].component_id` values map directly to node IDs in the canvas,
    so the frontend can flash the correct components.
    """,
)
async def analyse_canvas_route(payload: CanvasStateRequest) -> CanvasAnalysisResponse:
    try:
        result = await analyse_canvas(
            canvas_json=payload.canvas_json,
            job_role="Software Engineer",  # Can be extended with session lookup
        )
        return CanvasAnalysisResponse(
            session_id=payload.session_id,
            flaws=result.get("flaws", []),
            suggestions=result.get("suggestions", []),
            overall_assessment=result.get("overall_assessment", ""),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Canvas analysis failed: {str(e)}",
        )
