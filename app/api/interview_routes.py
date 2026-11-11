"""
API Routes — Interview Endpoints
----------------------------------
All endpoints that the Node.js backend (Component 3) calls
to interact with the agent orchestrator.
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    StartInterviewRequest, StartInterviewResponse,
    AnswerRequest, AnswerResponse,
    SessionSummaryResponse,
)
from app.services.orchestrator import (
    start_interview_session,
    process_candidate_answer,
    get_session_summary,
)

router = APIRouter(prefix="/interview", tags=["Interview"])


@router.post(
    "/start",
    response_model=StartInterviewResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new interview session",
    description="""
    Kicks off a new AI-powered interview session.

    **What happens internally:**
    1. The Technical Interviewer generates questions based on the candidate's
       resume, role, and tech stack.
    2. The Devil's Advocate debates each question behind the scenes and hardens it.
    3. The hardened questions + debate log are returned.

    The debate_log field shows the internal back-and-forth — purely for transparency.
    The candidate only sees the final hardened questions.
    """,
)
async def start_interview(payload: StartInterviewRequest) -> StartInterviewResponse:
    try:
        return await start_interview_session(
            candidate_name=payload.candidate_name,
            job_role=payload.job_role,
            resume_text=payload.resume_text,
            round_type=payload.round_type,
            tech_stack=payload.tech_stack,
            session_id=payload.session_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start interview session: {str(e)}",
        )


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Submit a candidate answer",
    description="""
    Submit the candidate's answer to the current question.

    **What happens internally:**
    1. The Technical Interviewer evaluates the answer (score 0–10 + feedback).
    2. If the score is < 7, a follow-up question is generated and inserted next.
    3. Returns the evaluation + the next question (or signals session completion).

    Pass `audio_transcript` from Component 5 for enriched evaluation context.
    """,
)
async def submit_answer(payload: AnswerRequest) -> AnswerResponse:
    try:
        return await process_candidate_answer(
            session_id=payload.session_id,
            question_id=payload.question_id,
            candidate_answer=payload.candidate_answer,
            audio_transcript=payload.audio_transcript,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process answer: {str(e)}",
        )


@router.get(
    "/summary/{session_id}",
    response_model=SessionSummaryResponse,
    summary="Get session summary and hiring recommendation",
    description="""
    Call this after the session is complete (`session_complete: true` from /answer).

    **What happens internally:**
    1. Aggregates all evaluation scores.
    2. HR Agent generates a holistic assessment and hiring recommendation.

    Returns: overall score, per-question breakdown, strengths, weaknesses,
    and a final hiring recommendation (Strong Hire | Hire | Borderline | No Hire).
    """,
)
async def get_summary(session_id: str) -> SessionSummaryResponse:
    try:
        return await get_session_summary(session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}",
        )
