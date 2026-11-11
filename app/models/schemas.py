"""
Pydantic request / response schemas used across the API.
Think of these as the "shapes" of data flowing in and out of every endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InterviewRound(str, Enum):
    TECHNICAL = "technical"
    HR = "hr"
    SYSTEM_DESIGN = "system_design"


# ---------------------------------------------------------------------------
# Request Bodies
# ---------------------------------------------------------------------------

class StartInterviewRequest(BaseModel):
    """Sent by the frontend to kick off a fresh interview session."""
    session_id: Optional[str] = Field(
        default=None,
        description="Optional UUID from Component 3 — used when gateway owns session identity",
    )
    candidate_name: str = Field(..., min_length=1, description="Full name of the candidate")
    job_role: str = Field(..., description="e.g. 'Backend Engineer', 'ML Engineer'")
    resume_text: str = Field(default="", description="Plain-text content of the parsed resume (optional)")
    round_type: InterviewRound = Field(default=InterviewRound.TECHNICAL)
    tech_stack: list[str] = Field(default_factory=list, description="e.g. ['Python', 'FastAPI', 'PostgreSQL']")
    num_questions: Optional[int] = Field(default=3, ge=1, le=15, description="Number of interview questions to generate (default 3)")


class AnswerRequest(BaseModel):
    """Sent by the frontend when the candidate submits an answer."""
    session_id: str = Field(..., description="UUID returned by /start")
    question_id: str = Field(..., description="UUID of the question being answered")
    candidate_answer: str = Field(..., min_length=1)
    audio_transcript: Optional[str] = Field(
        default=None,
        description="Optional transcript injected by Component 5's audio pipeline"
    )


class CanvasStateRequest(BaseModel):
    """Sent by Component 2 so the agent can reason about the system-design canvas."""
    session_id: str
    canvas_json: dict = Field(..., description="Node/Edge topology graph from the canvas")


# ---------------------------------------------------------------------------
# Shared Structures
# ---------------------------------------------------------------------------

class Question(BaseModel):
    question_id: str
    text: str
    source_agent: str           # 'technical_interviewer' | 'hr_agent'
    hardened: bool = False      # True once Devil's Advocate has toughened it
    rationale: Optional[str] = None   # why this question was chosen


class DebateEntry(BaseModel):
    """One round of the behind-the-scenes debate loop."""
    round_number: int
    original_question: str
    devils_challenge: str       # Devil's Advocate critique
    hardened_question: str      # final, toughened version


class EvaluationResult(BaseModel):
    question_id: str
    question_text: str
    candidate_answer: str
    score: int = Field(..., ge=0, le=10, description="0–10 score")
    feedback: str
    follow_up_question: Optional[str] = None


# ---------------------------------------------------------------------------
# Response Bodies
# ---------------------------------------------------------------------------

class StartInterviewResponse(BaseModel):
    session_id: str
    candidate_name: str
    round_type: InterviewRound
    questions: list[Question]
    debate_log: list[DebateEntry]   # the internal debate that produced these questions
    message: str


class AnswerResponse(BaseModel):
    session_id: str
    evaluation: EvaluationResult
    next_question: Optional[Question] = None
    session_complete: bool = False


class CanvasAnalysisResponse(BaseModel):
    session_id: str
    flaws: list[dict]       # [{component_id, flaw_description, severity}]
    suggestions: list[str]
    overall_assessment: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    candidate_name: str
    overall_score: float
    per_question_results: list[EvaluationResult]
    hr_assessment: str
    strengths: list[str]
    weaknesses: list[str]
    hiring_recommendation: str
