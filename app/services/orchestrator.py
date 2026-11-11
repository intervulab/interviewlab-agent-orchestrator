"""
Orchestrator Service
---------------------
This is the BRAIN of Component 4. It coordinates all three agents:
  1. Technical Interviewer  — generates + evaluates questions
  2. Devil's Advocate       — hardens questions behind the scenes
  3. HR Agent               — generates HR questions + final assessment

It also manages in-memory session state (replace with Redis in production).

Flow for a new interview session:
  StartInterviewRequest
       │
       ▼
  Technical Interviewer generates N questions
       │
       ▼  (for each question, run in parallel)
  Devil's Advocate debate loop → hardened questions
       │
       ▼
  Return hardened questions + debate log to frontend
       │
       ▼  (candidate answers one by one via /answer)
  Technical Interviewer evaluates each answer
       │
       ▼  (when all answered)
  HR Agent produces final assessment
"""

import uuid
import asyncio
from typing import Optional

from app.core.config import get_settings
from app.models.schemas import (
    Question, DebateEntry, EvaluationResult,
    InterviewRound, StartInterviewResponse, AnswerResponse,
    SessionSummaryResponse,
)
from app.agents.technical_interviewer import generate_technical_questions, evaluate_answer
from app.agents.devils_advocate import run_debate_loop
from app.agents.hr_agent import generate_hr_questions, generate_hr_assessment

settings = get_settings()


def sanitize_text(text: str) -> str:
    """Normalize tricky unicode characters (non-breaking hyphens, math symbols, em-dashes, curly quotes) to clean ASCII."""
    if not text:
        return ""
    replacements = {
        "≤": "<=",
        "≥": ">=",
        "≈": "~",
        "≠": "!=",
        "–": "-",
        "—": "--",
        "‑": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "\u202f": " ",
        "\u00a0": " ",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode('ascii', errors='replace').decode('ascii')


# ---------------------------------------------------------------------------
# In-memory session store
# In production, replace this dict with a Redis hash per session_id
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Public Orchestrator Functions
# ---------------------------------------------------------------------------

async def start_interview_session(
    candidate_name: str,
    job_role: str,
    resume_text: str,
    round_type: InterviewRound,
    tech_stack: list[str],
    session_id: Optional[str] = None,
) -> StartInterviewResponse:
    """
    Kick off a full interview session.
    Steps:
      1. Generate technical OR HR questions based on round_type.
      2. Run the debate loop on EVERY question in parallel.
      3. Store session state.
      4. Return hardened questions + debate log.
    """
    session_id = session_id or str(uuid.uuid4())
    num_q = settings.MAX_QUESTIONS_PER_ROUND

    # --- Step 1: Generate initial questions ---
    if round_type == InterviewRound.HR:
        raw_questions = await generate_hr_questions(
            candidate_name=candidate_name,
            job_role=job_role,
            resume_text=resume_text,
            num_questions=num_q,
        )
    else:
        # TECHNICAL or SYSTEM_DESIGN — both use the technical interviewer
        raw_questions = await generate_technical_questions(
            candidate_name=candidate_name,
            job_role=job_role,
            tech_stack=tech_stack,
            resume_text=resume_text,
            round_type=round_type.value,
            num_questions=num_q,
        )

    # --- Step 2: Run debate loop on questions with smooth pacing ---
    debate_results: list[tuple[Question, list[DebateEntry]]] = []
    for q in raw_questions:
        try:
            res = await run_debate_loop(
                question=q,
                job_role=job_role,
                tech_stack=tech_stack,
                rounds=settings.MAX_DEBATE_ROUNDS,
            )
            debate_results.append(res)
            # Brief pause to respect Groq tokens-per-minute rate limits
            await asyncio.sleep(0.2)
        except Exception as deb_err:
            debate_results.append((q, []))

    # Unpack and sanitize results for clean database persistence
    hardened_questions: list[Question] = []
    all_debate_logs: list[DebateEntry] = []
    for hardened_q, debate_log in debate_results:
        sanitized_q = Question(
            question_id=hardened_q.question_id,
            text=sanitize_text(hardened_q.text),
            source_agent=hardened_q.source_agent,
            hardened=hardened_q.hardened,
            rationale=sanitize_text(hardened_q.rationale) if hardened_q.rationale else None,
        )
        hardened_questions.append(sanitized_q)
        for entry in debate_log:
            all_debate_logs.append(DebateEntry(
                round_number=entry.round_number,
                original_question=sanitize_text(entry.original_question),
                devils_challenge=sanitize_text(entry.devils_challenge),
                hardened_question=sanitize_text(entry.hardened_question),
            ))

    # --- Step 3: Store session ---
    _sessions[session_id] = {
        "candidate_name": candidate_name,
        "job_role": job_role,
        "resume_text": resume_text,
        "round_type": round_type,
        "tech_stack": tech_stack,
        "questions": hardened_questions,
        "evaluations": [],
        "current_question_index": 0,
        "complete": False,
    }

    return StartInterviewResponse(
        session_id=session_id,
        candidate_name=candidate_name,
        round_type=round_type,
        questions=hardened_questions,
        debate_log=all_debate_logs,
        message=f"Interview session started. {len(hardened_questions)} questions ready.",
    )


async def process_candidate_answer(
    session_id: str,
    question_id: str,
    candidate_answer: str,
    audio_transcript: Optional[str] = None,
) -> AnswerResponse:
    """
    Evaluate a candidate's answer and advance the session.

    If audio_transcript is provided (from Component 5), we append it to the
    answer for a richer evaluation context.
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    if session["complete"]:
        raise ValueError("This session is already complete.")

    # Find the question being answered
    question_map = {q.question_id: q for q in session["questions"]}
    question = question_map.get(question_id)
    if not question:
        raise ValueError(f"Question {question_id} not found in session {session_id}.")

    # If audio transcript exists, enrich the answer
    effective_answer = candidate_answer
    if audio_transcript:
        effective_answer = f"{candidate_answer}\n\n[Audio Transcript]: {audio_transcript}"

    # Evaluate
    raw_eval = await evaluate_answer(
        question_id=question_id,
        question_text=question.text,
        candidate_answer=effective_answer,
        job_role=session["job_role"],
    )
    evaluation = EvaluationResult(
        question_id=raw_eval.question_id,
        question_text=sanitize_text(raw_eval.question_text),
        candidate_answer=sanitize_text(raw_eval.candidate_answer),
        score=raw_eval.score,
        feedback=sanitize_text(raw_eval.feedback),
        follow_up_question=sanitize_text(raw_eval.follow_up_question) if raw_eval.follow_up_question else None,
    )
    session["evaluations"].append(evaluation)

    # Advance question pointer
    session["current_question_index"] += 1
    next_question: Optional[Question] = None
    session_complete = False

    if session["current_question_index"] >= len(session["questions"]):
        session["complete"] = True
        session_complete = True
    else:
        # If there's a follow-up question from a weak answer, insert it next
        if evaluation.follow_up_question:
            follow_up = Question(
                question_id=str(uuid.uuid4()),
                text=evaluation.follow_up_question,
                source_agent="technical_interviewer",
                hardened=False,
                rationale="Follow-up generated due to incomplete answer.",
            )
            # Insert follow-up at current position
            session["questions"].insert(session["current_question_index"], follow_up)

        next_question = session["questions"][session["current_question_index"]]

    return AnswerResponse(
        session_id=session_id,
        evaluation=evaluation,
        next_question=next_question,
        session_complete=session_complete,
    )


async def get_session_summary(session_id: str) -> SessionSummaryResponse:
    """
    Generate the final interview summary with HR assessment.
    Call this after the session is marked complete.
    """
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found.")

    evaluations: list[EvaluationResult] = session["evaluations"]
    scores = [ev.score for ev in evaluations]
    overall_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    # HR Agent produces the final assessment
    hr_data = await generate_hr_assessment(
        candidate_name=session["candidate_name"],
        job_role=session["job_role"],
        tech_score=overall_score,
        evaluations=evaluations,
    )

    return SessionSummaryResponse(
        session_id=session_id,
        candidate_name=session["candidate_name"],
        overall_score=overall_score,
        per_question_results=evaluations,
        hr_assessment=hr_data["assessment"],
        strengths=hr_data["strengths"],
        weaknesses=hr_data["weaknesses"],
        hiring_recommendation=hr_data["recommendation"],
    )
