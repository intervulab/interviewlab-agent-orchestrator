"""
Integration tests for the Agent Orchestrator.
Run with: pytest tests/ -v

NOTE: These tests call the Groq API.
Set GROQ_API_KEY in your .env before running.
To mock the LLM calls, replace with LangChain's FakeListChatModel.
"""

import pytest
import asyncio

# ---------------------------------------------------------------------------
# Test: Technical Interviewer — Question Generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_technical_questions():
    from app.agents.technical_interviewer import generate_technical_questions

    questions = await generate_technical_questions(
        candidate_name="Alice",
        job_role="Backend Engineer",
        tech_stack=["Python", "FastAPI", "PostgreSQL"],
        resume_text="3 years experience building REST APIs",
        round_type="technical",
        num_questions=3,
    )

    assert len(questions) == 3
    for q in questions:
        assert q.question_id
        assert len(q.text) > 10
        assert q.source_agent == "technical_interviewer"
        assert q.hardened is False

    print("\n✅ Generated questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q.text}")


# ---------------------------------------------------------------------------
# Test: Devil's Advocate — Debate Loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_debate_loop():
    from app.agents.technical_interviewer import generate_technical_questions
    from app.agents.devils_advocate import run_debate_loop

    # Generate one question first
    questions = await generate_technical_questions(
        candidate_name="Bob",
        job_role="ML Engineer",
        tech_stack=["Python", "PyTorch"],
        resume_text="2 years ML experience",
        round_type="technical",
        num_questions=1,
    )
    original_q = questions[0]

    # Run debate
    hardened_q, debate_log = await run_debate_loop(
        question=original_q,
        job_role="ML Engineer",
        tech_stack=["Python", "PyTorch"],
        rounds=1,
    )

    assert hardened_q.hardened is True
    assert len(debate_log) == 1
    assert debate_log[0].devils_challenge
    assert debate_log[0].hardened_question

    print(f"\n✅ Original: {original_q.text}")
    print(f"   Challenge: {debate_log[0].devils_challenge}")
    print(f"   Hardened: {hardened_q.text}")


# ---------------------------------------------------------------------------
# Test: Answer Evaluation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_answer():
    from app.agents.technical_interviewer import evaluate_answer

    result = await evaluate_answer(
        question_id="test-q-1",
        question_text="Explain the difference between a process and a thread.",
        candidate_answer="A process is an independent program in memory. A thread is a lightweight unit within a process that shares memory.",
        job_role="Backend Engineer",
    )

    assert 0 <= result.score <= 10
    assert result.feedback
    print(f"\n✅ Score: {result.score}/10")
    print(f"   Feedback: {result.feedback}")


# ---------------------------------------------------------------------------
# Test: Full Orchestrator Flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_orchestrator_flow():
    from app.services.orchestrator import (
        start_interview_session, process_candidate_answer, get_session_summary
    )
    from app.models.schemas import InterviewRound

    # Start session
    session = await start_interview_session(
        candidate_name="Charlie",
        job_role="Backend Engineer",
        resume_text="5 years Python, built distributed systems",
        round_type=InterviewRound.TECHNICAL,
        tech_stack=["Python", "FastAPI", "Redis"],
    )

    assert session.session_id
    assert len(session.questions) > 0
    print(f"\n✅ Session started: {session.session_id}")
    print(f"   {len(session.questions)} hardened questions generated")
    print(f"   {len(session.debate_log)} debate entries logged")

    # Answer first question
    first_q = session.questions[0]
    answer_resp = await process_candidate_answer(
        session_id=session.session_id,
        question_id=first_q.question_id,
        candidate_answer="I would use connection pooling and horizontal scaling with load balancers.",
    )

    assert answer_resp.evaluation.score >= 0
    print(f"   First answer score: {answer_resp.evaluation.score}/10")


# ---------------------------------------------------------------------------
# Test: HR Agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hr_question_generation():
    from app.agents.hr_agent import generate_hr_questions

    questions = await generate_hr_questions(
        candidate_name="Diana",
        job_role="Senior Engineer",
        resume_text="Led a team of 5 engineers, shipped 3 major products",
        num_questions=2,
    )

    assert len(questions) == 2
    for q in questions:
        assert q.source_agent == "hr_agent"
    print("\n✅ HR Questions:")
    for q in questions:
        print(f"  - {q.text}")
