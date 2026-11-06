"""
HR Agent
---------
Responsibilities:
  1. Generate HR / behavioural interview questions (cultural fit, teamwork, etc.)
  2. Produce a holistic HR assessment at the end of a session.
  3. Provide final hiring recommendation based on combined technical + HR scores.

This agent uses the same LangChain pattern as the Technical Interviewer
but with HR-focused system prompts.
"""

import uuid
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.core.config import get_settings
from app.models.schemas import Question, EvaluationResult

settings = get_settings()


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.6,
        openai_api_key=settings.OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# 1. HR Question Generation
# ---------------------------------------------------------------------------

HR_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior HR interviewer and culture-fit evaluator.
Generate behavioural interview questions using the STAR framework.

Focus areas:
- Teamwork and collaboration
- Conflict resolution
- Ownership and accountability
- Growth mindset and learning
- Communication under pressure

Output ONLY a numbered list of questions, nothing else.""",
    ),
    (
        "human",
        """Candidate: {candidate_name}
Job Role: {job_role}
Resume Summary: {resume_text}

Generate {num_questions} HR/behavioural interview questions.""",
    ),
])


async def generate_hr_questions(
    candidate_name: str,
    job_role: str,
    resume_text: str,
    num_questions: int = 3,
) -> list[Question]:
    llm = _build_llm()
    chain = HR_QUESTION_PROMPT | llm | StrOutputParser()

    raw: str = await chain.ainvoke({
        "candidate_name": candidate_name,
        "job_role": job_role,
        "resume_text": resume_text or "Not provided",
        "num_questions": num_questions,
    })

    questions: list[Question] = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for sep in [". ", ") ", "."]:
            if line[0].isdigit() and sep in line:
                line = line.split(sep, 1)[1].strip()
                break
        if line:
            questions.append(Question(
                question_id=str(uuid.uuid4()),
                text=line,
                source_agent="hr_agent",
                hardened=False,
            ))

    return questions[:num_questions]


# ---------------------------------------------------------------------------
# 2. Final HR Assessment
# ---------------------------------------------------------------------------

HR_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior HR evaluator writing a final candidate assessment.
Be concise, specific, and fair. Avoid vague statements.

Output in this EXACT format:
ASSESSMENT: <2-3 sentence holistic HR evaluation>
STRENGTHS: <comma-separated list of 2-3 key strengths>
WEAKNESSES: <comma-separated list of 1-2 areas to improve>
RECOMMENDATION: <one of: Strong Hire | Hire | Borderline | No Hire>""",
    ),
    (
        "human",
        """Candidate: {candidate_name}
Job Role: {job_role}
Overall Technical Score: {tech_score}/10
HR Q&A Summary:
{hr_qa_summary}

Write the final HR assessment.""",
    ),
])


async def generate_hr_assessment(
    candidate_name: str,
    job_role: str,
    tech_score: float,
    evaluations: list[EvaluationResult],
) -> dict:
    """
    Summarise the HR Q&A and produce a final hiring recommendation.
    Returns a dict with keys: assessment, strengths, weaknesses, recommendation
    """
    llm = _build_llm()
    chain = HR_ASSESSMENT_PROMPT | llm | StrOutputParser()

    # Build a readable Q&A summary
    hr_qa_parts = []
    for ev in evaluations:
        hr_qa_parts.append(
            f"Q: {ev.question_text}\nA: {ev.candidate_answer}\nScore: {ev.score}/10"
        )
    hr_qa_summary = "\n\n".join(hr_qa_parts) if hr_qa_parts else "No HR answers recorded."

    raw: str = await chain.ainvoke({
        "candidate_name": candidate_name,
        "job_role": job_role,
        "tech_score": round(tech_score, 1),
        "hr_qa_summary": hr_qa_summary,
    })

    result = {
        "assessment": "",
        "strengths": [],
        "weaknesses": [],
        "recommendation": "Borderline",
    }

    for line in raw.strip().split("\n"):
        if line.startswith("ASSESSMENT:"):
            result["assessment"] = line.replace("ASSESSMENT:", "").strip()
        elif line.startswith("STRENGTHS:"):
            raw_str = line.replace("STRENGTHS:", "").strip()
            result["strengths"] = [s.strip() for s in raw_str.split(",")]
        elif line.startswith("WEAKNESSES:"):
            raw_wk = line.replace("WEAKNESSES:", "").strip()
            result["weaknesses"] = [w.strip() for w in raw_wk.split(",")]
        elif line.startswith("RECOMMENDATION:"):
            result["recommendation"] = line.replace("RECOMMENDATION:", "").strip()

    return result
