"""
HR Agent
---------
Responsibilities:
  1. Generate HR / behavioural interview questions (cultural fit, teamwork, etc.)
  2. Produce a holistic HR assessment at the end of a session.
  3. Provide final hiring recommendation based on combined technical + HR scores.

Supports Groq LPU API via build_llm().
"""

import uuid
import re
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import build_llm
from app.models.schemas import Question, EvaluationResult

logger = logging.getLogger(__name__)

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
    llm = build_llm(temperature=0.6)
    chain = HR_QUESTION_PROMPT | llm | StrOutputParser()

    try:
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

        if questions:
            return questions[:num_questions]
    except Exception as e:
        logger.warning(f"HR question generation LLM call failed, using fallback: {e}")

    fallbacks = [
        "Describe a critical technical disagreement with a team member and how you resolved it constructively.",
        "Tell me about a production incident you caused or owned, and how you communicated during the outage.",
        "How do you prioritize technical debt against urgent feature delivery requests from product management?",
    ]
    return [
        Question(
            question_id=str(uuid.uuid4()),
            text=t,
            source_agent="hr_agent",
            hardened=False,
        )
        for t in fallbacks[:num_questions]
    ]


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
    llm = build_llm(temperature=0.4)
    chain = HR_ASSESSMENT_PROMPT | llm | StrOutputParser()

    hr_qa_parts = []
    for ev in evaluations:
        hr_qa_parts.append(
            f"Q: {ev.question_text}\nA: {ev.candidate_answer}\nScore: {ev.score}/10"
        )
    hr_qa_summary = "\n\n".join(hr_qa_parts) if hr_qa_parts else "No HR answers recorded."

    result = {
        "assessment": f"Candidate demonstrated solid technical fundamentals for the {job_role} role.",
        "strengths": ["Structured problem solving", "Clear articulation of trade-offs"],
        "weaknesses": ["Deep operational production tuning"],
        "recommendation": "Hire" if tech_score >= 7.0 else "Borderline",
    }

    try:
        raw: str = await chain.ainvoke({
            "candidate_name": candidate_name,
            "job_role": job_role,
            "tech_score": round(tech_score, 1),
            "hr_qa_summary": hr_qa_summary,
        })

        for line in raw.strip().split("\n"):
            clean = line.replace("*", "").strip()
            if re.search(r"^ASSESSMENT:\s*(.+)", clean, re.IGNORECASE):
                m = re.search(r"^ASSESSMENT:\s*(.+)", clean, re.IGNORECASE)
                result["assessment"] = m.group(1).strip()
            elif re.search(r"^STRENGTHS:\s*(.+)", clean, re.IGNORECASE):
                m = re.search(r"^STRENGTHS:\s*(.+)", clean, re.IGNORECASE)
                result["strengths"] = [s.strip() for s in m.group(1).split(",") if s.strip()]
            elif re.search(r"^WEAKNESSES:\s*(.+)", clean, re.IGNORECASE):
                m = re.search(r"^WEAKNESSES:\s*(.+)", clean, re.IGNORECASE)
                result["weaknesses"] = [w.strip() for w in m.group(1).split(",") if w.strip()]
            elif re.search(r"^RECOMMENDATION:\s*(.+)", clean, re.IGNORECASE):
                m = re.search(r"^RECOMMENDATION:\s*(.+)", clean, re.IGNORECASE)
                rec = m.group(1).strip()
                result["recommendation"] = rec
    except Exception as e:
        logger.warning(f"HR assessment LLM call failed, using default evaluation: {e}")

    return result
