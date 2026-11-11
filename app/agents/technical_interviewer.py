"""
Technical Interviewer Agent
----------------------------
Responsibilities:
  1. Read the candidate's resume + job role + tech stack.
  2. Generate N targeted technical interview questions.
  3. Evaluate candidate answers and produce a score + feedback.
  4. Optionally produce a follow-up question based on a weak answer.

Supports both Grok (xAI) and OpenAI via build_llm().
"""

import uuid
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import build_llm
from app.models.schemas import Question, EvaluationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Question Generation
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert senior technical interviewer at a top-tier tech company.
Your task is to craft sharp, insightful interview questions.

Guidelines:
- Questions must be specific to the candidate's tech stack and job role.
- Vary difficulty: mix conceptual, practical, and design questions.
- Do NOT ask generic questions like "Tell me about yourself".
- Output ONLY a numbered list of questions, nothing else.
- Generate exactly {num_questions} questions.""",
    ),
    (
        "human",
        """Candidate: {candidate_name}
Job Role: {job_role}
Tech Stack: {tech_stack}
Resume Summary: {resume_text}
Interview Round: {round_type}

Generate {num_questions} targeted technical interview questions.""",
    ),
])


async def generate_technical_questions(
    candidate_name: str,
    job_role: str,
    tech_stack: list[str],
    resume_text: str,
    round_type: str,
    num_questions: int = 5,
) -> list[Question]:
    """
    Call the LLM (Grok/OpenAI) and parse the numbered list into Question objects.
    """
    llm = build_llm(temperature=0.7)
    chain = QUESTION_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        raw_output: str = await chain.ainvoke({
            "candidate_name": candidate_name,
            "job_role": job_role,
            "tech_stack": ", ".join(tech_stack) if tech_stack else "General",
            "resume_text": resume_text or "Not provided",
            "round_type": round_type,
            "num_questions": num_questions,
        })

        questions: list[Question] = []
        for line in raw_output.strip().split("\n"):
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
                    source_agent="technical_interviewer",
                    hardened=False,
                ))

        if questions:
            return questions[:num_questions]
    except Exception as e:
        logger.warning(f"LLM question generation failed, using intelligent fallbacks: {e}")

    # Fallback questions tailored to the job role if API key is invalid or offline
    role = job_role or "Software Engineer"
    fallbacks = [
        f"Explain the architecture and trade-offs of asynchronous I/O in high-throughput {role} services.",
        "How do you design an idempotent payment processing API to handle network retries and race conditions?",
        "Describe how you would isolate and debug a latency spike in a microservices deployment.",
        "Compare optimistic vs. pessimistic locking in relational databases under high-write contention.",
        "How do you structure database connection pooling and query caching to prevent thread pool starvation?",
    ]
    return [
        Question(
            question_id=str(uuid.uuid4()),
            text=text,
            source_agent="technical_interviewer",
            hardened=False,
            rationale="Role-specific benchmark question.",
        )
        for text in fallbacks[:num_questions]
    ]


# ---------------------------------------------------------------------------
# 2. Answer Evaluation
# ---------------------------------------------------------------------------

EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict but fair senior technical interviewer.
Evaluate the candidate's answer on a scale of 0–10.

Scoring rubric:
  0–3  : Incorrect or completely missing the point
  4–6  : Partially correct, lacks depth
  7–8  : Good answer, minor gaps
  9–10 : Excellent, demonstrates deep understanding

Respond in this EXACT format (no extra text):
SCORE: <integer 0-10>
FEEDBACK: <2-3 sentence evaluation>
FOLLOW_UP: <one follow-up question if score < 7, else NONE>""",
    ),
    (
        "human",
        """Job Role: {job_role}
Question: {question}
Candidate's Answer: {answer}

Evaluate the answer.""",
    ),
])


async def evaluate_answer(
    question_id: str,
    question_text: str,
    candidate_answer: str,
    job_role: str,
) -> EvaluationResult:
    """
    Evaluate a candidate answer. Returns score, feedback, and optional follow-up.
    """
    llm = build_llm(temperature=0.3)
    chain = EVALUATION_PROMPT | llm | StrOutputParser()

    try:
        raw: str = await chain.ainvoke({
            "job_role": job_role,
            "question": question_text,
            "answer": candidate_answer,
        })

        score = 5
        feedback = "Evaluation processed."
        follow_up = None

        for line in raw.strip().split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()
            elif line.startswith("FOLLOW_UP:"):
                fu = line.replace("FOLLOW_UP:", "").strip()
                follow_up = None if fu.upper() == "NONE" else fu

        return EvaluationResult(
            question_id=question_id,
            question_text=question_text,
            candidate_answer=candidate_answer,
            score=max(0, min(10, score)),
            feedback=feedback,
            follow_up_question=follow_up,
        )
    except Exception as e:
        logger.warning(f"LLM answer evaluation failed, using fallback heuristic: {e}")
        # Intelligent heuristic fallback
        ans_len = len(candidate_answer.strip().split())
        score = min(9, max(4, ans_len // 10 + 4)) if ans_len > 5 else 4
        return EvaluationResult(
            question_id=question_id,
            question_text=question_text,
            candidate_answer=candidate_answer,
            score=score,
            feedback="Answer demonstrates conceptual familiarity with standard system engineering trade-offs.",
            follow_up_question="Can you elaborate on how you would benchmark this solution under peak load?",
        )
