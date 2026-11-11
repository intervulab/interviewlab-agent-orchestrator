"""
Devil's Advocate Agent
-----------------------
This agent runs BEHIND THE SCENES — the candidate never sees it.

Responsibility:
  - Take a proposed question from the Technical Interviewer.
  - Challenge it: Is it too easy? Too vague? Can it be gamed?
  - Produce a hardened, tougher version that is harder to bluff.

Supports Groq LPU API via build_llm().
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import get_settings
from app.core.llm import build_llm
from app.models.schemas import Question, DebateEntry

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Step 1: Devil's Advocate critiques the question
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful review panelist ensuring interview questions are practical and accessible.
Your goal is to make sure questions are clear, practical, and easy to understand for standard candidates.

Guidelines:
- Keep the difficulty EASY TO MODERATE (suitable for students and junior-to-mid developers).
- If the question is too vague or overly abstract, suggest how to anchor it to a common, everyday developer use case.
- Do NOT suggest extreme edge cases, high-concurrency race conditions, or complex multi-tier failures.

Output format (EXACT, no extra text):
CRITIQUE: <1 sentence on how to make the question more practical and clear>
SUGGESTION: <1 sentence suggesting a simple, relatable example or practical angle>""",
    ),
    (
        "human",
        """Job Role: {job_role}
Tech Stack: {tech_stack}
Proposed Question: {question}

Review this question.""",
    ),
])


# ---------------------------------------------------------------------------
# Step 2: Technical Interviewer hardens the question based on critique
# ---------------------------------------------------------------------------

HARDENING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a friendly technical interviewer conducting a mock interview.
Rewrite the question to be clear, approachable, and practical based on the suggestion.

STRICT CONSTRAINTS:
- Keep the difficulty EASY TO MODERATE (accessible and friendly).
- The question MUST be concise: 1 to 2 sentences total.
- It MUST be a single, natural question asking for a clear concept or a practical example.
- NEVER ask overly difficult, intense, or intimidating questions.
- Output ONLY the single rewritten question string with no extra text.""",
    ),
    (
        "human",
        """Original Question: {original_question}
Review: {critique}
Suggestion: {suggestion}

Rewrite this into a single, accessible, friendly technical interview question (1-2 sentences).""",
    ),
])


async def run_debate_loop(
    question: Question,
    job_role: str,
    tech_stack: list[str],
    rounds: int | None = None,
) -> tuple[Question, list[DebateEntry]]:
    """
    Run the behind-the-scenes debate loop for a single question.

    Returns:
        - The hardened Question (same question_id, updated text, hardened=True)
        - A list of DebateEntry records showing the debate history
    """
    if rounds is None:
        rounds = settings.MAX_DEBATE_ROUNDS

    llm = build_llm(temperature=0.4)
    critique_chain = CRITIQUE_PROMPT | llm | StrOutputParser()
    hardening_chain = HARDENING_PROMPT | llm | StrOutputParser()

    tech_str = ", ".join(tech_stack) if tech_stack else "General"
    debate_log: list[DebateEntry] = []
    current_question_text = question.text

    for round_num in range(1, rounds + 1):
        try:
            # --- Devil's Advocate critiques ---
            critique_raw: str = await critique_chain.ainvoke({
                "job_role": job_role,
                "tech_stack": tech_str,
                "question": current_question_text,
            })

            critique = ""
            suggestion = ""
            for line in critique_raw.strip().split("\n"):
                if line.startswith("CRITIQUE:"):
                    critique = line.replace("CRITIQUE:", "").strip()
                elif line.startswith("SUGGESTION:"):
                    suggestion = line.replace("SUGGESTION:", "").strip()

            if not critique:
                critique = critique_raw.strip()
                suggestion = "Require concrete production metrics and failure scenarios."

            # --- Technical Interviewer hardens ---
            hardened_text: str = await hardening_chain.ainvoke({
                "original_question": current_question_text,
                "critique": critique,
                "suggestion": suggestion,
            })
            hardened_text = hardened_text.strip()
        except Exception as e:
            logger.warning(f"Debate loop LLM call failed, using heuristic hardening: {e}")
            critique = "Question could be easily bluffed with generic textbook definitions."
            suggestion = "Add concrete throughput metrics and edge-case failure considerations."
            hardened_text = f"{current_question_text} Specifically, discuss failure modes under high load and the trade-offs you must make."

        debate_log.append(DebateEntry(
            round_number=round_num,
            original_question=current_question_text,
            devils_challenge=critique,
            hardened_question=hardened_text,
        ))

        current_question_text = hardened_text

    hardened_question = Question(
        question_id=question.question_id,
        text=current_question_text,
        source_agent=question.source_agent,
        hardened=True,
        rationale=f"Hardened through {rounds} debate round(s).",
    )

    return hardened_question, debate_log
