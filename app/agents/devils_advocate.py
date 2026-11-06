"""
Devil's Advocate Agent
-----------------------
This agent runs BEHIND THE SCENES — the candidate never sees it.

Responsibility:
  - Take a proposed question from the Technical Interviewer.
  - Challenge it: Is it too easy? Too vague? Can it be gamed?
  - Produce a hardened, tougher version that is harder to bluff.

This creates the "Internal Debate Loop":
  Technical Interviewer proposes → Devil's Advocate critiques → 
  Technical Interviewer hardens → Final question delivered to candidate.

We run this for MAX_DEBATE_ROUNDS rounds to ensure quality.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from app.core.config import get_settings
from app.models.schemas import Question, DebateEntry

settings = get_settings()


def _build_llm() -> ChatOpenAI:
    # Slightly lower temperature for more critical/analytical output
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.4,
        openai_api_key=settings.OPENAI_API_KEY,
    )


# ---------------------------------------------------------------------------
# Step 1: Devil's Advocate critiques the question
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are the Devil's Advocate in a technical interview quality panel.
Your ONLY job is to find weaknesses in proposed interview questions.

Ask yourself:
- Can a candidate bluff this answer with generic buzzwords?
- Is the question too broad or too narrow?
- Does it actually test the skill it claims to test?
- Can a candidate memorize a textbook answer without real understanding?

Output format (EXACT, no extra text):
CRITIQUE: <2-3 sentences describing why this question is weak or could be improved>
SUGGESTION: <one specific way to make it harder to bluff>""",
    ),
    (
        "human",
        """Job Role: {job_role}
Tech Stack: {tech_stack}
Proposed Question: {question}

Critique this question.""",
    ),
])


# ---------------------------------------------------------------------------
# Step 2: Technical Interviewer hardens the question based on critique
# ---------------------------------------------------------------------------

HARDENING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior technical interviewer.
You have received a critique of your question. Now rewrite it to address the weaknesses.

Rules:
- Keep the core topic the same.
- Make it specific enough that buzzword-based bluffing won't work.
- Ask for concrete examples, specific numbers, or real trade-off reasoning.
- Output ONLY the improved question, nothing else.""",
    ),
    (
        "human",
        """Original Question: {original_question}
Devil's Advocate Critique: {critique}
Suggestion: {suggestion}

Rewrite the question to be harder to bluff.""",
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

    llm = _build_llm()
    critique_chain = CRITIQUE_PROMPT | llm | StrOutputParser()
    hardening_chain = HARDENING_PROMPT | llm | StrOutputParser()

    tech_str = ", ".join(tech_stack) if tech_stack else "General"
    debate_log: list[DebateEntry] = []
    current_question_text = question.text

    for round_num in range(1, rounds + 1):
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

        # Fallback if parsing fails
        if not critique:
            critique = critique_raw.strip()
            suggestion = "Make the question more specific and concrete."

        # --- Technical Interviewer hardens ---
        hardened_text: str = await hardening_chain.ainvoke({
            "original_question": current_question_text,
            "critique": critique,
            "suggestion": suggestion,
        })
        hardened_text = hardened_text.strip()

        # Log this debate round
        debate_log.append(DebateEntry(
            round_number=round_num,
            original_question=current_question_text,
            devils_challenge=critique,
            hardened_question=hardened_text,
        ))

        # The hardened question becomes input for the next round
        current_question_text = hardened_text

    # Build the final hardened Question object
    hardened_question = Question(
        question_id=question.question_id,
        text=current_question_text,
        source_agent=question.source_agent,
        hardened=True,
        rationale=f"Hardened through {rounds} debate round(s).",
    )

    return hardened_question, debate_log
