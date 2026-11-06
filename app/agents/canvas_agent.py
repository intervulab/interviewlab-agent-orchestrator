"""
Canvas Analysis Agent
----------------------
Receives a JSON graph (nodes + edges) from the System Design Canvas (Component 2)
and uses an LLM to reason about architectural flaws.

This agent produces:
  - A list of specific flaws with component IDs (so the canvas can flash them)
  - Improvement suggestions
  - An overall architecture assessment

This is a READ-ONLY agent — it never modifies the canvas state.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import get_settings

settings = get_settings()


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        temperature=0.3,   # Low temperature for deterministic architectural reasoning
        openai_api_key=settings.OPENAI_API_KEY,
    )


CANVAS_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a principal systems architect reviewing a candidate's system design.
You will receive a JSON graph representing their architecture (nodes = components, edges = connections).

Your job:
1. Identify SPECIFIC architectural flaws (single points of failure, missing load balancers,
   no caching layer, missing CDN, unscalable bottlenecks, insecure direct DB access, etc.)
2. For each flaw, reference the specific component ID from the graph.
3. Suggest concrete improvements.
4. Give an overall assessment.

Respond in VALID JSON only, following this exact structure:
{{
  "flaws": [
    {{
      "component_id": "<id from the node graph, or 'overall'>",
      "flaw_description": "<specific problem>",
      "severity": "<high|medium|low>"
    }}
  ],
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "overall_assessment": "<2-3 sentence summary of the architecture quality>"
}}""",
    ),
    (
        "human",
        """Job Role: {job_role}
System Design Canvas (JSON Graph):
{canvas_json}

Analyse this architecture and return JSON.""",
    ),
])


async def analyse_canvas(
    canvas_json: dict,
    job_role: str = "Software Engineer",
) -> dict:
    """
    Analyse a system design canvas graph and return structured flaw data.
    The component_id fields in flaws can be used by the frontend to flash specific nodes.
    """
    import json

    llm = _build_llm()
    chain = CANVAS_ANALYSIS_PROMPT | llm | StrOutputParser()

    raw: str = await chain.ainvoke({
        "job_role": job_role,
        "canvas_json": json.dumps(canvas_json, indent=2),
    })

    # Try to parse JSON; fall back to a safe default
    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        result = json.loads(clean.strip())
    except Exception:
        result = {
            "flaws": [{
                "component_id": "overall",
                "flaw_description": "Could not parse architectural analysis. Please check canvas data.",
                "severity": "medium",
            }],
            "suggestions": ["Ensure your canvas has at least nodes and edges defined."],
            "overall_assessment": raw[:300] if raw else "Analysis failed.",
        }

    return result
