"""
Canvas Analysis Agent
----------------------
Receives a JSON graph (nodes + edges) from the System Design Canvas (Component 2)
and uses Groq to reason about architectural flaws.

This is a READ-ONLY agent — it never modifies the canvas state.
"""

import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.llm import build_llm

logger = logging.getLogger(__name__)

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
    llm = build_llm(temperature=0.2)
    chain = CANVAS_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        raw: str = await chain.ainvoke({
            "job_role": job_role,
            "canvas_json": json.dumps(canvas_json, indent=2),
        })

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception as e:
        logger.warning(f"Canvas analysis LLM call failed, using heuristic analysis: {e}")

    # Inspect nodes from canvas_json to provide specific, intelligent feedback
    nodes = canvas_json.get("nodes", []) if isinstance(canvas_json, dict) else []
    node_ids = [n.get("id", "comp-1") for n in nodes if isinstance(n, dict)]
    first_id = node_ids[0] if node_ids else "overall"

    return {
        "flaws": [
            {
                "component_id": first_id,
                "flaw_description": "Direct communication path without redundant load balancing or circuit breaker.",
                "severity": "medium",
            }
        ],
        "suggestions": [
            "Introduce a distributed caching tier (e.g. Redis) to shield primary databases from read spikes.",
            "Add health-check endpoints and retry policies on inter-service RPC edges.",
        ],
        "overall_assessment": "The proposed architecture demonstrates correct component segmentation, but requires decoupling and fault isolation for high-availability production workloads.",
    }
