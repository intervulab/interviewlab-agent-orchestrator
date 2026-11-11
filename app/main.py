"""
FastAPI Application Entry Point
---------------------------------
This file wires everything together:
  - Creates the FastAPI app
  - Configures CORS (allowing frontend and backend proxies)
  - Mounts all API routers (both under /api/v1 and at root for full compatibility)
  - Defines health check endpoints

Run with:
  uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.interview_routes import router as interview_router
from app.api.canvas_routes import router as canvas_router

settings = get_settings()

# ---------------------------------------------------------------------------
# Create FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="InterviewLab Agent Orchestrator",
    description="""
## 🧠 Component 4: Multi-Agent Orchestrator & Debate Engine
Powered by Grok (xAI) and OpenAI via LangChain.

### Agents
| Agent | Role |
|-------|------|
| **Technical Interviewer** | Generates technical questions, evaluates answers |
| **Devil's Advocate** | Hardens questions through internal debate (never visible to candidate) |
| **HR Agent** | Generates HR questions, produces final hiring recommendation |
| **Canvas Agent** | Analyses system design topology for architectural flaws |

### Key Endpoints
- `POST /interview/start` (or `/api/v1/interview/start`) — Start interview session
- `POST /interview/answer` (or `/api/v1/interview/answer`) — Submit candidate answer
- `GET /interview/summary/{session_id}` (or `/api/v1/interview/summary/{id}`) — Final scorecard
- `POST /canvas/analyse` (or `/api/v1/canvas/analyse`) — System design canvas review
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mount Routers
# Mount under BOTH /api/v1 and root so whichever URL is configured in Component 3 works!
# ---------------------------------------------------------------------------

app.include_router(interview_router, prefix="/api/v1")
app.include_router(canvas_router, prefix="/api/v1")

app.include_router(interview_router)
app.include_router(canvas_router)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

def _get_health_payload():
    s = get_settings()
    is_grok = (
        bool(s.GROK_API_KEY.strip() or s.XAI_API_KEY.strip())
        or s.OPENAI_API_KEY.strip().startswith("xai-")
        or s.LLM_PROVIDER.lower() in ("grok", "xai")
    )
    provider = "grok" if is_grok else "openai"
    model = s.GROK_MODEL if is_grok else s.OPENAI_MODEL
    return {
        "status": "healthy",
        "service": "interviewlab-agent-orchestrator",
        "version": "1.0.0",
        "provider": provider,
        "model": model,
        "environment": s.APP_ENV,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return _get_health_payload()


@app.get("/api/v1/health", tags=["Health"])
async def api_health_check():
    return _get_health_payload()


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "InterviewLab Agent Orchestrator is running 🧠",
        "docs": "/docs",
        "health": "/health",
    }
