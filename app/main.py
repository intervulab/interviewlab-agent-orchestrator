"""
FastAPI Application Entry Point
---------------------------------
This file wires everything together:
  - Creates the FastAPI app
  - Configures CORS (so the Next.js frontend can call us)
  - Mounts all API routers
  - Defines health check endpoint

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
## 🧠 Component 4: LangChain Multi-Agent Orchestrator

This microservice powers all generative AI workflows for InterviewLab.

### Agents
| Agent | Role |
|-------|------|
| **Technical Interviewer** | Generates technical questions, evaluates answers |
| **Devil's Advocate** | Hardens questions through internal debate (never visible to candidate) |
| **HR Agent** | Generates HR questions, produces final hiring recommendation |
| **Canvas Agent** | Analyses system design topology for architectural flaws |

### Key Endpoints
- `POST /interview/start` — Start a new interview session
- `POST /interview/answer` — Submit a candidate answer
- `GET /interview/summary/{session_id}` — Get final results
- `POST /canvas/analyse` — Analyse a system design canvas
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",     # ReDoc UI at http://localhost:8000/redoc
)

# ---------------------------------------------------------------------------
# CORS Middleware
# Allows the Next.js frontend (Component 1) to call this service
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
# ---------------------------------------------------------------------------

app.include_router(interview_router, prefix="/api/v1")
app.include_router(canvas_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    Called by Docker health checks and the Node.js backend to verify this
    microservice is running.
    """
    return {
        "status": "healthy",
        "service": "interviewlab-agent-orchestrator",
        "version": "1.0.0",
        "model": settings.OPENAI_MODEL,
        "environment": settings.APP_ENV,
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "InterviewLab Agent Orchestrator is running 🧠",
        "docs": "/docs",
    }
