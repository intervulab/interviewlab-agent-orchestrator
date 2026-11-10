# 🧠 InterviewLab — LangChain Multi-Agent Orchestrator

> **Part of the [InterviewLab](https://github.com/orgs/intervulab/repositories) group project.**
> This microservice is owned by **Component 4** and powers all Generative AI workflows.

---

## What This Does

This is a **FastAPI microservice** that runs three AI agents using **LangChain**:

| Agent | What It Does | Visible to Candidate? |
|---|---|---|
| 🎙️ **Technical Interviewer** | Generates interview questions, evaluates answers | ✅ Yes |
| 😈 **Devil's Advocate** | Secretly challenges and hardens every question | ❌ No |
| 🤝 **HR Agent** | Generates HR questions, writes final hiring recommendation | ✅ Yes |
| 🗺️ **Canvas Agent** | Reviews system design diagrams for architectural flaws | ✅ Yes |

The killer feature is the **Internal Debate Loop**:

```
Technical Interviewer generates question
        ↓
Devil's Advocate: "This question is too easy to bluff!"
        ↓
Technical Interviewer hardens it (runs N times)
        ↓
Final hardened question delivered to candidate
```

---

## Project Structure

```
interviewlab-agent-orchestrator/
│
├── app/
│   ├── main.py                    ← FastAPI app, CORS, router mounting
│   ├── core/
│   │   └── config.py              ← All settings loaded from .env
│   ├── models/
│   │   └── schemas.py             ← Pydantic request/response shapes
│   ├── agents/
│   │   ├── technical_interviewer.py  ← Generates + evaluates questions
│   │   ├── devils_advocate.py        ← Hardens questions (behind the scenes)
│   │   ├── hr_agent.py              ← HR questions + hiring recommendation
│   │   └── canvas_agent.py          ← Analyses system design diagrams
│   ├── services/
│   │   └── orchestrator.py        ← Coordinates all agents, manages sessions
│   └── api/
│       ├── interview_routes.py    ← /interview/start, /answer, /summary
│       └── canvas_routes.py       ← /canvas/analyse
│
├── tests/
│   └── test_agents.py             ← Integration tests
│
├── .env.example                   ← Copy this to .env
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How to Run Locally

### 1. Clone & set up environment

```bash
git clone https://github.com/intervulab/interviewlab-agent-orchestrator.git
cd interviewlab-agent-orchestrator

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Now open .env and add your OPENAI_API_KEY
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now running at **http://localhost:8000**

- **Swagger UI (interactive docs):** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

---

## API Endpoints

### `POST /api/v1/interview/start`
Start a new interview session. Generates and hardens all questions.

**Request:**
```json
{
  "candidate_name": "Alice",
  "job_role": "Backend Engineer",
  "resume_text": "3 years Python, built REST APIs at scale",
  "round_type": "technical",
  "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Redis"]
}
```

**Response includes:**
- `session_id` — save this for all future calls
- `questions` — the hardened questions the candidate will see
- `debate_log` — the internal Devil's Advocate debate (for transparency/logging)

---

### `POST /api/v1/interview/answer`
Submit a candidate's answer to a question.

**Request:**
```json
{
  "session_id": "uuid-from-start",
  "question_id": "uuid-of-question",
  "candidate_answer": "I would use connection pooling and horizontal scaling...",
  "audio_transcript": "(optional — injected by Component 5)"
}
```

**Response includes:**
- `evaluation` — score (0–10) + feedback + optional follow-up
- `next_question` — the next question to show
- `session_complete` — `true` when all questions are answered

---

### `GET /api/v1/interview/summary/{session_id}`
Get the final hiring recommendation. Call after `session_complete: true`.

**Response includes:**
- `overall_score` — average across all questions
- `hr_assessment` — narrative HR evaluation
- `strengths` / `weaknesses`
- `hiring_recommendation` — `Strong Hire | Hire | Borderline | No Hire`

---

### `POST /api/v1/canvas/analyse`
Analyse a system design canvas (called by Component 2).

**Request:**
```json
{
  "session_id": "uuid",
  "canvas_json": {
    "nodes": [
      {"id": "lb-1", "type": "LoadBalancer"},
      {"id": "db-1", "type": "Database"}
    ],
    "edges": [
      {"from": "lb-1", "to": "db-1"}
    ]
  }
}
```

**Response includes:**
- `flaws` — list with `component_id` (maps to canvas node), `flaw_description`, `severity`
- `suggestions`
- `overall_assessment`

> The `component_id` in flaws maps back to canvas node IDs so Component 2 can flash the correct nodes! 🎯

---

## How to Run Tests

```bash
# Make sure your .env has a valid OPENAI_API_KEY
pytest tests/ -v
```

---

## Integration with Other Components

| Component | How it connects |
|---|---|
| **Component 1** (Next.js Frontend) | Calls `/interview/start`, `/interview/answer` via the Node.js proxy |
| **Component 2** (Canvas) | Calls `/canvas/analyse` with the JSON graph |
| **Component 3** (Node.js Backend) | Acts as the proxy — routes requests from frontend to this service |
| **Component 5** (Audio/ML) | Injects `audio_transcript` into the `/interview/answer` payload |

---

## Architecture Diagram

```
Next.js Frontend (Component 1)
         │
         ▼
Node.js Backend / Proxy (Component 3)
         │
         ├──► POST /api/v1/interview/start  ──► Orchestrator
         │                                          │
         │                                    ┌─────┴──────┐
         │                             Tech Interviewer   Devil's Advocate
         │                             (generates Qs)    (hardens Qs, hidden)
         │                                    └─────┬──────┘
         │                                          │
         ├──► POST /api/v1/interview/answer ──► Orchestrator
         │                                          │
         │                                   Tech Interviewer
         │                                   (evaluates answer)
         │
         ├──► GET  /api/v1/interview/summary ──► HR Agent
         │                                      (final assessment)
         │
Canvas (Component 2) ──► POST /api/v1/canvas/analyse ──► Canvas Agent

Audio Pipeline (Component 5) ──► injects audio_transcript into /answer
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key | **Required** |
| `OPENAI_MODEL` | LLM model name | `gpt-4o-mini` |
| `MAX_DEBATE_ROUNDS` | Devil's Advocate debate iterations | `2` |
| `MAX_QUESTIONS_PER_ROUND` | Questions per session | `5` |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated) | `http://localhost:3000` |

---

## Run with Docker

```bash
docker build -t interviewlab-agent-orchestrator .
docker run -p 8000:8000 --env-file .env interviewlab-agent-orchestrator
```
