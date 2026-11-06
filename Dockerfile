# InterviewLab Agent Orchestrator — Docker Setup

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the server
# --host 0.0.0.0 makes it reachable from other Docker containers (Component 3 Node.js)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
