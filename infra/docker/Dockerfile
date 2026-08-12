# AirPulse — Production FastAPI & UI Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir typing-extensions filelock jinja2 sympy fsspec networkx && \
    pip install --no-cache-dir torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128 --no-deps || \
    pip install --no-cache-dir torch && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code & data
COPY src/ ./src/
COPY data/ ./data/
COPY mlflow.db ./mlflow.db

# Expose FastAPI port
EXPOSE 8000

# Environment variables
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Launch Uvicorn server
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
