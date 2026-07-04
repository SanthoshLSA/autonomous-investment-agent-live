# Dockerfile for Production Deployment
FROM python:3.11-slim-bookworm

# System configuration variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy config and pyproject definition files
COPY pyproject.toml uv.lock ./

# Install python dependencies using uv (cache mounts speed up builds)
RUN uv sync --frozen --no-dev

# Copy application source directories
COPY src/ ./src/
COPY streamlit_app.py config.yaml ./

# Create log/cache/reports directories for local files
RUN mkdir -p logs cache reports/output

EXPOSE 8501

# Entry point starts the Streamlit dashboard on $PORT
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
