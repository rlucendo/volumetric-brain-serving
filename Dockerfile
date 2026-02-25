# ==============================================================================
# Stage 1: Builder (Compiles dependencies and keeps the final image clean)
# ==============================================================================
FROM python:3.10-slim AS builder

# Set environment variables for non-interactive installation and stable Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to easily copy it to the runner stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the dependency manager configuration
COPY pyproject.toml .

# Install dependencies strictly targeting CPU to reduce image size and cloud costs.
# We upgrade pip, install PyTorch CPU explicitly, and then install our project.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir .

# ==============================================================================
# Stage 2: Runner (Production ready, minimal attack surface)
# ==============================================================================
FROM python:3.10-slim AS runner

# Set environment variables for runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH="/app/models/last.ckpt"

# Create a non-root user for security compliance (Standard in Cloud deployments)
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy the pre-compiled virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the necessary source code and scripts (ignoring tests and local configs)
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create the models directory and grant ownership to the non-root user.
# This ensures the download script can save the .ckpt file here during CI/CD.
# RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Copy the locally downloaded model into the container and set permissions
COPY --chown=appuser:appuser models/ ./models/

# Switch to the restricted user profile
USER appuser

# Expose the port Uvicorn will listen on
EXPOSE 8000

# Command to boot the FastAPI application via Uvicorn with a single worker
# (Volumetric processing is highly CPU/RAM bound, 1 worker prevents Out-of-Memory crashes)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]