# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project configuration files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# We point to the CPU-only index for torch to avoid downloading 5GB+ of CUDA binaries
RUN UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY . .

# Expose the port used by Cloud Run (default is 8080)
EXPOSE 8080

# Environment variables for uv
ENV UV_LINK_MODE=copy

# Command to run the application using uvicorn
# agent:app corresponds to the 'app' object in 'agent.py'
CMD ["sh", "-c", "uv sync --frozen && uv run uvicorn agent:app --host 0.0.0.0 --port $PORT"]
