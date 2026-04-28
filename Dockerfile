# Use the official Python image
FROM python:3.12-slim

# Install system dependencies required for OpenCV (used by ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user (specifically required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Install uv globally before switching user
RUN pip install --no-cache-dir uv

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy project configuration files
COPY --chown=user pyproject.toml uv.lock ./

# Install dependencies using uv
# We point to the CPU-only index for torch to avoid downloading 5GB+ of CUDA binaries
RUN UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
    uv sync --frozen --no-install-project \
    && uv cache clean

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Expose the port used by Hugging Face Spaces
EXPOSE 7860

# Environment variables for uv
ENV UV_LINK_MODE=copy

# Command to run the application using uvicorn
# main:app corresponds to the 'app' object in 'main.py'
CMD ["sh", "-c", "uv sync --frozen --no-cache && uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
