# Deepfake Detection Streamlit Application
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, git (for CLIP), curl for health checks, and Chromium for PDF charts
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libgthread-2.0-0 \
    chromium \
    chromium-driver \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome path for kaleido
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMIUM_PATH=/usr/bin/chromium

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -s /bin/bash --uid 1001 appuser

# Copy application files with correct ownership
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser config.py .
COPY --chown=appuser:appuser shared_functions.py .
COPY --chown=appuser:appuser classifier.py .
COPY --chown=appuser:appuser detector.py .
COPY --chown=appuser:appuser spai_detector.py .
COPY --chown=appuser:appuser cloud_providers.py .
COPY --chown=appuser:appuser generate_report_updated.py .
COPY --chown=appuser:appuser models.json.example .
COPY --chown=appuser:appuser prompt_version.py .

# Copy prompts directory (file-per-version system)
COPY --chown=appuser:appuser prompts/ ./prompts/

# Copy SPAI module and required subdirectories
COPY --chown=appuser:appuser spai/ ./spai/

# NOTE: SPAI weights must be downloaded separately due to size
# Download spai.pth from: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view
# Place in: spai/weights/spai.pth before building the Docker image
# The file should exist at: spai/weights/spai.pth

# Create necessary directories with correct ownership
RUN mkdir -p results testing_files misc analysis_output && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
