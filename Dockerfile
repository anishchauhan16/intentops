FROM python:3.11.17-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    INTENTOPS_ARTIFACT_DIR=/app/artifacts \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY requirements-runtime.txt ./
COPY src ./src
RUN python -m pip install -r requirements-runtime.txt . \
    && useradd --create-home --uid 10001 appuser

# Produce these files using scripts/train.py BEFORE building the image.
COPY artifacts/baseline.joblib artifacts/metadata.json ./artifacts/
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/health', timeout=3)"
CMD ["python", "-m", "intentops.serve"]
