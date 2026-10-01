# Finpre — API, web app and scheduler share this image (see docker-compose.yml).
# Foundation models (TimesFM/Chronos, ~2 GB) are optional: build with --build-arg WITH_MODELS=1.
FROM python:3.11-slim

ARG WITH_MODELS=0
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TZ=Asia/Taipei
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY fintech_agent ./fintech_agent
RUN pip install -e ".[llm,api]" && if [ "$WITH_MODELS" = "1" ]; then pip install -e ".[models]"; fi
COPY config ./config
COPY scripts ./scripts
COPY docs ./docs
RUN useradd --create-home --uid 1000 finpre && mkdir -p runs data_cache checkpoints reports logs && chown -R finpre /app
USER finpre
EXPOSE 8000 8501
HEALTHCHECK --interval=60s --timeout=5s CMD curl -fs http://localhost:8000/v1/health || exit 1
CMD ["uvicorn", "fintech_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
