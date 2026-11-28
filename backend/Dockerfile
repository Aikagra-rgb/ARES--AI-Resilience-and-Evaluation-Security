FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Third-party dependencies installed strictly from the canonical lockfile.
COPY requirements.lock pyproject.toml ./
RUN pip install -r requirements.lock

# 2) Application code and packaged red-team target/generator modules.
COPY app ./app
COPY ares ./ares
COPY migrations ./migrations
COPY alembic.ini ./
COPY tests/targets ./tests/targets
COPY tests/redteam ./tests/redteam

# 3) Baseline models/stats + deterministic build-time per-client model compilation.
COPY data/models ./data/models
COPY ml_vector_analysis_report.json ./
RUN ENVIRONMENT=test DATABASE_URL=sqlite:///:memory: python -m ares.training.client_model_trainer --org=all

# Fail the build (not the first request) if a required artifact or module is missing.
RUN test -f data/models/classifier.joblib \
    && test -f data/client_models_ml_stats.json \
    && test -f models/org_acme_fintech/intent_classifier/ml_stats.json \
    && test -f ml_vector_analysis_report.json \
    && ENVIRONMENT=test DATABASE_URL=sqlite:///:memory: python -c "import prometheus_client, ares.gateway.enforcer, ares.api.routes.tests, ares.api.routes.clients, ares.api.routes.redteam, ares.api.routes.optimizer, ares.intent.per_client_classifier, ares.vectordb.store, app.dependencies, tests.targets, tests.redteam.attack_generator"

# Run as an unprivileged user after patching base OS security updates.
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home ares && chown -R ares:ares /app
USER ares

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
