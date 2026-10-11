FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) Third-party dependencies. They are derived from pyproject.toml (single source of truth) so
#    this expensive layer is cached until the dependency list changes, and no application source
#    needs to exist yet.
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/runtime-requirements.txt \
    && pip install -r /tmp/runtime-requirements.txt \
    && rm /tmp/runtime-requirements.txt

# 2) Application code. `app.main` imports the `ares` package, so BOTH must ship in the image.
COPY app ./app
COPY ares ./ares
COPY migrations ./migrations
COPY alembic.ini ./

# 3) Runtime artifacts that the code loads relative to the backend root (/app):
#    - data/models/classifier.joblib  -> gateway Tier-1 ML classifier (ares.evaluation.ml_stats)
#    - ml_vector_analysis_report.json -> /api/stats model report
COPY data/models ./data/models
COPY ml_vector_analysis_report.json ./

# Fail the build (not the first request) if a required artifact or module is missing.
RUN test -f data/models/classifier.joblib \
    && test -f ml_vector_analysis_report.json \
    && python -c "import ares.gateway.enforcer, ares.api.routes.tests, ares.vectordb.store, app.dependencies"

# Run as an unprivileged user.
RUN useradd --system --uid 10001 --no-create-home ares && chown -R ares:ares /app
USER ares

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
