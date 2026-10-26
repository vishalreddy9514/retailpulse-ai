# syntax=docker/dockerfile:1
# RetailPulse AI: one image for the API, dashboard, stream producer/consumer and batch pipeline.
# Data, models and reports are mounted at runtime (volumes), never baked into the image.

# Behind a TLS-inspecting proxy, pass its CA as an optional build secret (never stored in the image):
#   docker build --secret id=extra_ca,src=/path/ca.crt --build-arg HTTPS_PROXY=... --network host .

# LightGBM needs the OpenMP runtime (libgomp.so.1). Take it from the full python image of the
# same Debian release instead of apt, so the build needs no OS package mirror.
FROM python:3.12 AS openmp
RUN cp "$(ldconfig -p | awk '/libgomp.so.1 /{print $NF; exit}')" /libgomp.so.1

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY --from=openmp /libgomp.so.1 /usr/local/lib/libgomp.so.1
RUN ldconfig
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; pip install .

# Test stage: lint + unit tests run during the build, so a broken commit cannot produce an image.
FROM base AS test
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; pip install ".[dev]"
COPY tests ./tests
COPY schemas ./schemas
COPY reports ./reports
RUN ruff check src tests && ruff format --check src tests && pytest -q -W error tests/unit

FROM base AS runtime
COPY schemas ./schemas
COPY reports ./reports
# /app/state holds mutable state (approvals DB, stream state); mount a named volume there.
RUN useradd --create-home --uid 10001 retailpulse && mkdir -p data models state && chown -R retailpulse /app
USER retailpulse
ENV RETAILPULSE_LOG_FORMAT=json
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
