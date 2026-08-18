# slim-bookworm, not alpine: musl would reject the manylinux aarch64 wheels
# (pyarrow/numpy/pandas/curl-cffi) that make Pi builds compile-free.
FROM python:3.12-slim-bookworm

# PYTHONPATH=/app: the repo uses PEP 420 namespace packages resolved off the
# repo root (`from src.stock_screener...`). WORKDIR /app also puts
# .streamlit/config.toml in streamlit's CWD search path, and cache.py's
# __file__-anchored ROOT resolves to /app, so all state lands under /app/data
# (the bind mount).
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TZ=America/New_York \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first: the fat pip layer caches across code-only deploys.
COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Filtered by the .dockerignore whitelist — only the cockpit closure plus the
# offline test suite (the deploy gate runs inside this image) is copied.
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "src/stock_screener/cockpit/app.py"]
