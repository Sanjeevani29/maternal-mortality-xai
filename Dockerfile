# Container for Hugging Face Spaces (Docker SDK) or any container host.
# The image serves pre-computed artefacts; run the pipeline locally first so that
# data/processed/, models/ and reports/figures/ are populated before building.
FROM python:3.11-slim

# OpenMP runtime required by XGBoost / LightGBM wheels on Linux.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run containers as uid 1000.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser config ./config
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser data/processed ./data/processed
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser reports ./reports

USER appuser
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
