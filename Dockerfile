FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt

FROM base AS backend
COPY backend ./backend
COPY docs ./docs
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend
COPY frontend ./frontend
COPY backend ./backend
COPY docs ./docs
COPY artifacts ./artifacts
COPY .streamlit ./.streamlit
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
