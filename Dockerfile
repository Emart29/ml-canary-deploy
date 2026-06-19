FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first for layer caching. All packages ship wheels — no apt-get needed.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8001 8503
