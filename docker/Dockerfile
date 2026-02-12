# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . /app

RUN chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
