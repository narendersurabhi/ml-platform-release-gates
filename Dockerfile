FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "mlprg.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
