FROM python:3.14-slim

WORKDIR /app

COPY services/llm-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/llm-service/app ./app

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]