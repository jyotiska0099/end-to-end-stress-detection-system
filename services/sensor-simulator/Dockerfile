FROM python:3.14-slim

WORKDIR /app

COPY services/sensor-simulator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulator + shared ml package
COPY services/sensor-simulator/simulator.py .
COPY ml ./ml

ENV PYTHONPATH=/app

CMD ["python", "simulator.py"]