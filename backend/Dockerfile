FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]>=0.29" "httpx>=0.27" "pydantic>=2.6" numpy pandas
COPY backend ./backend
# реплеер NDTP для демо-эндпоинта /api/v1/admin/replay
COPY ndtp ./ndtp
COPY ml/__init__.py ml/data.py ml/geo.py ./ml/
ENV ML_URL=http://ml-service:8001     MT_DATA_DIR=/data     SCHEDULE_SPLIT=test     NDTP_HOST=ingest     EMULATOR_URL=http://emulator:18080     EMULATOR_TARGET_HOST=ingest
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
