FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas
COPY ndtp ./ndtp
COPY ml/__init__.py ml/data.py ml/geo.py ./ml/
ENV MT_DATA_DIR=/data
EXPOSE 9201
CMD ["python", "-m", "ndtp.server", "--port", "9201", "--sink", "http://backend:8000/api/v1/telemetry"]
