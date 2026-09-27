FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt
COPY ml ./ml
COPY services ./services
# продовая компактная модель (+ исследовательский ансамбль как опция ML_MODEL=ensemble)
COPY artifacts/compact ./artifacts/compact
COPY artifacts/models ./artifacts/models
# метрики для карточки модели (/v1/model)
COPY reports/cv_report_compact.json reports/explainability.json ./reports/
ENV ML_MODEL=compact \
    ML_NN_BACKEND=onnx \
    MT_DATA_DIR=/data
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8001/health')"
CMD ["uvicorn", "services.ml_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
