# Streamlit app only (no torch). Runs anywhere: reads parquet exports or CDW
# Impala, and calls the CAI model endpoint for Customer 360, what-ifs and the
# live categoriser.
#
#   docker build -t spend-app .
#   docker run -p 8501:8501 -v $PWD/data/parquet:/app/data/parquet spend-app
#   docker run -p 8501:8501 -e SPEND_STORAGE_BACKEND=impala -e SPEND_IMPALA_USER=... \
#       -e SPEND_IMPALA_PASSWORD=... -e SPEND_ENDPOINT_URL=... spend-app
FROM python:3.11-slim

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY spend/ spend/
COPY app/ app/
COPY config/ config/

ENV SPEND_STORAGE_BACKEND=parquet \
    SPEND_APP_HOST=0.0.0.0 \
    CDSW_APP_PORT=8501
EXPOSE 8501
CMD ["python", "app/run.py"]
