FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --system --uid 65532 --no-create-home embeddings

COPY embeddings ./embeddings
USER 65532:65532

EXPOSE 8030
CMD ["python", "-m", "uvicorn", "embeddings.main:app", "--host", "0.0.0.0", "--port", "8030"]
