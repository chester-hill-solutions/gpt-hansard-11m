# Build from the repo root so the context includes the checkpoints:
#   docker build -f model-service/Dockerfile -t gpt-hansard-11m .
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy
COPY model-service/main.py model-service/home.py ./
COPY gpt-11m-sft-bilingual-named.pt .
EXPOSE 8787
CMD ["python", "main.py"]