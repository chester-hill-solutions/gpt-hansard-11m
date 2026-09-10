FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy
COPY main.py home.py ./
COPY gpt-11m-sft-en-v3.pt .
EXPOSE 8787
CMD ["python", "main.py"]
