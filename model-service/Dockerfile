FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy
COPY main.py home.py ./
EXPOSE 8787
CMD ["python", "main.py"]