FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY prompts/ ./prompts/
COPY src/ ./src/

RUN pip install --no-cache-dir .
RUN python -m spacy download en_core_web_lg

EXPOSE 4005

CMD ["uvicorn", "hey_jude.main:app", "--host", "0.0.0.0", "--port", "4005"]
