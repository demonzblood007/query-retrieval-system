FROM python:3.12-slim

ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# System deps for pdfplumber and qdrant local
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    zlib1g \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Qdrant binary
RUN curl -L -o /usr/local/bin/qdrant https://github.com/qdrant/qdrant/releases/latest/download/qdrant && \
    chmod +x /usr/local/bin/qdrant

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create qdrant dirs
RUN mkdir -p /qdrant/storage /qdrant/snapshots /qdrant/config

ENV PORT=8080 \
    START_LOCAL_QDRANT=true \
    ENABLE_QDRANT_CACHE=true \
    PYTHONUNBUFFERED=1

EXPOSE 8080 6333 6334

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["./run.sh"]


