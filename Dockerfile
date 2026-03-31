FROM python:3.11-slim-bookworm

# Install GIS dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    g++ \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

# Copy package metadata and source before installing
COPY pyproject.toml .
COPY src/ ./src/

# Install python package and dependencies
RUN pip install --no-cache-dir .

# Copy remaining assets
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python3", "-m", "lamp.api.cli"]
