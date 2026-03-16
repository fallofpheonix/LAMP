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

# Install python packages
COPY requirements.minimal.txt .
RUN pip install --no-cache-dir -r requirements.minimal.txt

# Copy source code
COPY shared_utils ./shared_utils
COPY task1-path-tracing ./task1-path-tracing
COPY task2-viewsheds ./task2-viewsheds
COPY config ./config
COPY main.py .

ENV PYTHONPATH=/app/shared_utils/src:/app/task1-path-tracing/src:/app/task2-viewsheds/src

ENTRYPOINT ["python3", "main.py"]
