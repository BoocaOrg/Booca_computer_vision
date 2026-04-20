FROM python:3.11.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics
ENV MPLCONFIGDIR=/tmp/matplotlib

# Ensure /tmp directories exist and have permissions
RUN mkdir -p /tmp/Ultralytics /tmp/matplotlib && chmod -R 777 /tmp

COPY . .

EXPOSE 8000

CMD ["uvicorn", "cv_service.cv_api:app", "--host", "0.0.0.0", "--port", "8000"]
