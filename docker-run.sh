#!/bin/bash
# Script to run docling-inference directly with Docker run

# Build the image if needed
docker build -t docling-inference:latest .

# Create volumes if they don't exist
docker volume create hf_cache
docker volume create ocr_cache

# Remove existing container if it exists
docker rm -f docling-inference 2>/dev/null || true

# Run with GPU access
docker run -d \
  --name docling-inference \
  --gpus all \
  -p 8877:8080 \
  -e DEV_MODE=0 \
  -e AUTH_TOKEN=dev-key \
  -e NUM_WORKERS=12 \
  -v $(pwd)/logs:/app/logs \
  -v hf_cache:/root/.cache/huggingface \
  -v ocr_cache:/root/.EasyOCR \
  docling-inference:latest

echo "Container started. API available at http://localhost:8877/docs"
echo "To check logs: docker logs -f docling-inference" 