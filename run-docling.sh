#!/bin/bash
# Script to run docling-inference service with standard Docker Compose

echo "Starting docling-inference with GPU support..."

# Build the Docker image
docker-compose -f docker-compose-regular.yaml build

# Create required volumes if they don't exist
docker volume create hf_cache
docker volume create ocr_cache

# Start the service
docker-compose -f docker-compose-regular.yaml up -d

echo "Docling inference service started!"
echo "API is available at: http://localhost:8878/docs"
echo "To view logs: docker-compose -f docker-compose-regular.yaml logs -f" 