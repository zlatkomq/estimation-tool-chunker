#!/bin/bash
# Script to deploy docling-inference service with Docker Swarm

STACK_NAME=docling

echo "Deploying docling-inference to Docker Swarm..."

# Create volumes if they don't exist
docker volume create hf_cache
docker volume create ocr_cache

# Check if stack already exists
if docker stack ls | grep -q "${STACK_NAME}"; then
  echo "Updating existing stack: ${STACK_NAME}"
  docker stack deploy -c docker-compose-experimental.yaml ${STACK_NAME}
else
  echo "Creating new stack: ${STACK_NAME}"
  docker stack deploy -c docker-compose-experimental.yaml ${STACK_NAME}
fi

echo "Deployment initiated!"
echo "API will be available at: http://localhost:8878/docs"
echo "To check service status: docker service ls"
echo "To view logs: docker service logs -f ${STACK_NAME}_docling-inference" 