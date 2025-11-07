#!/bin/bash
set -euo pipefail

# ==========================================================
# Build script per AN2DL Challenge 1 Docker image
# Autore: Cristiano Battistini
# ==========================================================

# Nome immagine (puoi cambiarlo liberamente)
IMAGE="battistini/test1:dev"

# Directory contenente il Dockerfile
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"

echo "🚀 Building Docker image: $IMAGE"
echo "📁 Using Dockerfile: $DOCKERFILE"

# Build
docker build \
  --build-arg USERNAME=battistini \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -t "$IMAGE" \
  -f "$DOCKERFILE" \
  "$SCRIPT_DIR"

echo "✅ Docker image built successfully: $IMAGE"
