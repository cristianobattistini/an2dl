#!/bin/bash
set -euo pipefail

######################################################################
# run_docker.sh — AN2DL Challenge 1
# Autore: Cristiano Battistini
# Esegue l’ambiente di training in container GPU isolato
######################################################################

# --- CONFIGURAZIONE PRINCIPALE ---
IMAGE_NAME="battistini/test1"
TAG="dev"
CONTAINER_NAME="an2dl_test1_$(date +%Y%m%d_%H%M%S)"

GPU_DEVICE=0          # cambia GPU se necessario
CPU_SET="0-3"         # opzionale, limita a CPU specifiche
MEMORY_LIMIT="16g"    # RAM limite
WORKING_DIR="/home/battistini/exp"

# --- PERCORSI LOCALI ---
CODE_FOLDER="/home/$(whoami)/storage/an2dl"
PRIVATE_DATASET_FOLDER="/multiverse/datasets/$(whoami)/"
SHARED_DATASET_FOLDER="/multiverse/datasets/shared/"
LOGS_FOLDER="/home/$(whoami)/storage/an2dl_logs"

mkdir -p "$LOGS_FOLDER"

echo "------------------------------------------------------------"
echo "🏁 Starting AN2DL container"
echo "Image:        ${IMAGE_NAME}:${TAG}"
echo "GPU device:   ${GPU_DEVICE}"
echo "CPU set:      ${CPU_SET}"
echo "Workdir:      ${WORKING_DIR}"
echo "Code folder:  ${CODE_FOLDER}"
echo "------------------------------------------------------------"

docker run -it --rm \
  --gpus "device=${GPU_DEVICE}" \
  --cpuset-cpus "${CPU_SET}" \
  --memory "${MEMORY_LIMIT}" \
  --name "${CONTAINER_NAME}" \
  --mount type=bind,source="${CODE_FOLDER}",target="${WORKING_DIR}" \
  --mount type=bind,source="${PRIVATE_DATASET_FOLDER}",target="${WORKING_DIR}/private_datasets" \
  --mount type=bind,source="${SHARED_DATASET_FOLDER}",target="${WORKING_DIR}/shared_datasets" \
  --mount type=bind,source="${LOGS_FOLDER}",target="${WORKING_DIR}/logs" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -w "${WORKING_DIR}" \
  "${IMAGE_NAME}:${TAG}" \
  /bin/bash
