#!/usr/bin/env bash
# download_dataset.sh — Download and extract the LIBERO-Spatial dataset
set -euo pipefail

DATASET_URL="https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
SCRATCH="${SCRATCH}"
DATA_DIR="${SCRATCH}/lerobot-demo-data"
ZIP_FILE="${SCRATCH}/libero_spatial.zip"

echo "=== Downloading LIBERO-Spatial dataset ==="
echo "URL: ${DATASET_URL}"
echo "Destination: ${ZIP_FILE}"

# Download (follow redirects, show progress)
if command -v wget &>/dev/null; then
    wget -O "${ZIP_FILE}" "${DATASET_URL}" --max-redirect=5 --show-progress
elif command -v curl &>/dev/null; then
    curl -L -o "${ZIP_FILE}" "${DATASET_URL}" --progress-bar
else
    echo "ERROR: Neither wget nor curl found."
    exit 1
fi

echo ""
echo "=== Extracting dataset ==="
mkdir -p "${DATA_DIR}"
unzip -o "${ZIP_FILE}" -d "${DATA_DIR}"

echo ""
echo "=== Dataset structure (top 3 levels) ==="
find "${DATA_DIR}" -maxdepth 3 -type f -o -type d | head -50

echo ""
echo "=== Done ==="
echo "Dataset extracted to: ${DATA_DIR}"
echo "Zip file kept at: ${ZIP_FILE}"
