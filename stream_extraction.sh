#!/bin/bash
set -e

echo "[1/3] Starting Streamed Extraction..."
echo "This pipes the contents of the 5 zip wrappers directly into unzip."
echo "No intermediate files will be created."

mkdir -p large_data_extracted

# The command uses process substitution <() to feed the output of the wrapper unzips
# sequentially as if they were a single concatenated file stream.
# unzip - (read from stdin)

cat <(unzip -p -q large_data/train.zip.001.zip) \
    <(unzip -p -q large_data/train.zip.002.zip) \
    <(unzip -p -q large_data/train.zip.003.zip) \
    <(unzip -p -q large_data/train.zip.004.zip) \
    <(unzip -p -q large_data/train.zip.005.zip) \
    | tar -x -f - -C large_data_extracted

echo "[2/3] Extraction Complete. Deleting wrappers to free space for training artifacts..."
rm large_data/train.zip.00*.zip

echo "[3/3] Starting Training..."
./.venv311/bin/python train_full_model.py
