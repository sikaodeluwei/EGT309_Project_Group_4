#!/bin/bash
set -e

PYTHON_CMD=${PYTHON_CMD:-python}

echo "Starting EGT309 ML pipeline..."

echo "Step 1: Ingest raw data"
$PYTHON_CMD src/ingest.py

echo "Step 2: Clean data"
$PYTHON_CMD src/clean_data_and_feature_engineer.py

echo "Step 3: Run baseline model selection"
$PYTHON_CMD src/base_model_v2.py

echo "Step 4: Run model tuning"
$PYTHON_CMD src/tuning.py

echo "Pipeline completed."
