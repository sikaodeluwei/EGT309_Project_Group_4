#!/bin/bash

echo "Starting EGT309 ML pipeline..."

echo "Step 1: Ingest raw data"
python src/ingest.py

echo "Step 2: Clean data"
python src/clean_data_and_feature_engineer.py

echo "Step 3: Run baseline model selection"
python src/base_model_v2.py

echo "Step 4: Run model tuning"
python src/tuning.py

echo "Step 5: Run model evaluation"
python src/evaluation.py

echo "Pipeline completed."
