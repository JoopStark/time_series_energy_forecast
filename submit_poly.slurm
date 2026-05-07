#!/bin/bash
#SBATCH --job-name=energy_poly
#SBATCH --output=logs/poly_%j.log
#SBATCH --error=logs/poly_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --partition=compute

# Ensure logs dir exists
mkdir -p logs

echo "Starting Polynomial Regression job for ${DATA_FILE:-data/PJME_hourly.parquet}"

# Use the absolute path to your venv's python
./.venv/bin/python3 train_poly_model.py --data "${DATA_FILE:-data/PJME_hourly.parquet}"

echo "Job completed at $(date)"
