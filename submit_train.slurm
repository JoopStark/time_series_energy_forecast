#!/bin/bash
#SBATCH --job-name=energy_forecast
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --partition=compute

# Ensure logs dir exists
mkdir -p logs

echo "Starting training job for ${DATA_FILE:-data/PJME_hourly.parquet}"

# Use the absolute path to your venv's python
./.venv/bin/python3 train_energy_model.py --data "${DATA_FILE:-data/PJME_hourly.parquet}"

echo "Job completed at $(date)"
