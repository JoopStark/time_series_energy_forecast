#!/bin/bash
# time_series_energy_forecast/get_joblibs.sh

# Load variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

LOCAL_JOBLIB_DIR="./joblibs"
REMOTE_PROJECT_DIR="$HPC_WORKSPACE_PATH/runs/time_series_energy_forecast"

echo "Pulling trained models from $HPC_SERVER_ADDRESS..."
echo "Destination: $LOCAL_JOBLIB_DIR"

# Ensure local directory exists
mkdir -p "$LOCAL_JOBLIB_DIR"

# Sync only .joblib files from the remote project directory to local joblibs/
# We use --include and --exclude patterns to be surgical
rsync -avP \
    --include="*.joblib" \
    --exclude="*" \
    "$HPC_SERVER_ADDRESS:$REMOTE_PROJECT_DIR/" \
    "$LOCAL_JOBLIB_DIR/"

if [ $? -eq 0 ]; then
    echo "Successfully updated local models in $LOCAL_JOBLIB_DIR"
else
    echo "Error: Model retrieval failed."
fi
