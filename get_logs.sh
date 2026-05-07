#!/bin/bash
# time_series_energy_forecast/get_logs.sh

# Load variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found!"
    exit 1
fi

LOCAL_LOG_DIR="./hpc_logs"
REMOTE_LOG_DIR="$HPC_WORKSPACE_PATH/runs/time_series_energy_forecast/logs/"

echo "Pulling logs from $HPC_SERVER_ADDRESS..."
echo "Source: $REMOTE_LOG_DIR"
echo "Destination: $LOCAL_LOG_DIR"

# Ensure local directory exists
mkdir -p "$LOCAL_LOG_DIR"

# Sync logs from server to local
rsync -avP "$HPC_SERVER_ADDRESS:$REMOTE_LOG_DIR" "$LOCAL_LOG_DIR/"

if [ $? -eq 0 ]; then
    echo "Successfully updated local logs in $LOCAL_LOG_DIR"
else
    echo "Error: Log retrieval failed."
fi
