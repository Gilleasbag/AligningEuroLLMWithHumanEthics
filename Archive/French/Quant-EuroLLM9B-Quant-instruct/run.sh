#!/bin/bash

# Script to run different Python scripts on separate GPUs

# List of Python files
scripts=(
    "quantCommonsenseBenchmark.py"
    "quantJusticeBenchmark.py"
    "quantVirtueBenchmark.py"
    "quantUtilityBenchmark.py"
    "quantDeontologyBenchmark.py"
)

# Check if the correct number of GPUs exists (assuming GPUs are indexed from 0 to at least 4)
if [ ${#scripts[@]} -gt 5 ]; then
    echo "Error: more scripts than expected GPUs (5)."
    exit 1
fi

# Loop through the scripts and assign each to a different GPU
# Since array indices are zero-based, we subtract 1 from the GPU index
for i in {1..5}
do
    # Adjust index for zero-based indexing in Bash arrays
    script_index=$((i - 1))
    
    # Set CUDA_VISIBLE_DEVICES to assign a specific GPU
    CUDA_VISIBLE_DEVICES=$i python ${scripts[$script_index]} &
done

# Wait for all background processes to finish
wait
echo "All scripts have been executed."
