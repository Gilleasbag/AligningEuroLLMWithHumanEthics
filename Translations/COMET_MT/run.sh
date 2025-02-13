#!/bin/bash

# Script to run different Python scripts on specific GPUs

# List of Python files
scripts=(
    "COMETdeontology.py"
    "COMETutility.py"
    "COMETvirtue.py"
)

# Define GPUs to use
gpus=(0 1 2)

# Check if there are more scripts than GPUs provided
if [ ${#scripts[@]} -gt ${#gpus[@]} ]; then
    echo "Error: There are more scripts than available GPUs."
    exit 1
fi

# Loop through the scripts and assign each to a specific GPU from the predefined list
for ((i=0; i<${#scripts[@]}; i++))
do
    # Get corresponding GPU index from the 'gpus' array
    gpu_index=${gpus[$i]}

    # Set CUDA_VISIBLE_DEVICES to assign a specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_index python ${scripts[$i]} &
done

# Wait for all background processes to finish
wait
echo "All scripts have been executed."
