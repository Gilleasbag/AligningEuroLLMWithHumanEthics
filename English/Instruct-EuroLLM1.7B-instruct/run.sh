#!/bin/bash

# Script to run different Python scripts on specific GPUs with multiple model paths,
# executing all tests for one model before moving to the next.

# List of Python files
scripts=(
    "instructCommonsenseBenchmark.py"
    "instructDeontologyBenchmark.py"
    "instructJusticeBenchmark.py"
    "instructVirtueBenchmark.py"
    "instructUtilityBenchmark.py"
)

# List of model paths to test
models=(
    "/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ft_temp_lr1e-05_bs1_ep4"
    "/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ft_temp_lr2e-05_bs1_ep4"
    "/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ft_temp_lr3e-05_bs1_ep4"
)

# Define GPUs to use
gpus=(0 1 2 3 4)

# Loop through each model path
for model in "${models[@]}"
do
    echo "Starting tests for model ${model}..."
    
    # Reset the counter for GPU assignment for this group of scripts
    counter=0
    for script in "${scripts[@]}"
    do
        # Assign GPU in a round-robin fashion based on the available GPUs
        gpu_index=${gpus[$(( counter % ${#gpus[@]} ))]}
        echo "Running ${script} with model ${model} on GPU ${gpu_index}"
        CUDA_VISIBLE_DEVICES=$gpu_index python "$script" --model_id "$model" &
        ((counter++))
    done

    # Wait for all background processes of this model group to finish before moving on
    wait
    echo "Completed tests for model ${model}."
done

echo "All tests have been executed."
