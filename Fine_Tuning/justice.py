"""
fine_tune_commonsense_generation.py

This script loads a justice training dataset from a CSV file ("justice.csv"),
formats each example into an instruction for text generation, and fine-tunes a
causal language model (e.g., 'utter-project/EuroLLM-1.7B-Instruct') to produce a one-word answer.

For each ethical scenario, the script creates a prompt asking for an answer
('reasonable' or 'unreasonable') and appends the expected answer (derived from the label)
at the end of the prompt. During training, the loss is computed only on the answer portion.
"""

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import numpy as np
import torch

# Set environment variables if necessary
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

# Set model identifier (adjust as necessary)
MODEL_ID = "utter-project/EuroLLM-1.7B-Instruct"

# Mapping integer labels to answer text.
# In this dataset, label 1 means "reasonable" and label 0 means "unreasonable".
LABEL_MAP = {0: " unreasonable", 1: " reasonable"}

def create_prompt(scenario: str) -> str:
    """
    Create the prompt using the new format.
    """
    prompt = (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: Would most people believe this reasonable or unreasonable to say? Respond using only one word, reasonable or unreasonable. \"{scenario}\"\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt

def preprocess_function(example):
    """
    Append the correct answer text to the prompt to create the full training example.
    Also record the length (in tokens) of the prompt part so that we later mask the prompt portion from the loss.
    """
    # Use the "scenario" field from the CSV.
    prompt = create_prompt(example["scenario"])
    
    # Convert numerical label to text using the updated mapping.
    answer_text = LABEL_MAP[example["label"]]
    
    # Full text: prompt + answer + an end-of-response marker.
    full_text = prompt + answer_text + "\n<|im_end|>"
    
    # Store both the full text and the prompt text.
    example["full_text"] = full_text
    example["prompt_text"] = prompt
    return example

def filter_by_length(example, tokenizer, max_tokens=128):
    """
    Ensure that the full prompt plus answer does not exceed max_tokens.
    """
    encoding = tokenizer(example["full_text"], truncation=False)
    return len(encoding["input_ids"]) <= max_tokens

def tokenize_and_create_labels(example, tokenizer, max_length=128):
    """
    Tokenize the full text and create labels such that the loss is computed only on the answer.
    The tokens corresponding to the prompt portion are masked out (-100).
    """
    # Tokenize the prompt to get its token length.
    prompt_ids = tokenizer(example["prompt_text"])["input_ids"]
    prompt_length = len(prompt_ids)
    
    # Tokenize the full text with padding and truncation.
    tokenized = tokenizer(
        example["full_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    
    # Create a copy of input_ids for labels.
    labels = tokenized["input_ids"].copy()
    
    # Mask out the prompt portion so that loss is computed only on the answer.
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized

def main():
    # Load the custom CSV dataset. Ensure "justice.csv" is available in the same directory.
    # The CSV should have columns "label" and "scenario".
    dataset = load_dataset('hendrycks/ethics', 'justice', split='train')  # train set  
    
    # Preprocess the dataset by adding prompt and full_text fields.
    dataset = dataset.map(preprocess_function)
    
    # Initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Filter out examples with a full text length exceeding the limit.
    dataset = dataset.filter(lambda ex: filter_by_length(ex, tokenizer, max_tokens=128))
    
    # Tokenize the dataset and create labels.
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_create_labels(ex, tokenizer, max_length=128),
        batched=False,
    )
    
    # Set format for PyTorch.
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Initialize the causal LM model.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.gradient_checkpointing_enable()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Set training arguments.
    training_args = TrainingArguments(
        output_dir="./finetuned_commonsense_model",
        evaluation_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=1,       # Adjust based on available GPU memory.
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=True,
        deepspeed="/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ds_config.json",  # Path to your DeepSpeed config.
    )
    
    # Initialize Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    torch.cuda.empty_cache()
    # Start training.
    trainer.train()
    
    # Save the model.
    trainer.save_model("./finetuned_commonsense_model")
    print("Fine-tuning complete. Model saved as './finetuned_commonsense_model'.")

if __name__ == "__main__":
    main()
