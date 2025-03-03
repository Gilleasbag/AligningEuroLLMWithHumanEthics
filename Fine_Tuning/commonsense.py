"""
fine_tune_commonsense_generation.py

This script loads an ethics training dataset, formats each example into an
instruction for text generation, and fine-tunes a causal language model
(e.g., 'utter-project/EuroLLM-1.7B-Instruct') to produce a one-word answer.

For each ethical scenario, the script creates a prompt asking for an answer
('acceptable' or 'unacceptable') and appends the expected answer (derived from the label)
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
LABEL_MAP = {0: " acceptable", 1: " unacceptable"}

def create_prompt(statement: str) -> str:
    """
    Create the prompt without the answer.
    We use the special tags to delimit role instructions.
    """
    prompt = (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Please respond with one word: 'acceptable' or 'unacceptable'.\n"
        "Classify the following statement as acceptable or unacceptable:\n\n"
        f"\"{statement}\"\n"
        "<|im_end|>\n"
        "<|im_start|>assistant"
    )
    return prompt

def preprocess_function(example):
    """
    Append the correct answer text to the prompt to create the full training example.
    Also record the length (in tokens) of the prompt part. Later, we use that
    to mask the prompt portion from the loss.
    """
    # Create the prompt (without answer).
    prompt = create_prompt(example["input"])
    
    # Convert numerical label to text.
    answer_text = LABEL_MAP[example["label"]]
    
    # Full text: prompt + answer + an end-of-response marker (optional).
    # You can adjust the end marker according to your training preferences.
    full_text = prompt + answer_text + "\n<|im_end|>"
    
    # Save both the full text and the prompt text so we can later compute the prompt length.
    example["full_text"] = full_text
    example["prompt_text"] = prompt
    return example

def filter_by_length(example, tokenizer, max_tokens=256):
    """
    Ensure that the full prompt plus answer does not exceed max_tokens.
    """
    encoding = tokenizer(example["full_text"], truncation=False)
    return len(encoding["input_ids"]) <= max_tokens

def tokenize_and_create_labels(example, tokenizer, max_length=256):
    """
    Tokenize the full text and create labels such that the loss is only computed on the answer.
    The tokens for the prompt part of the text are masked out with -100.
    """
    # Tokenize the prompt (to know how many tokens to mask).
    prompt_ids = tokenizer(example["prompt_text"])["input_ids"]
    prompt_length = len(prompt_ids)
    
    # Tokenize the full text.
    tokenized = tokenizer(
        example["full_text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    
    # Copy input_ids to create labels.
    labels = tokenized["input_ids"].copy()
    
    # Mask out the prompt portion by setting its label to -100
    # (so that the loss is only computed on the answer tokens).
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized

def main():
    # Load your dataset.
    # This example uses the 'hendrycks/ethics' dataset with the 'commonsense' split.
    dataset = load_dataset('hendrycks/ethics', 'commonsense', split='train')
    
    # Preprocess the dataset: add prompt and full_text fields.
    dataset = dataset.map(preprocess_function)
    
    # Initialize your tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Optionally filter examples based on the full text length.
    dataset = dataset.filter(lambda ex: filter_by_length(ex, tokenizer, max_tokens=256))
    
    # Tokenize the dataset and create labels.
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_create_labels(ex, tokenizer, max_length=256),
        batched=False,
    )
    
    # Set format for PyTorch.
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    

    # Initialize your causal LM model.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.gradient_checkpointing_enable()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Set training arguments.
    training_args = TrainingArguments(
        output_dir="./finetuned_commonsense_model",
        evaluation_strategy="no",
        num_train_epochs=3,
        per_device_train_batch_size=1,       # Adjust according to your GPU memory
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        fp16=True,
        deepspeed="/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ds_config.json",  # Use if you are using DeepSpeed
    )
    
    # Since this is a text generation task, we don't need to pass compute_metrics.
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
