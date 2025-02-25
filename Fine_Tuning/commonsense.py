"""
fine_tune_commonsense.py

This script loads the commonsense ethics training dataset, preprocesses the data
by creating a prompt for ethical classification, tokenizes the inputs, and uses
the Hugging Face Trainer API to fine-tune a model (e.g., "utter-project/EuroLLM-1.7B-Instruct")
on the dataset as a sequence classification task.

Ensure your training CSV file ("commonsense_train.csv") is in the same directory,
with columns:
    - label (0 for acceptable, 1 for unacceptable)
    - input (the statement to classify)
"""

import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import numpy as np

PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
TOKENIZERS_PARALLELISM = "false"
# Set model identifier (adjust as necessary)
MODEL_ID = "utter-project/EuroLLM-1.7B-Instruct"
# Define our special prompt wrapper. You can adjust the text per your preference.
def create_prompt(statement: str) -> str:
    """
    Wrap the input statement in a prompt designed to ask for a one-word ethical judgment.
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
    # Add the "prompt" field to each example
    example["prompt"] = create_prompt(example["input"])
    return example

def filter_by_length(example, tokenizer, max_tokens=256):
    # Tokenize without truncation to get the true token length.
    encoding = tokenizer(example["prompt"], truncation=False)
    return len(encoding["input_ids"]) <= max_tokens

def main():
    # Load the dataset (this example uses the 'hendrycks/ethics' dataset for the commonsense split)
    train = load_dataset('hendrycks/ethics', 'commonsense', split='train')
    
    # Preprocess: create the prompt field.
    train = train.map(preprocess_function)
    
    # Initialize your tokenizer, ensuring the pad token is set.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Filter out examples whose tokenized prompt length is greater than 256 tokens.
    # Note: This can be slow if your dataset is large since each prompt is tokenized.
    train = train.filter(lambda example: filter_by_length(example, tokenizer, max_tokens=256))
    
    # Tokenize the dataset with a set max_length (e.g., 256)
    def tokenize_function(example):
        return tokenizer(
            example["prompt"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Now using 256 tokens as max length.
        )
    
    tokenized_dataset = train.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Initialize your model, enable gradient checkpointing, etc.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, num_labels=2)
    #model.gradient_checkpointing_enable()
    #model.config.use_cache = False
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create a data collator for dynamic padding.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Set training arguments.
    training_args = TrainingArguments(
    output_dir="./finetuned_commonsense_model",
    evaluation_strategy="no",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    #gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=True,
    deepspeed="/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ds_config.json",  # Integrate DeepSpeed with your config
    )   
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    trainer.save_model("./finetuned_commonsense_model")
    print("Fine-tuning complete. Model saved as './finetuned_commonsense_model'.")

if __name__ == "__main__":
    main()