import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset, DownloadMode
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch
from peft import get_peft_model, LoraConfig, TaskType
import random
import csv

# -------------------------------
# Environment and Model Setup
# -------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
MODEL_ID = "utter-project/EuroLLM-1.7B-Instruct"

# -------------------------------
# Helper: Finalize Each Example
# -------------------------------
def finalize_example(example):
    """
    Combine the prompt and answer into a full_text.
    The full_text is constructed as:
      full_text = prompt_text + answer_text + "\n<|im_end|>"
    """
    example["full_text"] = example["prompt_text"] + example["answer_text"] + "\n<|im_end|>"
    return example

# -------------------------------
# Transformation Functions for Each Dataset
# -------------------------------

# 1. Commonsense
def transform_commonsense(example):
    input_text = example["input"].strip()
    prompt = (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Please respond with one word: 'acceptable' or 'unacceptable'.\n"
        "Classify the following statement as acceptable or unacceptable:\n\n"
        f"\"{input_text}\"\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer = "acceptable" if int(example["label"]) == 0 else "unacceptable"
    example["prompt_text"] = prompt
    example["answer_text"] = answer
    return example

# 2. Deontology
def transform_deontology(example):
    scenario = example["scenario"].strip()
    excuse = example["excuse"].strip()
    prompt = (
        "<|im_start|>user\n"
        f"Scenario: \"{scenario}\"\n"
        f"Response: \"{excuse}\"\n"
        "Question: Is this response a reasonable exemption? Answer with only a 'Yes' or a 'No':"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer = "Yes" if int(example["label"]) == 1 else "No"
    example["prompt_text"] = prompt
    example["answer_text"] = answer
    return example

# 3. Justice
def transform_justice(example):
    scenario = example["scenario"].strip()
    prompt = (
        "<|im_start|>system\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Question: Would most people believe this reasonable or unreasonable to say? Respond using only one word, reasonable or unreasonable. \"{scenario}\"\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    answer = "reasonable" if int(example["label"]) == 1 else "unreasonable"
    example["prompt_text"] = prompt
    example["answer_text"] = answer
    return example

# 4. Utility (Split into two separate examples using batched mapping)
def transform_utility_batched(examples):
    new_prompt_texts = []
    new_answer_texts = []
    for baseline, less_pleasant in zip(examples["baseline"], examples["less_pleasant"]):
        baseline = baseline.strip()
        less_pleasant = less_pleasant.strip()
        prompt_base = (
            "<|im_start|>user\n"
            f"Scenario: \"{baseline}\"\n"
            "Question: On a scale from 1 (extremely unpleasant) to 10 (extremely pleasant), "
            "rate the pleasantness of this scenario. Respond only with a number.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        prompt_less = (
            "<|im_start|>user\n"
            f"Scenario: \"{less_pleasant}\"\n"
            "Question: On a scale from 1 (extremely unpleasant) to 10 (extremely pleasant), "
            "rate the pleasantness of this scenario. Respond only with a number.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        new_prompt_texts.append(prompt_base)
        new_answer_texts.append(str(random.choice([7, 8, 9, 10])))
        new_prompt_texts.append(prompt_less)
        new_answer_texts.append(str(random.choice([1, 2, 3, 4])))
    return {"prompt_text": new_prompt_texts, "answer_text": new_answer_texts}

# Instead of using flat_map (which may not be available), we unbatch manually.
def manual_unbatch(ds):
    unbatched = []
    for ex in ds:
        if isinstance(ex["prompt_text"], list):
            # Unroll each batched pair into individual examples.
            for i in range(len(ex["prompt_text"])):
                unbatched.append({
                    "prompt_text": ex["prompt_text"][i],
                    "answer_text": ex["answer_text"][i]
                })
        else:
            # The example is already singular.
            unbatched.append(ex)
    return Dataset.from_list(unbatched)

# 5. Virtue (Grouped by scenario with dual output for correct and not applicable traits)
def load_and_transform_virtue(csv_path):
    df = load_dataset(csv_path, 'virtue', split='train').to_pandas()
    df[['scenario_text', 'trait']] = df['scenario'].str.split(r'\s*\[SEP\]\s*', expand=True)
    records = []
    for scenario, group in df.groupby('scenario_text'):
        # Get unique traits, stripping whitespace and punctuation
        traits = list(group['trait'].str.strip().str.rstrip('.').unique())
        random.shuffle(traits)
        trait_list = ", ".join(traits)
        # Identify all correct traits (label == 1)
        correct_traits = group[group['label'] == 1]['trait'].str.strip().tolist()
        if not correct_traits:
            correct_str = "unknown"
        else:
            correct_str = ", ".join(correct_traits)
        # All other traits are considered not applicable
        not_applicable_traits = [t for t in traits if t not in correct_traits]
        not_applicable_str = ", ".join(not_applicable_traits) if not_applicable_traits else "none"
        
        prompt = (
            "<|im_start|>user\n"
            f"Scenario: \"{scenario.strip()}\"\n"
            f"Which of the following traits applies to the scenario: {trait_list}?\n"
            "Indicate also which traits do NOT apply.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        answer = (
            f"Correct trait(s): {correct_str}\n"
            f"Not applicable: {not_applicable_str}\n"
            "<|im_end|>"
        )
        records.append({"prompt_text": prompt, "answer_text": answer})
    return Dataset.from_pandas(pd.DataFrame(records))

# -------------------------------
# Tokenization Helpers
# -------------------------------
def choose_max_tokens(example):
    """
    Set maximum tokens for every example to 256.
    """
    return 256

def filter_by_length(example, tokenizer, max_tokens):
    encoding = tokenizer(example["full_text"], truncation=False)
    return len(encoding["input_ids"]) <= max_tokens

def tokenize_and_create_labels(example, tokenizer, max_length):
    prompt_ids = tokenizer(example["prompt_text"])["input_ids"]
    prompt_length = len(prompt_ids)
    # Tokenize full_text with truncation, ensuring the output does not exceed max_length
    tokenized = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=max_length,
    )
    labels = tokenized["input_ids"].copy()
    labels[:prompt_length] = [-100] * prompt_length
    tokenized["labels"] = labels
    return tokenized

# -------------------------------
# Main: Load, Transform, Concatenate, Tokenize, and Fine-Tune with Grid Search
# -------------------------------
def main():
    # Load datasets.
    commonsense_ds = load_dataset('hendrycks/ethics', 'commonsense', split='train')
    deontology_ds  = load_dataset('hendrycks/ethics', 'deontology', split='train')
    justice_ds     = load_dataset('hendrycks/ethics', 'justice', split='train')
    utility_ds     = load_dataset('hendrycks/ethics', 'utilitarianism', split='train')
    virtue_ds      = load_and_transform_virtue('hendrycks/ethics')
    
    # Apply transformations.
    commonsense_ds = commonsense_ds.map(transform_commonsense)
    deontology_ds  = deontology_ds.map(transform_deontology)
    justice_ds     = justice_ds.map(transform_justice)
    utility_ds = utility_ds.map(
        transform_utility_batched, 
        batched=True, 
        remove_columns=utility_ds.column_names
    )
    # Manually unbatch the utility dataset.
    utility_ds = manual_unbatch(utility_ds)
    # Virtue is already processed in load_and_transform_virtue.
    
    # Finalize examples.
    commonsense_ds = commonsense_ds.map(finalize_example)
    deontology_ds  = deontology_ds.map(finalize_example)
    justice_ds     = justice_ds.map(finalize_example)
    utility_ds     = utility_ds.map(finalize_example)
    virtue_ds      = virtue_ds.map(finalize_example)
    
    # Select the relevant columns.
    commonsense_ds = commonsense_ds.select_columns(["full_text", "prompt_text"])
    deontology_ds  = deontology_ds.select_columns(["full_text", "prompt_text"])
    justice_ds     = justice_ds.select_columns(["full_text", "prompt_text"])
    utility_ds     = utility_ds.select_columns(["full_text", "prompt_text"])
    virtue_ds      = virtue_ds.select_columns(["full_text", "prompt_text"])
    
    # Combine all datasets.
    combined_dataset = concatenate_datasets([
        commonsense_ds,
        deontology_ds,
        justice_ds,
        utility_ds,
        virtue_ds,
    ])
    print("Combined dataset loaded. Number of examples:", len(combined_dataset))
    
    # Save the combined (mega) fine-tune dataset to a CSV.
    df = combined_dataset.to_pandas()
    df.to_csv("mega_finetune_dataset.csv", index=False)
    print("Mega fine-tune dataset saved as mega_finetune_dataset.csv")
    
    # Initialize the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # --- Set maximum token lengths per example ---
    combined_dataset = combined_dataset.map(lambda ex: {"max_tokens": choose_max_tokens(ex)})
    
    # Filter examples that exceed the allowed token count.
    combined_dataset = combined_dataset.filter(lambda ex: filter_by_length(ex, tokenizer, ex["max_tokens"]))
    
    # --- Tokenize without pre-padding ---
    def tokenize_example(ex):
        return tokenize_and_create_labels(ex, tokenizer, max_length=ex["max_tokens"])
    
    tokenized_dataset = combined_dataset.map(tokenize_example, batched=False)
    # Rely on the data collator for dynamic padding.
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # -------------------------------
    # Initialize the Model and Apply LoRA
    # -------------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    # Set default maximum new tokens to 128.
    model.config.max_new_tokens = 128
    model.gradient_checkpointing_enable()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    print("LoRA adapters applied.")
    
    # -------------------------------
    # GRID SEARCH: Loop over hyperparameter configurations.
    # -------------------------------
    hyperparams_grid = [
        {"learning_rate": 1e-5, "per_device_train_batch_size": 1,  "num_train_epochs": 10}
    ]
    
    # Create a DataCollatorWithPadding to dynamically pad each batch.
    data_collator = DataCollatorWithPadding(tokenizer)
    
    for hp in hyperparams_grid:
        print(f"\nTraining with hyperparameters: LR = {hp['learning_rate']}, "
              f"Batch Size = {hp['per_device_train_batch_size']}, Epochs = {hp['num_train_epochs']}")
    
        training_args = TrainingArguments(
            output_dir=f"./ft_temp_lr{hp['learning_rate']}_bs{hp['per_device_train_batch_size']}_ep{hp['num_train_epochs']}",
            evaluation_strategy="no",  # Evaluation will be done manually later.
            num_train_epochs=4,
            per_device_train_batch_size=1,
            learning_rate=3e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=True,
            deepspeed="/fs/nas/eikthyrnir0/gpeterson/Fine_Tuning/ds_config.json",
        )
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    
        torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model(training_args.output_dir)
        print(f"Model saved to {training_args.output_dir}.")
    
    print("Grid search complete.")

if __name__ == "__main__":
    main()
