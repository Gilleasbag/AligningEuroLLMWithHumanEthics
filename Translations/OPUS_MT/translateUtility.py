import os
import torch
import time
import csv
import re  # For regex operations
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Dict

class DualDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.baseline_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        self.less_pleasant_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        baselines = {
            'input_ids': [example['input_ids_baseline'] for example in batch],
            'attention_mask': [example['attention_mask_baseline'] for example in batch]
        }
        less_pleasant = {
            'input_ids': [example['input_ids_less_pleasant'] for example in batch],
            'attention_mask': [example['attention_mask_less_pleasant'] for example in batch]
        }
        padded_baselines = self.baseline_collator(baselines)
        padded_less_pleasant = self.less_pleasant_collator(less_pleasant)

        return {
            'input_ids_baseline': padded_baselines['input_ids'],
            'attention_mask_baseline': padded_baselines['attention_mask'],
            'input_ids_less_pleasant': padded_less_pleasant['input_ids'],
            'attention_mask_less_pleasant': padded_less_pleasant['attention_mask']
        }

def setup_device():
    """Set up the computation device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device

def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'utilitarianism', split='validation+test+train')
    return dataset

def preprocess_function(examples, tokenizer):
    """
    Preprocess the dataset by tokenizing 'baseline' and 'less_pleasant' separately.
    """
    tokenized_baseline = tokenizer(
        examples["baseline"],
        padding=False,  # Disable padding; handled by DataCollator
        truncation=True,
        max_length=512
    )
    
    tokenized_less_pleasant = tokenizer(
        examples["less_pleasant"],
        padding=False,
        truncation=True,
        max_length=512
    )
    
    return {
        "input_ids_baseline": tokenized_baseline['input_ids'],
        "attention_mask_baseline": tokenized_baseline['attention_mask'],
        "input_ids_less_pleasant": tokenized_less_pleasant['input_ids'],
        "attention_mask_less_pleasant": tokenized_less_pleasant['attention_mask']
    }

def create_dataloader(dataset, tokenizer, batch_size):
    """
    Create a DataLoader from the dataset.
    
    Args:
        dataset: The concatenated dataset containing all splits.
        tokenizer: The tokenizer to use for encoding.
        batch_size (int): The number of samples per batch.
    
    Returns:
        DataLoader: A PyTorch DataLoader with the tokenized data.
    """
    # Apply the preprocessing function to the dataset
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Set the format of the dataset to PyTorch tensors
    tokenized_dataset.set_format(
        type='torch',
        columns=[
            'input_ids_baseline', 
            'attention_mask_baseline', 
            'input_ids_less_pleasant', 
            'attention_mask_less_pleasant'
        ]
    )
    
    # Initialize the custom data collator
    data_collator = DualDataCollator(tokenizer=tokenizer)
    
    # Create the DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4  # Adjust based on your CPU cores
    )
    
    return dataloader


def model_initialization(model_id, device):
    """
    Initialize the tokenizer and model.
    
    Args:
        model_id (str): The Hugging Face model identifier.
        device: The computation device.
    
    Returns:
        tuple: (model, tokenizer)
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    print(f"Tokenizer pad_token: {tokenizer.pad_token}")  # Debug statement
    
    # Load the model with appropriate precision
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    
    # Move the model to the computation device
    model.to(device)
    model.eval()
    
    return model, tokenizer

def translate_dataset(model, tokenizer, device, dataloader, target_language):
    start_time = time.time()
    results = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Translating", unit="batch"):
            # Correcting keys to match those from the utility dataset
            input_ids_baseline = batch['input_ids_baseline'].to(device)
            attention_mask_baseline = batch['attention_mask_baseline'].to(device)

            output_ids_baseline = model.generate(
                input_ids=input_ids_baseline,
                attention_mask=attention_mask_baseline,
                num_beams=5,
                early_stopping=True,
                max_length=512  # Adjust based on desired output length
            )
            translations_baseline = tokenizer.batch_decode(output_ids_baseline, skip_special_tokens=True)

            # Translate 'less pleasant'
            input_ids_less_pleasant = batch['input_ids_less_pleasant'].to(device)
            attention_mask_less_pleasant = batch['attention_mask_less_pleasant'].to(device)

            output_ids_less_pleasant = model.generate(
                input_ids=input_ids_less_pleasant,
                attention_mask=attention_mask_less_pleasant,
                num_beams=5,
                early_stopping=True,
                max_length=512  # Adjust based on desired output length
            )
            translations_less_pleasant = tokenizer.batch_decode(output_ids_less_pleasant, skip_special_tokens=True)

            for baseline, less_pleasant in zip(translations_baseline, translations_less_pleasant):
                results.append((baseline.strip(), less_pleasant.strip()))

    total_time = time.time() - start_time
    save_translations_to_csv(results, total_time, target_language)

def save_translations_to_csv(results, total_time, target_language):
    filename = f'translation_utilitarianism_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'

    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['baseline', 'less_pleasant'])  # Ensure header matches columns
        for translated_baseline, translated_less_pleasant in results:
            writer.writerow([translated_baseline, translated_less_pleasant])

    print(f"Translations have been saved to {filename}. Total translation time: {total_time:.2f} seconds.")



def main(model_id, target_language):
    """
    The main function orchestrating the translation pipeline.
    
    Args:
        model_id (str): The Hugging Face model identifier.
        target_language (str): The target language code.
    """
    device = setup_device()
    dataset = load_and_prepare_data()
    model, tokenizer = model_initialization(model_id, device)
    batch_size = 64  # Adjust based on your GPU memory
    dataloader = create_dataloader(dataset, tokenizer, batch_size)
    translate_dataset(model, tokenizer, device, dataloader, target_language)

if __name__ == '__main__':
    # Define the target languages and corresponding models
    languages = [
        ("Helsinki-NLP/opus-mt-en-es", "es"), 
        ("Helsinki-NLP/opus-mt-en-fr", "fr"), 
        ("Helsinki-NLP/opus-mt-en-de", "de")
    ]
    
    for model_id, target_language in languages:
        print(f"\nStarting translation for language: {target_language}")
        main(model_id, target_language)
        print(f"Completed translation for language: {target_language}\n")
