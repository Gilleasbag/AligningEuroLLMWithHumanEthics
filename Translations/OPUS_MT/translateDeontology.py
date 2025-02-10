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

# Custom Data Collator to handle both 'scenario' and 'excuse' fields
class DualDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.scenario_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        self.excuse_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        # Extract 'scenario' related fields
        scenarios = {
            'input_ids': [example['input_ids_scenario'] for example in batch],
            'attention_mask': [example['attention_mask_scenario'] for example in batch]
        }
        
        # Extract 'excuse' related fields
        excuses = {
            'input_ids': [example['input_ids_excuse'] for example in batch],
            'attention_mask': [example['attention_mask_excuse'] for example in batch]
        }
        
        # Pad scenarios
        padded_scenarios = self.scenario_collator(scenarios)
        
        # Pad excuses
        padded_excuses = self.excuse_collator(excuses)
        
        # Extract labels
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)
        
        return {
            'input_ids_scenario': padded_scenarios['input_ids'],
            'attention_mask_scenario': padded_scenarios['attention_mask'],
            'input_ids_excuse': padded_excuses['input_ids'],
            'attention_mask_excuse': padded_excuses['attention_mask'],
            'labels': labels
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
    dataset = load_dataset('hendrycks/ethics', 'deontology', split='validation+test+train')
    return dataset

def preprocess_function(examples, tokenizer):
    """
    Preprocess the dataset by tokenizing 'scenario' and 'excuse' separately.
    
    Args:
        examples: A batch of examples from the dataset.
        tokenizer: The tokenizer to use for encoding the prompts.
    
    Returns:
        A dictionary of tokenized inputs.
    """
    # Tokenize 'scenario'
    tokenized_scenarios = tokenizer(
        examples["scenario"],
        padding=False,  # Disable padding; handled by DataCollator
        truncation=True,
        max_length=512
    )
    
    # Tokenize 'excuse'
    tokenized_excuses = tokenizer(
        examples["excuse"],
        padding=False,  # Disable padding; handled by DataCollator
        truncation=True,
        max_length=512
    )
    
    return {
        "input_ids_scenario": tokenized_scenarios['input_ids'],
        "attention_mask_scenario": tokenized_scenarios['attention_mask'],
        "input_ids_excuse": tokenized_excuses['input_ids'],
        "attention_mask_excuse": tokenized_excuses['attention_mask'],
        "labels": examples['label']
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
    tokenized_dataset.set_format(type='torch', columns=[
        'input_ids_scenario', 'attention_mask_scenario',
        'input_ids_excuse', 'attention_mask_excuse', 'labels'
    ])
    
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
    """
    Translate the dataset and extract separate translations for scenario and excuse.
    
    Args:
        model: The translation model.
        tokenizer: The tokenizer associated with the model.
        device: The computation device (CPU or GPU).
        dataloader: DataLoader for batching the dataset.
        target_language (str): The target language code (e.g., 'es', 'fr', 'de').
    """
    start_time = time.time()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Translating", unit="batch"):
            # Translate 'scenario'
            input_ids_scenario = batch['input_ids_scenario'].to(device)
            attention_mask_scenario = batch['attention_mask_scenario'].to(device)
            
            outputs_scenario = model.generate(
                input_ids=input_ids_scenario,
                attention_mask=attention_mask_scenario,
                num_beams=5,
                early_stopping=True,
                max_length=512  # Adjust based on desired output length
            )
            translations_scenario = tokenizer.batch_decode(outputs_scenario, skip_special_tokens=True)
            
            # Translate 'excuse'
            input_ids_excuse = batch['input_ids_excuse'].to(device)
            attention_mask_excuse = batch['attention_mask_excuse'].to(device)
            
            outputs_excuse = model.generate(
                input_ids=input_ids_excuse,
                attention_mask=attention_mask_excuse,
                num_beams=5,
                early_stopping=True,
                max_length=512  # Adjust based on desired output length
            )
            translations_excuse = tokenizer.batch_decode(outputs_excuse, skip_special_tokens=True)
            
            # Get labels
            labels = batch['labels'].tolist()
            
            # Combine translations with labels
            for label, translated_scenario, translated_excuse in zip(labels, translations_scenario, translations_excuse):
                results.append((label, translated_scenario.strip(), translated_excuse.strip()))
    
    total_time = time.time() - start_time
    save_translations_to_csv(results, total_time, target_language)

def save_translations_to_csv(results, total_time, target_language):
    """
    Save the translation results to a CSV file with separate columns for scenario and excuse.
    
    Args:
        results (list of tuples): Each tuple contains (label, translated_scenario, translated_excuse).
        total_time (float): Total time taken for the translation process.
        target_language (str): The target language code.
    """
    filename = f'translation__deontology_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    
    # Write the results to a CSV file
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['label', 'scenario', 'excuse'])
        # Write each translated entry
        for label, translated_scenario, translated_excuse in results:
            writer.writerow([label, translated_scenario, translated_excuse])
    
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
