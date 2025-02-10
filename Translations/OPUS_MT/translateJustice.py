import os
import torch
import time
import csv
from datetime import datetime
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print("Using CPU")
    return device

def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'justice', split='validation+test+train')
    return dataset

def preprocess_function(examples, tokenizer):
    model_inputs = tokenizer(examples["scenario"], padding=True, truncation=True)
    # Include labels for later use
    model_inputs["labels"] = examples["label"]
    return model_inputs

def create_dataloader(dataset, tokenizer, batch_size):
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True, num_workers=4)

def model_initialization(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    model.to(device)
    model.eval()
    return model, tokenizer

def translate_dataset(model, tokenizer, device, dataloader, target_language):
    start_time = time.time()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Translating", unit="batch"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].tolist()

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=5, early_stopping=True)
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            results.extend(zip(translations, labels))

    total_time = time.time() - start_time
    save_translations_to_csv(results, total_time, target_language)

def save_translations_to_csv(results, total_time, target_language):
    filename = f"translation_justice_opus_mt_{target_language}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        # Write a header to describe the content with the total translation time
        writer.writerow(['label', 'scenario', f'Total Translation Time: {total_time:.2f} seconds'])
        # Write each result, where each result is a tuple of (translation, label)
        for translation, label in results:
            writer.writerow([label, translation.strip()])
    print(f"Translations have been saved to {filename}.")


def main(model_id, target_language):
    device = setup_device()
    dataset = load_and_prepare_data()
    model, tokenizer = model_initialization(model_id, device)
    batch_size = 48
    dataloader = create_dataloader(dataset, tokenizer, batch_size)
    translate_dataset(model, tokenizer, device, dataloader, target_language)

if __name__ == '__main__':
    languages = [("Helsinki-NLP/opus-mt-en-es", "es"), ("Helsinki-NLP/opus-mt-en-fr", "fr"), ("Helsinki-NLP/opus-mt-en-de", "de")]
    for model_id, target_language in languages:
        main(model_id, target_language)
