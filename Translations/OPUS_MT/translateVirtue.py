import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datetime import datetime
import csv
import time
from typing import List, Dict

class DataCollatorWithPaddingForScenario:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    
    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        # Collate sentences
        sentences = {
            'input_ids': [example['input_ids_sentence'] for example in batch],
            'attention_mask': [example['attention_mask_sentence'] for example in batch]
        }
        padded_sentences = self.collator(sentences)
        
        # Collate traits
        traits = {
            'input_ids': [example['input_ids_trait'] for example in batch],
            'attention_mask': [example['attention_mask_trait'] for example in batch]
        }
        padded_traits = self.collator(traits)
        
        labels = torch.tensor([example['labels'] for example in batch], dtype=torch.long)
        
        return {
            'input_ids_sentence': padded_sentences['input_ids'],
            'attention_mask_sentence': padded_sentences['attention_mask'],
            'input_ids_trait': padded_traits['input_ids'],
            'attention_mask_trait': padded_traits['attention_mask'],
            'labels': labels
        }

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device

def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'virtue', split='validation+test+train')
    return dataset

def preprocess_function(examples, tokenizer):
    # Split each scenario into sentence and trait
    split_scenarios = [scenario.split(" [SEP] ") for scenario in examples["scenario"]]
    sentences = [s[0] for s in split_scenarios]
    traits = [s[1] for s in split_scenarios]
    
    # Tokenize sentences and traits separately
    tokenized_sentences = tokenizer(
        sentences,
        padding=False,
        truncation=True,
        max_length=512
    )
    tokenized_traits = tokenizer(
        traits,
        padding=False,
        truncation=True,
        max_length=512
    )
    
    return {
        'input_ids_sentence': tokenized_sentences['input_ids'],
        'attention_mask_sentence': tokenized_sentences['attention_mask'],
        'input_ids_trait': tokenized_traits['input_ids'],
        'attention_mask_trait': tokenized_traits['attention_mask'],
        'labels': examples['label']
    }

def create_dataloader(dataset, tokenizer, batch_size):
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    tokenized_dataset.set_format('torch', columns=[
        'input_ids_sentence', 'attention_mask_sentence',
        'input_ids_trait', 'attention_mask_trait',
        'labels'
    ])
    
    data_collator = DataCollatorWithPaddingForScenario(tokenizer)
    
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=4
    )

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
            # Translate sentences
            input_ids_sentence = batch['input_ids_sentence'].to(device)
            attention_mask_sentence = batch['attention_mask_sentence'].to(device)
            outputs_sentence = model.generate(
                input_ids=input_ids_sentence,
                attention_mask=attention_mask_sentence,
                num_beams=5,
                early_stopping=True,
                max_length=512
            )
            translated_sentences = tokenizer.batch_decode(outputs_sentence, skip_special_tokens=True)
            
            # Translate traits
            input_ids_trait = batch['input_ids_trait'].to(device)
            attention_mask_trait = batch['attention_mask_trait'].to(device)
            outputs_trait = model.generate(
                input_ids=input_ids_trait,
                attention_mask=attention_mask_trait,
                num_beams=5,
                early_stopping=True,
                max_length=512
            )
            translated_traits = tokenizer.batch_decode(outputs_trait, skip_special_tokens=True)
            
            # Combine translated parts with [SEP]
            labels = batch['labels'].tolist()
            for label, sent, trait in zip(labels, translated_sentences, translated_traits):
                combined = f"{sent.strip()} [SEP] {trait.strip()}"
                results.append((label, combined))
    
    total_time = time.time() - start_time
    save_translations_to_csv(results, total_time, target_language)

def save_translations_to_csv(results, total_time, target_language):
    filename = f'translation_virtue_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'scenario'])
        for label, translated_scenario in results:
            writer.writerow([label, translated_scenario])
    
    print(f"Translations saved to {filename}. Total time: {total_time:.2f}s.")

def main(model_id, target_language):
    device = setup_device()
    dataset = load_and_prepare_data()
    model, tokenizer = model_initialization(model_id, device)
    batch_size = 64
    dataloader = create_dataloader(dataset, tokenizer, batch_size)
    translate_dataset(model, tokenizer, device, dataloader, target_language)

if __name__ == '__main__':
    languages = [
        ("Helsinki-NLP/opus-mt-en-de", "de"),
        ("Helsinki-NLP/opus-mt-en-fr", "fr"),
        ("Helsinki-NLP/opus-mt-en-es", "es")
    ]

    for model_id, target_language in languages:
        print(f"\nStarting translation for {target_language}")
        main(model_id, target_language)
        print(f"Completed {target_language}\n")