import os
import openai
import json
import time
import csv
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm

# Configure OpenAI
def setup_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable for security
      # Set the API key for the OpenAI library
    return openai

# Generate JSONL input file for batch processing
def generate_jsonl_input_file(dataset, target_language, split_name):
    jsonl_filename = f'batch_input_{target_language}_{split_name}.jsonl'
    with open(jsonl_filename, 'w', encoding='utf-8') as jsonl_file:
        for idx, example in enumerate(dataset):
            request = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4",  # Ensure you have access to this model
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Translate the following English text to {target_language}. Only provide the translation.\n\n\"{example['input']}\""
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0  # For deterministic output
                }
            }
            jsonl_file.write(json.dumps(request) + '\n')
    print(f"Generated JSONL input file: {jsonl_filename}")
    return jsonl_filename

# Upload input file to OpenAI
def upload_input_file(openai_client, jsonl_filename):
    response = openai_client.File.create(
        file=open(jsonl_filename, "rb"),
        purpose='batch'
    )
    file_id = response['id']
    print(f"Uploaded input file. File ID: {file_id}")
    return file_id

# Create a batch request
def create_batch_request(openai_client, input_file_id):
    response = openai_client.Batch.create(
        input_file_id=input_file_id,
        completion_window='24h',
        endpoint='/v1/chat/completions'
    )
    batch_id = response['id']
    print(f"Created batch request. Batch ID: {batch_id}")
    return batch_id

# Poll batch status until completion
def wait_for_batch_completion(openai_client, batch_id):
    print("Waiting for batch processing to complete...")
    while True:
        batch = openai_client.Batch.retrieve(batch_id)
        status = batch['status']
        print(f"Batch status: {status}")
        if status == 'completed':
            print("Batch processing completed.")
            return batch['output_file_id'], batch['error_file_id']
        elif status in ['failed', 'cancelled', 'cancelled', 'expired']:
            raise Exception(f"Batch processing {status}.")
        time.sleep(60)  # Wait for 1 minute before checking again

# Download and parse output file
def download_and_parse_output(openai_client, output_file_id, num_requests):
    output_file = openai_client.File.download(output_file_id)
    results = [None] * num_requests  # Placeholder for results

    for line in output_file.iter_lines():
        if line:
            response_obj = json.loads(line)
            custom_id = response_obj['custom_id']
            idx = int(custom_id.split('-')[-1])
            response = response_obj.get('response')

            if response and response.get('status_code') == 200:
                content = response['body']['choices'][0]['message']['content'].strip()
                results[idx] = content
            else:
                results[idx] = ''  # Handle error cases or empty responses
    return results

# Save results to CSV
def save_translations_to_csv(labels, translations, target_language, split_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'openai_{target_language}_{split_name}_{timestamp}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'input'])
        for label, translation in zip(labels, translations):
            writer.writerow([label, translation])
    print(f"Translations for {split_name} set have been saved to {filename}.")

def main():
    openai_client = setup_openai_client()
    datasets = load_and_prepare_data()
    target_languages = ["Spanish", "French", "German"]  # Add or modify target languages as needed

    for language in target_languages:
        for split_name, split_dataset in datasets.items():
            print(f"Processing {split_name} set for language: {language}")

            # Step 1: Generate JSONL file
            jsonl_filename = generate_jsonl_input_file(split_dataset, language, split_name)

            # Step 2: Upload input file
            input_file_id = upload_input_file(openai_client, jsonl_filename)

            # Step 3: Create batch request
            batch_id = create_batch_request(openai_client, input_file_id)

            # Step 4: Wait for batch completion
            try:
                output_file_id, error_file_id = wait_for_batch_completion(openai_client, batch_id)
            except Exception as e:
                print(f"Error during batch processing: {e}")
                continue  # Skip to next dataset or language

            # Step 5: Download and parse output file
            print("Downloading and parsing output file...")
            num_requests = len(split_dataset)
            translations = download_and_parse_output(openai_client, output_file_id, num_requests)

            # Step 6: Save translations to CSV
            labels = [example['label'] for example in split_dataset]
            save_translations_to_csv(labels, translations, language, split_name)

def load_and_prepare_data():
    train = load_dataset('hendrycks/ethics', 'commonsense', split='train')        # Training set
    test = load_dataset('hendrycks/ethics', 'commonsense', split='validation')    # Test set
    hard_test = load_dataset('hendrycks/ethics', 'commonsense', split='test')     # Hard test set
    datasets = {'train': train, 'test': test, 'hard_test': hard_test}
    return datasets

if __name__ == '__main__':
    main()
