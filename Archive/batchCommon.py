import os
import requests
import csv
import json
import time
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Set your OpenAI API key and endpoint
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

API_BASE_URL = 'https://api.openai.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
}

# Load dataset
def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'justice', split='validation[:50]')
    return dataset

# Create batch input file
def create_batch_input_file(dataset, target_language):
    input_filename = f'batch_input_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.jsonl'
    with open(input_filename, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(dataset):
            text = example['scenario']
            prompt = f"Translate this text to {target_language}: {text}\n\nGive only the translation."
            request_data = {
                "custom_id": f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",  # Replace with the model you have access to
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1024,
                    "temperature": 0  # Optional: set temperature to 0 for deterministic results
                }
            }
            f.write(json.dumps(request_data, ensure_ascii=False) + '\n')
    print(f"Batch input file created: {input_filename}")
    return input_filename

# Upload input file to OpenAI
def upload_input_file(input_filename):
    upload_url = f'{API_BASE_URL}/files'
    with open(input_filename, 'rb') as f:
        files = {
            'file': (input_filename, f, 'application/jsonl'),
            'purpose': (None, 'batch'),
        }
        response = requests.post(
            upload_url,
            headers={'Authorization': f'Bearer {API_KEY}'},
            files=files
        )
    if response.status_code != 200:
        print(f"Error uploading file: {response.status_code} {response.text}")
        response.raise_for_status()
    file_info = response.json()
    file_id = file_info['id']
    print(f"Input file uploaded. File ID: {file_id}")
    return file_id

# Create batch job
def create_batch_job(input_file_id):
    create_batch_url = f'{API_BASE_URL}/batches'
    data = {
        'input_file_id': input_file_id,
        'endpoint': '/v1/chat/completions',
        'completion_window': '24h',
    }
    response = requests.post(create_batch_url, headers={**HEADERS, 'Content-Type': 'application/json'}, json=data)
    if response.status_code != 200:
        print(f"Error creating batch: {response.status_code} {response.text}")
        response.raise_for_status()
    batch_info = response.json()
    batch_id = batch_info['id']
    print(f"Batch job created. Batch ID: {batch_id}")
    return batch_id

# Monitor batch status
def wait_for_batch_completion(batch_id):
    batch_url = f'{API_BASE_URL}/batches/{batch_id}'
    while True:
        response = requests.get(batch_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"Error checking batch status: {response.status_code} {response.text}")
            response.raise_for_status()
        batch_info = response.json()
        status = batch_info['status']
        print(f"Batch {batch_id} status: {status}")
        if status == 'completed':
            print("Batch processing completed.")
            return batch_info
        elif status in ['failed', 'cancelled', 'expired']:
            raise Exception(f"Batch processing ended with status: {status}")
        time.sleep(60)  # Wait before checking again

# Download and save results
def download_and_save_results(batch_info, target_language, dataset):
    output_file_id = batch_info.get('output_file_id')
    if not output_file_id:
        raise Exception("No output file ID found in the batch info.")

    # Download the output file content
    file_url = f"{API_BASE_URL}/files/{output_file_id}/content"
    response = requests.get(file_url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Error downloading output file: {response.status_code} {response.text}")
        response.raise_for_status()
    output_content = response.content.decode('utf-8')

    # Parse the output and extract translations
    translations_dict = {}
    for line in output_content.strip().split('\n'):
        result = json.loads(line)
        custom_id = result['custom_id']
        idx = int(custom_id.split('-')[1])
        if result.get('error'):
            print(f"Error in request {custom_id}: {result['error']['message']}")
            translation = None
        else:
            translation = result['response']['body']['choices'][0]['message']['content'].strip()
        translations_dict[idx] = translation

    # Prepare results by matching translations back to the dataset entries
    results = []
    for idx, example in enumerate(dataset):
        label = example['label']
        translation = translations_dict.get(idx)
        results.append((label, translation))

    # Save translations to CSV
    filename = f'openai_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'translation'])
        for result in results:
            writer.writerow([result[0], result[1]])
    print(f"Translations have been saved to {filename}.")

# Main function
def main():
    dataset = load_and_prepare_data()
    target_languages = ["Spanish", "French", "German"]

    for language in target_languages:
        print(f"\nProcessing translations for {language}...")
        start_time = datetime.now()

        # Step 1: Prepare batch input file
        input_filename = create_batch_input_file(dataset, language)

        # Step 2: Upload input file
        input_file_id = upload_input_file(input_filename)

        # Step 3: Create batch job
        batch_id = create_batch_job(input_file_id)

        # Step 4: Monitor batch status
        batch_info = wait_for_batch_completion(batch_id)

        # Step 5: Retrieve and process results
        download_and_save_results(batch_info, language, dataset)

        total_time = datetime.now() - start_time
        print(f"Total Translation Time for {language}: {total_time}")

    print("\nAll translations completed.")

if __name__ == '__main__':
    main()
