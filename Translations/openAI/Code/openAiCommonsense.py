import os
import openai
import csv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Configure OpenAI
# Set your OpenAI API key and endpoint
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

API_BASE_URL = 'https://api.openai.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
}


# Function to handle translation using the OpenAI Chat API
def translate_text(openai_client, text, target_language):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this model is correct
        messages=[
            {"role": "user", "content": f"Translate this text to {target_language}: {text}. Give only the translation."}
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()


# Load dataset
def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'commonsense', split='validation+test+train')
    return dataset

# Translate the dataset
def translate_dataset(openai_client, dataset, target_language):
    start_time = datetime.now()
    results = []

    for example in tqdm(dataset, desc="Translating", unit="text"):
        translation = translate_text(openai_client, example['input'], target_language)
        results.append((example['label'], translation))

    total_time = datetime.now() - start_time
    save_translations_to_csv(results, total_time.total_seconds(), target_language)

# Save results to CSV
def save_translations_to_csv(results, total_time, target_language):
    filename = f'openai_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['label', 'input', f'Total Translation Time: {total_time:.2f} seconds'])
        for result in results:
            writer.writerow([result[0], result[1]])
    print(f"Translations have been saved to {filename}.")

def main():
    openai_client = setup_openai_client()
    dataset = load_and_prepare_data()
    target_languages = ["Spanish", "French", "German"]

    for language in target_languages:
        translate_dataset(openai_client, dataset, language)

if __name__ == '__main__':
    main()
