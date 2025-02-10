import os
import openai
import csv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Configure OpenAI
def setup_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable for security
    return openai.OpenAI(api_key=api_key)

# Function to handle translation using the OpenAI Chat API
def translate_text(openai_client, text, target_language):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this model is correct
        messages=[
            {"role": "user", "content": f"Translate this text to {target_language}: {text}. Give only the translation."}
        ],
        max_tokens=1024,
        temperature=0.0
    )
    translation = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    return translation, total_tokens, completion_tokens, prompt_tokens



# Load dataset
def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'justice', split='validation+test+train', download_mode='force_redownload')
    return dataset

# Translate the dataset
def translate_dataset(openai_client, dataset, target_language):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_used = 0
    results = []

    for example in tqdm(dataset, desc=f"Translating to {target_language}", unit="example"):
        text = example['scenario']
        label = example['label']
        try:
            translation, total_tokens, completion_tokens, prompt_tokens = translate_text(openai_client, text, target_language)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens_used += total_tokens
            results.append((label, translation))
        except Exception as e:
            print(f"Error translating text: {text}")
            print(e)
            # Handle the error by appending an error message or the original text
            results.append((label, f"Error: Could not translate text."))

    # Calculate costs
    prompt_cost = (total_prompt_tokens / 1_000_000) * 0.15
    completion_cost = (total_completion_tokens / 1_000_000) * 0.60
    total_cost = prompt_cost + completion_cost

    # Save translations and token usage to CSV
    save_translations_to_csv(
        results,
        total_prompt_tokens,
        total_completion_tokens,
        total_tokens_used,
        prompt_cost,
        completion_cost,
        total_cost,
        target_language
    )

# Save results to CSV
def save_translations_to_csv(
    results,
    total_prompt_tokens,
    total_completion_tokens,
    total_tokens_used,
    prompt_cost,
    completion_cost,
    total_cost,
    target_language
):
    filename = f'justice_translations_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['label', 'scenario'])
        # Write translation results
        for result in results:
            writer.writerow([result[0], result[1]])
        # Leave a blank line and write token usage and cost statistics
        writer.writerow([])
        writer.writerow(['Total Prompt Tokens', total_prompt_tokens])
        writer.writerow(['Total Completion Tokens', total_completion_tokens])
        writer.writerow(['Total Tokens Used', total_tokens_used])
        writer.writerow([])
        writer.writerow(['Prompt Cost (USD)', f"${prompt_cost:.6f}"])
        writer.writerow(['Completion Cost (USD)', f"${completion_cost:.6f}"])
        writer.writerow(['Total Cost (USD)', f"${total_cost:.6f}"])
    print(f"Translations and token usage have been saved to {filename}.")
    print(f"Total Cost for {target_language}: ${total_cost:.6f}")

def main():
    client = setup_openai_client()
    dataset = load_and_prepare_data()
    target_languages = ["Spanish", "French", "German"]

    for language in target_languages:
        translate_dataset(client, dataset, language)

if __name__ == '__main__':
    main()