import os
import openai
import csv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Configure OpenAI client
def setup_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')  # Get API key from environment variable for security
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    return openai

# Function to handle translation using the OpenAI Chat API
def translate_text(openai_client, text, target_language):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this model is correct or update if needed.
        messages=[
            {"role": "user", "content": f"Translate this text to {target_language}: {text}. Give only the translation."}
        ],
        max_tokens=1024,
        temperature=0.0
    )
    translation = response.choices[0].message.content.strip()
    # Each response contains token usage details
    usage = response.usage
    total_tokens = response.usage.total_tokens
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    
    return translation, total_tokens, completion_tokens, prompt_tokens

# Load the Utilitarianism dataset from Hugging Face
def load_and_prepare_data():
    # Loads the dataset using the 'utilitarianism' configuration.
    dataset = load_dataset('hendrycks/ethics', 'utilitarianism', split='validation+test+train', download_mode='force_redownload')
    return dataset

# Translate the Utilitarianism dataset
def translate_dataset(openai_client, dataset, target_language):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_used = 0
    results = []

    # Iterate over each example in the utilitarianism dataset
    for example in tqdm(dataset, desc=f"Translating to {target_language}", unit="example"):
        baseline_text = example["baseline"]
        less_pleasant_text = example["less_pleasant"]

        try:
            # Translate the baseline text
            baseline_translation, tot_tkn_b, comp_tkn_b, prompt_tkn_b = translate_text(openai_client, baseline_text, target_language)
            # Translate the less_pleasant text
            less_pleasant_translation, tot_tkn_lp, comp_tkn_lp, prompt_tkn_lp = translate_text(openai_client, less_pleasant_text, target_language)
            
            # Update token counts from both translations
            total_prompt_tokens += (prompt_tkn_b + prompt_tkn_lp)
            total_completion_tokens += (comp_tkn_b + comp_tkn_lp)
            total_tokens_used += (tot_tkn_b + tot_tkn_lp)
            
            # Append the translations
            results.append((baseline_translation, less_pleasant_translation))
        except Exception as e:
            print(f"Error translating text:\nBaseline: {baseline_text}\nLess Pleasant: {less_pleasant_text}")
            print(e)
            # Append error messages if translation fails
            results.append(("Error: Could not translate baseline.", "Error: Could not translate less_pleasant."))

    # Calculate costs (example rates: Prompt Cost = $0.15 per 1M tokens, Completion Cost = $0.60 per 1M tokens)
    prompt_cost = (total_prompt_tokens / 1_000_000) * 0.15
    completion_cost = (total_completion_tokens / 1_000_000) * 0.60
    total_cost = prompt_cost + completion_cost

    # Save translations with token usage and cost statistics to a CSV file
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

# Save the translations and token usage statistics to a CSV file
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
    filename = f'utilitarianism_translations_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header for translations
        writer.writerow(['baseline', 'less_pleasant'])
        # Write the translation results
        for baseline_trans, less_pleasant_trans in results:
            writer.writerow([baseline_trans, less_pleasant_trans])
        # Write an empty row and then token usage and cost statistics
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
    dataset = load_and_prepare_data()  # Load the Utilitarianism dataset
    target_languages = ["Spanish", "French", "German"]

    for language in target_languages:
        translate_dataset(client, dataset, language)

if __name__ == '__main__':
    main()
