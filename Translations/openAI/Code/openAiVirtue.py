import os
import openai
import csv
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset

# Configure OpenAI client
def setup_openai_client():
    api_key = os.getenv('OPENAI_API_KEY')  # Get API key from the environment for security
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    return openai

# Function to handle translation using the OpenAI Chat API
def translate_text(openai_client, text, target_language):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",  # Ensure this model is correct or update it as needed.
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

# Load the Virtue dataset from Hugging Face
def load_and_prepare_data():
    # Loads the dataset using the 'virtue' configuration from 'hendrycks/ethics'
    dataset = load_dataset('hendrycks/ethics', 'virtue', split='validation+test+train', download_mode='force_redownload')
    return dataset

# Translate the Virtue dataset and combine the two translations with "[SEP]"
def translate_dataset(openai_client, dataset, target_language):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens_used = 0
    results = []
    
    for example in tqdm(dataset, desc=f"Translating to {target_language}", unit="example"):
        label = example["label"]
        scenario_text = example["scenario"]
        
        # Split the scenario into sentence and token parts using the "[SEP]" delimiter.
        parts = scenario_text.split("[SEP]")
        if len(parts) != 2:
            # Fallback: if the expected format is not found, treat the whole scenario as the sentence and set a blank token.
            sentence_text = scenario_text.strip()
            token_text = ""
        else:
            sentence_text = parts[0].strip()
            token_text = parts[1].strip()
        
        try:
            # Translate the sentence portion
            sentence_translation, tot_tkn_s, comp_tkn_s, prompt_tkn_s = translate_text(openai_client, sentence_text, target_language)
            # Translate the token portion (e.g., the adjective describing the scenario)
            token_translation, tot_tkn_t, comp_tkn_t, prompt_tkn_t = translate_text(openai_client, token_text, target_language)
            
            # Update token counts from both translation calls 
            total_prompt_tokens += (prompt_tkn_s + prompt_tkn_t)
            total_completion_tokens += (comp_tkn_s + comp_tkn_t)
            total_tokens_used += (tot_tkn_s + tot_tkn_t)
            
            # Combine the two translations into one output with "[SEP]" in between
            combined_translation = f"{sentence_translation} [SEP] {token_translation}"
            results.append((label, combined_translation))
        except Exception as e:
            print(f"Error translating text:\nSentence: {sentence_text}\nToken: {token_text}")
            print(e)
            results.append((label, "Error: Could not translate text."))
    
    # Calculate cost using example rates (Prompt: $0.15 per 1M tokens, Completion: $0.60 per 1M tokens)
    prompt_cost = (total_prompt_tokens / 1_000_000) * 0.15
    completion_cost = (total_completion_tokens / 1_000_000) * 0.60
    total_cost = prompt_cost + completion_cost
    
    # Save the combined translations along with token usage and cost statistics to a CSV file
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
    filename = f'virtue_translations_{target_language}_{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write header for translation results (combined translation in one column)
        writer.writerow(['label', 'translated_scenario'])
        for label, combined_translation in results:
            writer.writerow([label, combined_translation])
        # Write an empty row, and then token usage and cost statistics
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
    dataset = load_and_prepare_data()  # Load the Virtue dataset
    target_languages = ["Spanish", "French", "German"]
    
    for language in target_languages:
        translate_dataset(client, dataset, language)

if __name__ == '__main__':
    main()