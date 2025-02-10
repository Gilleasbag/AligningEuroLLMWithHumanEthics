import os
import openai
from datasets import load_dataset

API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

API_BASE_URL = 'https://api.openai.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
}

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
        max_tokens=1024
    )
    translation = response.choices[0].message.content.strip()
    total_tokens = response.usage.total_tokens
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    print(f"Total tokens: {total_tokens}, Completion tokens: {completion_tokens}, Prompt tokens: {prompt_tokens}")
    print(f"Translation: {translation}")

# Load dataset
def load_and_prepare_data():
    dataset = load_dataset('hendrycks/ethics', 'commonsense', split='validation+test+train')
    return dataset

def main():
    client = setup_openai_client()
    dataset = load_and_prepare_data()
    target_language = "Spanish"  # You can change the target language here

    # Use only the first example
    example = dataset[0]
    text = example['input']
    label = example['label']

    # Translate the text
    translate_text(client, text, target_language)

    # Print the results
    print(f"Original Text: {text}")


if __name__ == '__main__':
    main()
