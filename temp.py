import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

# Load the dataset (modify path as needed)
DATASET_PATH = "/fs/nas/eikthyrnir0/gpeterson/clozeTest.csv"
df = pd.read_csv(DATASET_PATH)

# Setup device (GPU/CPU)
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device

# Load and prepare model and tokenizer
def model_initialization(model_id, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # Ensure pad_token is set for the model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Set padding side to 'left' for decoder-only models

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer

# Function to query the local LLM (Generate prediction)
def get_llm_prediction(story, ending1, ending2, model, tokenizer, device):
    prompt = f"""
    Here is a short story followed by two possible endings. Choose the most plausible ending.

    Story: {story}

    1. {ending1}
    2. {ending2}

    Respond with '1' or '2' based on the best ending.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs, 
        max_length=200, 
        num_return_sequences=1,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Check which ending (1 or 2) the model chose by looking at the prompt's structure
    if '1' in generated_text:
        return '1'
    elif '2' in generated_text:
        return '2'
    else:
        return None  # If model generates unexpected output

# Evaluate the model on the dataset
def evaluate_llm(model, tokenizer, device):
    correct = 0
    total = len(df)

    for _, row in tqdm(df.iterrows(), total=total):
        # Construct the story from the first four sentences
        story = " ".join([row["InputSentence1"], row["InputSentence2"], row["InputSentence3"], row["InputSentence4"]])
        
        # The two possible endings
        ending1 = row["RandomFifthSentenceQuiz1"]
        ending2 = row["RandomFifthSentenceQuiz2"]
        
        # Correct answer is expected to be in the 'ending1' or 'ending2', we assume 'RandomFifthSentenceQuiz1' is the correct answer.
        # If there's a 'correct_ending' column, use it. If not, we'll assume quiz1 is correct.
        correct_answer = '1'  # If you have a column to specify this, adjust this accordingly.

        prediction = get_llm_prediction(story, ending1, ending2, model, tokenizer, device)
        if prediction == correct_answer:
            correct += 1

    accuracy = correct / total
    print(f"LLM Accuracy: {accuracy:.2%}")

# Main function to load the model and evaluate
def main():
    model_id = "utter-project/EuroLLM-1.7B-Instruct"  # Example model, replace with your fine-tuned model
    device = setup_device()
    
    # Initialize model and tokenizer
    model, tokenizer = model_initialization(model_id, device)
    
    # Evaluate the model
    evaluate_llm(model, tokenizer, device)

if __name__ == "__main__":
    main()
