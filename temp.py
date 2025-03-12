import openai  # Use an appropriate LLM API
import pandas as pd
from tqdm import tqdm

# Load the dataset (modify path as needed)
DATASET_PATH = "story_cloze_test.csv"
df = pd.read_csv(DATASET_PATH)

# Function to query the LLM
def get_llm_prediction(story, ending1, ending2):
    prompt = f"""
    Here is a short story followed by two possible endings. Choose the most plausible ending.

    Story: {story}

    1. {ending1}
    2. {ending2}

    Respond with '1' or '2' based on the best ending.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Modify for other LLMs
        messages=[{"role": "system", "content": "You are a helpful AI."},
                  {"role": "user", "content": prompt}],
        temperature=0  # Make predictions deterministic
    )
    
    answer = response["choices"][0]["message"]["content"].strip()
    return answer if answer in ["1", "2"] else None  # Handle edge cases

# Evaluate the model
def evaluate_llm():
    correct = 0
    total = len(df)

    for _, row in tqdm(df.iterrows(), total=total):
        story = row["story"]
        ending1 = row["ending1"]
        ending2 = row["ending2"]
        correct_answer = str(row["correct_ending"])

        prediction = get_llm_prediction(story, ending1, ending2)
        if prediction == correct_answer:
            correct += 1

    accuracy = correct / total
    print(f"LLM Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    evaluate_llm()
