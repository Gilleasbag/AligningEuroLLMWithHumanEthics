import pandas as pd

# Load your existing dataset
df = pd.read_csv("commonsense_train.csv")

# Map labels to responses
label_map = {0: "acceptable", 1: "unacceptable"}
df['response'] = df['label'].map(label_map)

# Rename columns to 'prompt' and 'response'
df.rename(columns={'input': 'prompt'}, inplace=True)

# Save the new CSV
df[['prompt', 'response']].to_csv("commonsense_finetune_generation.csv", index=False)
