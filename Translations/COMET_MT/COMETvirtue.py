import os
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

def download_comet_model(model_name):
    """
    Downloads the specified COMET model and returns its local path.
    """
    print(f"Downloading COMET model: {model_name}...")
    model_path = download_model(model_name)
    print(f"Model downloaded to: {model_path}")
    return model_path

def load_and_split_dataset(file_path, num_entries=None):
    """
    Loads the dataset from a CSV file and splits the 'scenario' column into 'sentence' and 'token'.
    
    Parameters:
        file_path (str): Path to the CSV file.
        num_entries (int, optional): Number of entries to load. Loads all if None.
        
    Returns:
        DataFrame: A pandas DataFrame with 'label', 'sentence', and 'token' columns.
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    if num_entries is not None:
        df = df.head(num_entries)
        
    print(df.iloc[0])
    # Split 'scenario' into 'sentence' and 'token'
    print("Splitting 'scenario' into 'sentence' and 'token'...")
    
    # Corrected split: Use regex escape or set regex=False
    # Option A: Escaping the brackets
    split_data = df['scenario'].str.split(r'\[SEP\]', expand=True)
    
    # Option B: Disable regex (uncomment the following line and comment out Option A if preferred)
    # split_data = df['scenario'].str.split('[SEP]', regex=False, expand=True)
    
    print(split_data.iloc[0])
    if split_data.shape[1] != 2:
        raise ValueError("Each 'scenario' entry must contain exactly one '[SEP]' separator.")
    
    df['sentence'] = split_data[0].str.strip()
    df['token'] = split_data[1].str.strip()
    
    # Optionally, drop the original 'scenario' column
    df = df.drop(columns=['scenario'])
    
    print(f"Loaded {len(df)} entries.")
    return df


def load_datasets_new(source_path, translation_paths, num_entries=None):
    """
    Loads and processes the source and translated datasets.
    
    Parameters:
        source_path (str): Path to the source (English) CSV file.
        translation_paths (dict): Dictionary mapping language codes to their translated CSV file paths.
        num_entries (int, optional): Number of entries to load from each dataset.
        
    Returns:
        dict: A dictionary containing processed DataFrames for each language.
    """
    datasets = {}
    
    # Load source dataset
    source_df = load_and_split_dataset(source_path, num_entries)
    datasets['en'] = source_df
    
    # Load translated datasets
    for lang, path in translation_paths.items():
        translated_df = load_and_split_dataset(path, num_entries)
        datasets[lang] = translated_df
    
    return datasets

def prepare_comet_data(source_sentences, translated_sentences, source_tokens, translated_tokens):
    """
    Prepares data dictionaries for COMET evaluation for sentences and tokens.
    
    Parameters:
        source_sentences (list): List of source sentences.
        translated_sentences (list): List of translated sentences.
        source_tokens (list): List of source tokens.
        translated_tokens (list): List of translated tokens.
        
    Returns:
        tuple: Two lists of dictionaries for sentences and tokens respectively.
    """
    print("Preparing data for COMET evaluation...")
    data_sentences = []
    data_tokens = []
    
    for src_sen, mt_sen, src_tok, mt_tok in zip(source_sentences, translated_sentences, source_tokens, translated_tokens):
        data_sentences.append({
            "src": src_sen.strip(),
            "mt": mt_sen.strip()
        })
        data_tokens.append({
            "src": src_tok.strip(),
            "mt": mt_tok.strip()
        })
    
    return data_sentences, data_tokens

def evaluate_comet(model, data, batch_size=64):
    """
    Evaluates translations using the provided COMET model.
    
    Parameters:
        model: The loaded COMET model.
        data (list): List of dictionaries with 'src' and 'mt' keys.
        batch_size (int): Number of samples per batch.
        
    Returns:
        list: COMET scores for each translation.
    """
    # Determine if a GPU is available
    if torch.cuda.is_available():
        accelerator = 'gpu'
        print("Using GPU for evaluation.")
    else:
        accelerator = 'cpu'
        print("Using CPU for evaluation.")

    print("Evaluating translations with COMET...")
    raw_scores = model.predict(
        data,
        batch_size=batch_size,
        accelerator=accelerator
    )
    scores = raw_scores.scores
    return scores

def save_combined_scores_to_csv(output_path, scores_sentences, scores_tokens, average_sent, average_tok):
    """
    Saves the COMET scores for sentences and tokens to a CSV file, including average scores.
    
    Parameters:
        output_path (str): Path to save the CSV file.
        scores_sentences (list): List of COMET scores for sentences.
        scores_tokens (list): List of COMET scores for tokens.
        average_sent (float): Average COMET score for sentences.
        average_tok (float): Average COMET score for tokens.
    """
    print(f"Saving combined scores to {output_path}...")
    df = pd.DataFrame({
        'sentence_COMET_score': scores_sentences,
        'token_COMET_score': scores_tokens
    })
    # Add average scores as the first row
    avg_row = {
        'sentence_COMET_score': average_sent,
        'token_COMET_score': average_tok
    }
    df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

def plot_score_distribution(scores, label):
    """
    Plots and saves the distribution of COMET scores.
    
    Parameters:
        scores (list): List of COMET scores.
        label (str): Label for the plot (e.g., 'English_Sentences').
    """
    plt.figure(figsize=(10,6))
    sns.histplot(scores, bins=50, kde=True)
    plt.title(f"COMET Score Distribution for {label.upper()}")
    plt.xlabel("COMET Score")
    plt.ylabel("Frequency")
    plot_filename = f"comet_score_distribution_{label}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Score distribution plot saved as {plot_filename}")

def main():
    # Define file paths
    source_csv = 'English_Datasets/virtue.csv'  # Replace with your actual source file path
    translation_csvs = {
        'de': 'Translations/OPUS_MT/translation_virtue_de_20250206-151330.csv',  # Replace with your actual translated file paths
        'es': 'Translations/OPUS_MT/translation_virtue_es_20250206-154521.csv',
        'fr': 'Translations/OPUS_MT/translation_virtue_fr_20250206-153059.csv'
    }

    # Define the COMET model to use
    comet_model_name = "Unbabel/wmt22-cometkiwi-da"

    # Step 1: Download COMET model
    print("Step 1: Downloading COMET model...")
    model_path = download_comet_model(comet_model_name)

    # Step 2: Load COMET model
    print("\nStep 2: Loading COMET model...")
    model = load_from_checkpoint(model_path)
    print("COMET model loaded successfully.")

    # Step 3: Load datasets
    print("\nStep 3: Loading datasets...")
    datasets = load_datasets_new(source_csv, translation_csvs)

    # Verify that all datasets have the same number of entries
    num_entries_loaded = len(datasets['en'])
    for lang, df in datasets.items():
        assert len(df) == num_entries_loaded, f"Number of entries in '{lang}' dataset does not match the source."
    print(f"All datasets are aligned with {num_entries_loaded} entries.")

    # Step 4: Evaluate each language
    languages = ['fr', 'de', 'es']  # Adjust as needed
    for lang in languages:
        print(f"\nEvaluating translations for language: {lang}")

        # Extract source and translated data
        source_df = datasets['en']
        translated_df = datasets[lang]

        source_sentences = source_df['sentence'].tolist()
        source_tokens = source_df['token'].tolist()

        translated_sentences = translated_df['sentence'].tolist()
        translated_tokens = translated_df['token'].tolist()

        # Prepare data for COMET evaluation
        data_sentences, data_tokens = prepare_comet_data(
            source_sentences, translated_sentences,
            source_tokens, translated_tokens
        )

        # Evaluate 'sentence' translations
        print("Evaluating 'sentence' translations...")
        scores_sentences = evaluate_comet(model, data_sentences, batch_size=48)
        scores_sentences = [float(score) for score in scores_sentences]

        # Evaluate 'token' translations
        print("Evaluating 'token' translations...")
        scores_tokens = evaluate_comet(model, data_tokens, batch_size=48)
        scores_tokens = [float(score) for score in scores_tokens]

        # Calculate average scores
        average_score_sent = sum(scores_sentences) / len(scores_sentences) if scores_sentences else 0
        average_score_tok = sum(scores_tokens) / len(scores_tokens) if scores_tokens else 0
        print(f"Average COMET Score for {lang} (sentence): {average_score_sent:.4f}")
        print(f"Average COMET Score for {lang} (token): {average_score_tok:.4f}")

        # Combine scores (simple average of sentence and token scores)
        combined_scores = [(s + t) / 2 for s, t in zip(scores_sentences, scores_tokens)]

        # Save combined scores
        output_csv = f"comet_scores_{lang}_combined.csv"
        save_combined_scores_to_csv(
            output_csv,
            scores_sentences,
            scores_tokens,
            average_score_sent,
            average_score_tok
        )

        # Optionally, save combined scores as well
        df_combined = pd.DataFrame({
            'combined_COMET_score': combined_scores
        })
        df_combined.to_csv(f"comet_scores_{lang}_only_combined.csv", index=False)
        print(f"Combined scores saved to comet_scores_{lang}_only_combined.csv")

        # Plot distributions
        print("Generating score distribution plots...")
        plot_score_distribution(scores_sentences, f"{lang}_sentence")
        plot_score_distribution(scores_tokens, f"{lang}_token")
        plot_score_distribution(combined_scores, f"{lang}_combined")

    print("\nEvaluation completed for all specified languages.")

if __name__ == "__main__":
    main()
