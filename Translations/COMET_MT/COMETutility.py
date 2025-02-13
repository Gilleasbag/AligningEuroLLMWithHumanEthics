#!/usr/bin/env python3
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

def load_utility_dataset(file_path, num_entries=None):
    """
    Loads a utility dataset from a CSV file that is expected to have two columns: 
    'baseline' and 'less_pleasant'.
    
    Parameters:
        file_path (str): Path to the CSV file.
        num_entries (int, optional): Number of entries to load. Loads all if None.
        
    Returns:
        DataFrame: A pandas DataFrame with 'baseline' and 'less_pleasant' columns.
    """
    print(f"Loading utility dataset from {file_path}...")
    df = pd.read_csv(file_path)

    required_cols = {'baseline', 'less_pleasant'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset at {file_path} must contain columns: {required_cols}")
        
    if num_entries is not None:
        df = df.head(num_entries)
    
    # Print the first row for verification
    print("First row of the dataset:")
    print(df.iloc[0])
    
    print(f"Loaded {len(df)} entries from {file_path}")
    return df

def load_datasets_new(source_path, translation_paths, num_entries=None):
    """
    Loads and processes both the source and translated utility datasets.
    
    Parameters:
        source_path (str): Path to the source CSV file.
        translation_paths (dict): Dictionary mapping language codes to their translated CSV file paths.
        num_entries (int, optional): Number of entries to load from each dataset.
        
    Returns:
        dict: A dictionary containing processed DataFrames for each language.
    """
    datasets = {}
    
    # Load source dataset
    source_df = load_utility_dataset(source_path, num_entries)
    datasets['en'] = source_df
    
    # Load translated datasets
    for lang, path in translation_paths.items():
        translated_df = load_utility_dataset(path, num_entries)
        datasets[lang] = translated_df
    
    return datasets

def prepare_comet_data(source_baseline, target_baseline, 
                       source_less_pleasant, target_less_pleasant):
    """
    Prepares data dictionaries for COMET evaluation for the baseline and less_pleasant columns.
    
    Parameters:
        source_baseline (list): List of baseline sentences from the source dataset.
        target_baseline (list): List of baseline sentences from the translated dataset.
        source_less_pleasant (list): List of less_pleasant sentences from the source dataset.
        target_less_pleasant (list): List of less_pleasant sentences from the translated dataset.
        
    Returns:
        tuple: Two lists of dictionaries corresponding to the baseline and less_pleasant evaluations.
    """
    print("Preparing data for COMET evaluation...")
    data_baseline = []
    data_less_pleasant = []
    
    for src_base, trg_base, src_lp, trg_lp in zip(source_baseline, target_baseline,
                                                  source_less_pleasant, target_less_pleasant):
        data_baseline.append({
            "src": src_base.strip(),
            "mt": trg_base.strip()
        })
        data_less_pleasant.append({
            "src": src_lp.strip(),
            "mt": trg_lp.strip()
        })
    
    return data_baseline, data_less_pleasant

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

def save_scores_to_csv(output_path, scores_baseline, scores_less_pleasant, avg_baseline, avg_less_pleasant):
    """
    Saves the COMET scores for baseline and less_pleasant translations to a CSV file, including average scores.
    
    Parameters:
        output_path (str): Path to save the CSV file.
        scores_baseline (list): List of COMET scores for baseline.
        scores_less_pleasant (list): List of COMET scores for less_pleasant.
        avg_baseline (float): Average COMET score for baseline.
        avg_less_pleasant (float): Average COMET score for less_pleasant.
    """
    print(f"Saving scores to {output_path}...")
    df = pd.DataFrame({
        'baseline_COMET_score': scores_baseline,
        'less_pleasant_COMET_score': scores_less_pleasant
    })
    # Insert average scores as the first row
    avg_row = {
        'baseline_COMET_score': avg_baseline,
        'less_pleasant_COMET_score': avg_less_pleasant
    }
    df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

def plot_score_distribution(scores, label):
    """
    Plots and saves the distribution of COMET scores.
    
    Parameters:
        scores (list): List of COMET scores.
        label (str): Label for the plot (e.g., 'en_baseline' or 'en_less_pleasant').
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, bins=50, kde=True)
    plt.title(f"COMET Score Distribution for {label.upper()}")
    plt.xlabel("COMET Score")
    plt.ylabel("Frequency")
    plot_filename = f"comet_score_distribution_{label}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Score distribution plot saved as {plot_filename}")

def main():
    # Define file paths for the utility dataset.
    source_csv = 'English_Datasets/utilitarianism.csv'  
    translation_csvs = {
        'de': 'Translations/openAI/Datasets/Combined/utilitarianism_translations_German_20250212-112737.csv',  
        'es': 'Translations/openAI/Datasets/Combined/utilitarianism_translations_Spanish_20250211-083810.csv',
        'fr': 'Translations/openAI/Datasets/Combined/utilitarianism_translations_French_20250211-221050.csv'
    }

    # Define the COMET model to use.
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

    # Evaluate each translation language for the utility dataset
    languages = ['fr', 'de', 'es']  # Adjust as needed
    for lang in languages:
        print(f"\nEvaluating translations for language: {lang}")

        # Extract data for baseline and less_pleasant
        source_df = datasets['en']
        translated_df = datasets[lang]

        source_baseline = source_df['baseline'].tolist()
        source_less_pleasant = source_df['less_pleasant'].tolist()
        target_baseline = translated_df['baseline'].tolist()
        target_less_pleasant = translated_df['less_pleasant'].tolist()

        # Prepare data for COMET evaluation for the two columns
        data_baseline, data_less_pleasant = prepare_comet_data(
            source_baseline, target_baseline,
            source_less_pleasant, target_less_pleasant
        )

        # Evaluate baseline translations
        print("Evaluating baseline translations...")
        scores_baseline = evaluate_comet(model, data_baseline, batch_size=48)
        scores_baseline = [float(score) for score in scores_baseline]

        # Evaluate less_pleasant translations
        print("Evaluating less_pleasant translations...")
        scores_less_pleasant = evaluate_comet(model, data_less_pleasant, batch_size=48)
        scores_less_pleasant = [float(score) for score in scores_less_pleasant]

        # Calculate average scores
        avg_baseline = sum(scores_baseline) / len(scores_baseline) if scores_baseline else 0
        avg_less_pleasant = sum(scores_less_pleasant) / len(scores_less_pleasant) if scores_less_pleasant else 0
        print(f"Average COMET Score for {lang} (baseline): {avg_baseline:.4f}")
        print(f"Average COMET Score for {lang} (less_pleasant): {avg_less_pleasant:.4f}")

        # Save scores to CSV
        output_csv = f"comet_scores_{lang}_utility.csv"
        save_scores_to_csv(output_csv, scores_baseline, scores_less_pleasant, avg_baseline, avg_less_pleasant)

        # Optionally, also save a CSV for combined scores if desired (here we do a simple average)
        combined_scores = [(base + lp) / 2 for base, lp in zip(scores_baseline, scores_less_pleasant)]
        df_combined = pd.DataFrame({
            'combined_COMET_score': combined_scores
        })
        combined_csv = f"comet_scores_{lang}_only_combined_utility.csv"
        df_combined.to_csv(combined_csv, index=False)
        print(f"Combined scores saved to {combined_csv}")

        # Generate score distribution plots
        print("Generating score distribution plots...")
        plot_score_distribution(scores_baseline, f"{lang}_baseline")
        plot_score_distribution(scores_less_pleasant, f"{lang}_less_pleasant")
        plot_score_distribution(combined_scores, f"{lang}_combined")

    print("\nEvaluation completed for all specified languages.")

if __name__ == "__main__":
    main()
