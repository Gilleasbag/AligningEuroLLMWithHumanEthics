import os
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def download_comet_model(model_name):
    print(f"Downloading COMET model: {model_name}...")
    model_path = download_model(model_name)
    print(f"Model downloaded to: {model_path}")
    return model_path

def load_datasets_with_hf(source_path, translation_paths, num_entries=None):
    datasets = {}
    # Load source data
    print(f"Loading source dataset from {source_path}...")
    source_dataset = load_dataset('csv', data_files=source_path, split='train')
    if num_entries is not None:
        source_dataset = source_dataset.select(range(num_entries))
    sources = source_dataset['scenario']
    datasets['en'] = sources

    # Load translated data
    for lang, path in translation_paths.items():
        print(f"Loading translated dataset for '{lang}' from {path}...")
        translated_dataset = load_dataset('csv', data_files=path, split='train')
        if num_entries is not None:
            translated_dataset = translated_dataset.select(range(num_entries))
        # Adjust the column name if necessary
        translations = translated_dataset['scenario']
        datasets[lang] = translations

    return datasets

def prepare_comet_data(source_texts, translated_texts):
    data = []
    for src, mt in zip(source_texts, translated_texts):
        data.append({
            "src": src.strip(),
            "mt": mt.strip()
        })
    return data

def evaluate_comet(model, data, batch_size=64):
    """
    Evaluates translations using the provided COMET model.
    Returns a list of COMET scores.
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

def save_scores_to_csv(output_path, scores, average_score):
    df = pd.DataFrame({
        'COMET_Score': scores
    })
    # save average score to the first row
    df.loc[-1] = [average_score]
    df.index = df.index + 1
    df = df.sort_index()
    
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

def plot_score_distribution(scores, language):
    plt.figure(figsize=(10,6))
    sns.histplot(scores, bins=50, kde=True)
    plt.title(f"COMET Score Distribution for {language.upper()}")
    plt.xlabel("COMET Score")
    plt.ylabel("Frequency")
    plt.savefig(f"comet_score_distribution_{language}_test.png")
    plt.close()
    print(f"Score distribution plot saved as comet_score_distribution_{language}_test.png")

def main():
    # Define file paths
    source_csv = 'English_Datasets/justice.csv'
    translation_csvs = {
        'fr': 'Translations/openAI/Datasets/Combined/justice_translations_French_20250207-095623.csv',
        'de': 'Translations/openAI/Datasets/Combined/justice_translations_German_20250207-181451.csv',
        'es': 'Translations/openAI/Datasets/Combined/justice_translations_Spanish_20250207-013445.csv'
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
    
    # Step 3: Load datasets using Hugging Face's datasets library
    print("\nStep 3: Loading datasets using Hugging Face's datasets library...")
    datasets = load_datasets_with_hf(source_csv, translation_csvs)
    
    # Verify that all datasets have the same number of entries
    num_entries_loaded = len(datasets['en'])
    for lang, texts in datasets.items():
        assert len(texts) == num_entries_loaded, f"Number of entries in '{lang}' dataset does not match the source."
    print(f"All datasets are aligned with {num_entries_loaded} entries.")
    
    # Step 4: Evaluate each language using the COMET model
    for lang in ['fr', 'de', 'es']:
        print(f"\nEvaluating translations for language: {lang}")
        
        # Prepare data
        source_texts = datasets['en']
        translated_texts = datasets[lang]
        data = prepare_comet_data(source_texts, translated_texts)
        
        # Evaluate
        scores = evaluate_comet(model, data, batch_size=48)
        # Verify and convert scores if necessary
        if all(isinstance(score, float) for score in scores):
            print("All scores are floats.")
        else:
            print("Converting scores to floats...")
            scores = [float(score) for score in scores]
            
        # Calculate and print average score
        average_score = sum(scores) / len(scores)
        print(f"Average COMET Score for {lang}: {average_score:.4f}")

        # Save scores
        output_csv = f"comet_scores_{lang}_test.csv"
        save_scores_to_csv(output_csv, scores, average_score)

        # Plot distribution
        print("Generating score distribution plot...")
        plot_score_distribution(scores, lang)
    
    print("\nEvaluation completed for all specified languages.")

if __name__ == "__main__":
    main()
