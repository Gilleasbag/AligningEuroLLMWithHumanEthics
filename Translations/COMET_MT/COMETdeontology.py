import os
import pandas as pd
import torch
from comet import download_model, load_from_checkpoint
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

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
    # Load 'scenario' and 'excuse' columns
    sources_scenario = source_dataset['scenario']
    sources_excuse = source_dataset['excuse']
    datasets['en'] = {'scenario': sources_scenario, 'excuse': sources_excuse}

    # Load translated data
    for lang, path in translation_paths.items():
        print(f"Loading translated dataset for '{lang}' from {path}...")
        translated_dataset = load_dataset('csv', data_files=path, split='train')
        if num_entries is not None:
            translated_dataset = translated_dataset.select(range(num_entries))
        # Load translated 'scenario' and 'excuse'
        translations_scenario = translated_dataset['scenario']
        translations_excuse = translated_dataset['excuse']
        datasets[lang] = {'scenario': translations_scenario, 'excuse': translations_excuse}

    return datasets

def prepare_comet_data_double(source_texts_scenario, translated_texts_scenario,
                              source_texts_excuse, translated_texts_excuse):
    data_scenario = []
    data_excuse = []
    for src_scenario, mt_scenario, src_excuse, mt_excuse in zip(
        source_texts_scenario, translated_texts_scenario,
        source_texts_excuse, translated_texts_excuse):
        data_scenario.append({
            "src": src_scenario.strip(),
            "mt": mt_scenario.strip()
        })
        data_excuse.append({
            "src": src_excuse.strip(),
            "mt": mt_excuse.strip()
        })
    return data_scenario, data_excuse

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

def save_combined_scores_to_csv(output_path, scores_scenario, scores_excuse,
                                average_score_scenario, average_score_excuse):
    df = pd.DataFrame({
        'scenario_COMET_score': scores_scenario,
        'excuse_COMET_score': scores_excuse
    })
    # Add average scores to the first row
    avg_row = {
        'scenario_COMET_score': average_score_scenario,
        'excuse_COMET_score': average_score_excuse
    }
    df = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

def plot_score_distribution(scores, label):
    plt.figure(figsize=(10,6))
    sns.histplot(scores, bins=50, kde=True)
    plt.title(f"COMET Score Distribution for {label.upper()}")
    plt.xlabel("COMET Score")
    plt.ylabel("Frequency")
    plt.savefig(f"comet_score_distribution_{label}.png")
    plt.close()
    print(f"Score distribution plot saved as comet_score_distribution_{label}.png")

def main():
    # Define file paths (UPDATE THESE TO ABSOLUTE PATHS)
    base_path = "/fs/nas/eikthyrnir0/gpeterson/"
    
    source_csv = f"{base_path}English_Datasets/deontology.csv"
    translation_csvs = {
        'fr': f"{base_path}Translations/openAI/Datasets/Combined/deontology_translations_French_20250211-203527.csv",
        'de': f"{base_path}Translations/openAI/Datasets/Combined/deontology_translations_German_20250212-085007.csv",
        'es': f"{base_path}Translations/openAI/Datasets/Combined/deontology_translations_Spanish_20250211-075051.csv"
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
    datasets = load_datasets_with_hf(source_csv, translation_csvs)

    # Verify that all datasets have the same number of entries
    num_entries_loaded = len(datasets['en']['scenario'])
    for lang in datasets.keys():
        assert len(datasets[lang]['scenario']) == num_entries_loaded, f"Number of entries in '{lang}' dataset does not match the source."
        assert len(datasets[lang]['excuse']) == num_entries_loaded, f"Number of entries in '{lang}' dataset does not match the source."
    print(f"All datasets are aligned with {num_entries_loaded} entries.")

    # Step 4: Evaluate each language
    languages = ['fr', 'de', 'es']  # Adjust as needed
    for lang in languages:
        print(f"\nEvaluating translations for language: {lang}")

        # Prepare data for COMET evaluation
        source_texts_scenario = datasets['en']['scenario']
        translated_texts_scenario = datasets[lang]['scenario']
        source_texts_excuse = datasets['en']['excuse']
        translated_texts_excuse = datasets[lang]['excuse']

        data_scenario, data_excuse = prepare_comet_data_double(
            source_texts_scenario, translated_texts_scenario,
            source_texts_excuse, translated_texts_excuse
        )

        # Evaluate 'scenario' translations
        print("Evaluating 'scenario' translations...")
        scores_scenario = evaluate_comet(model, data_scenario, batch_size=48)
        scores_scenario = [float(score) for score in scores_scenario]

        # Evaluate 'excuse' translations
        print("Evaluating 'excuse' translations...")
        scores_excuse = evaluate_comet(model, data_excuse, batch_size=48)
        scores_excuse = [float(score) for score in scores_excuse]

        # Calculate average scores
        average_score_scenario = sum(scores_scenario) / len(scores_scenario)
        average_score_excuse = sum(scores_excuse) / len(scores_excuse)
        print(f"Average COMET Score for {lang} (scenario): {average_score_scenario:.4f}")
        print(f"Average COMET Score for {lang} (excuse): {average_score_excuse:.4f}")

        # Save combined scores
        output_csv = f"comet_scores_{lang}_combined.csv"
        save_combined_scores_to_csv(
            output_csv,
            scores_scenario,
            scores_excuse,
            average_score_scenario,
            average_score_excuse
        )

        # Plot distributions
        print("Generating score distribution plots...")
        plot_score_distribution(scores_scenario, f"{lang}_scenario")
        plot_score_distribution(scores_excuse, f"{lang}_excuse")

    print("\nEvaluation completed for all specified languages.")

if __name__ == "__main__":
    main()
