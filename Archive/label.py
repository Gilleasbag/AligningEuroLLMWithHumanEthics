import csv
from datasets import load_dataset, concatenate_datasets

def load_and_prepare_data():
    """Load the ETHICS dataset."""
    val = load_dataset('hendrycks/ethics', 'commonsense', split='validation')
    train = load_dataset('hendrycks/ethics', 'commonsense', split='train')
    test = load_dataset('hendrycks/ethics', 'commonsense', split='test')
    
    # Concatenate all the splits into a single dataset
    dataset = concatenate_datasets([train, val, test])
    return dataset

def load_translations_add_labels(translations_file='translation_results_opus_mt_en_fr_20250110-202543.csv'):
    """Load translations from a CSV, replace the original statement with labels, and save to a new CSV."""
    dataset = load_and_prepare_data()

    # Open a new file to save translations with labels
    new_filename = 'translations_with_labels.csv'
    with open(new_filename, mode='w', newline='', encoding='utf-8-sig') as new_file:
        writer = csv.writer(new_file)
        writer.writerow(['label', 'input'])

        # Open the original translation file
        with open(translations_file, mode='r', newline='', encoding='utf-8-sig') as old_file:
            reader = csv.reader(old_file)
            
            # Skip through the first three rows and then handle the header on the fourth row
            for _ in range(2):  # Skipping actual summary and header beyond data row
                next(reader)
            header = next(reader)  # This should be ['Original Statement', 'Translated Statement']
            
            # Process the actual data starting from row 4
            for idx, row in enumerate(reader):
                if idx < len(dataset):  # Ensure we do not go out of bounds
                    label = dataset[idx]['label']  # Get label from the dataset
                    translated_statement = row[1]  # Second column, translation
                    writer.writerow([label, translated_statement])

    print(f"Translations with labels have been saved to {new_filename}")

# Example usage:
if __name__ == "__main__":
    load_translations_add_labels()
