import csv
import os

def get_english_counts(base_path, category):
    """ Reads the English dataset files based on category and returns the counts for the splits. """
    counts = {}
    splits = ['hard', 'test', 'train']
    for split_name in splits:
        file_path = os.path.join(base_path, f"{category}_{split_name}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"English dataset file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader, None)  # Skip header
            counts[split_name] = sum(1 for _ in reader)
    return counts

def slice_translated_data(file_path, counts):
    """ Slice the provided translated file using the English dataset counts and order. """
    datasets = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read header once
        for split_name in ['test', 'hard', 'train']:
            data = [header]
            for _ in range(counts[split_name]):
                try:
                    data.append(next(reader))
                except StopIteration:
                    print(f"Warning: Reached end of file before expected for {split_name} in {file_path}")
                    break
            datasets[split_name] = data
    return datasets

def save_datasets(datasets, output_folder, category, language, date):
    """ Save each dataset split into separate files based on category and language. """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for split_name, dataset in datasets.items():
        filename = f"{category}_{split_name}_{language}_{date}.csv"
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(dataset)
        print(f"Saved {split_name} dataset for {category} in {language} to {filepath}")

def process_translations(base_path, output_folder, file_path):
    """ Process a single translated file based on the counts from the English files. """
    base_name = os.path.basename(file_path)
    category, _, language, raw_datetime = base_name.rstrip('.csv').split('_')
    date = raw_datetime.split('-')[0]

    print(f"Processing category: '{category}', language: '{language}', date: '{date}'")

    # Get counts from English datasets
    english_counts = get_english_counts(base_path, category)

    # Process translated file
    translated_datasets = slice_translated_data(file_path, english_counts)

    # Save the datasets
    save_datasets(translated_datasets, output_folder, category, language, date)

def main():
    base_path = 'English_Datasets'
    output_folder = 'Translated_Splits'
    translations = [
        'Translations/openAI/Code/justice_translations_Spanish_20250207-013445.csv',
        'Translations/openAI/Code/justice_translations_German_20250207-181451.csv',
        'Translations/openAI/Code/justice_translations_French_20250207-095623.csv'
    ]
    
    for file_path in translations:
        print(f"Processing translation: {file_path}")
        try:
            process_translations(base_path, output_folder, file_path)
            print("Completed processing.\n")
        except Exception as e:
            print(f"Error processing {file_path}: {e}\n")

if __name__ == '__main__':
    main()
