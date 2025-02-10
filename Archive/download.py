from datasets import load_dataset, concatenate_datasets

# List of datasets to download
datasets = ['commonsense', 'justice', 'virtue', 'deontology', 'utilitarianism']

# Loop through each dataset
for dataset_name in datasets:
    # Load the dataset
    dataset = load_dataset('hendrycks/ethics', dataset_name)
    
    # Get the train, validation, and test splits
    train = dataset['train']
    val = dataset['validation'] #test
    test = dataset['test'] # hard
    
    # Save the training set
    train_df = train.to_pandas()
    train_df.to_csv(f'{dataset_name}_train.csv', index=False)

    # Save the validation set
    val_df = val.to_pandas()
    val_df.to_csv(f'{dataset_name}_test.csv', index=False)

    # Save the test set
    test_df = test.to_pandas()
    test_df.to_csv(f'{dataset_name}_hard.csv', index=False)
    
    # Concatenate the splits
    dataset = load_dataset('hendrycks/ethics', dataset_name, split='validation+test+train')

    # Save the concatenated dataset
    dataset_df = dataset.to_pandas()
    dataset_df.to_csv(f'{dataset_name}.csv', index=False)
    

print("Datasets have been saved locally as CSV files.")
