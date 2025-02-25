import csv
import os

# List of dataset files
dataset_files = [
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_hard_French.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_hard_German.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_hard_Spanish.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_test_French.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_test_German.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_test_Spanish.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_train_French.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_train_German.csv',
    '/fs/nas/eikthyrnir0/gpeterson/Translations/openAI/Datasets/Splits/Virtue/virtue_train_Spanish.csv'
]

# Process each file
for input_file in dataset_files:
    output_file = os.path.splitext(input_file)[0] + '_modified.csv'  # Create output file name

    # Read the CSV data
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip the header
        data = list(reader)

    # Process the data
    modified_data = []
    for i in range(0, len(data), 5):
        # Get the group of 5 rows
        group = data[i:i+5]
        
        # Check if the group has at least 1 element
        if len(group) > 0:
            # Extract the scenario from the first row
            first_row = group[0]
            scenario = first_row[1].split(' [SEP] ')[0]  # Get the scenario before [SEP]
            trait = first_row[1].split(' [SEP] ')[1] if ' [SEP] ' in first_row[1] else ''  # Get the trait if it exists
            
            # Add the first row with its original label and scenario
            modified_data.append([first_row[0], scenario + (' [SEP] ' + trait if trait else '')])  # First row with original label and scenario
            
            # Process the remaining rows
            for j in range(1, len(group)):
                # Extract the trait from the current row
                current_trait = group[j][1].split(' [SEP] ')[1]  # Get the trait after [SEP]
                modified_data.append([group[j][0], scenario + ' [SEP] ' + current_trait])  # Keep the original label and add the trait

    # Write the modified data to a new CSV file
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header back to the output file
        writer.writerows(modified_data)

    print(f"Modified data has been written to {output_file}.")
