#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

def benchmark():
    # Updated data dictionary with averaged values
    data = {
        "Model": ["EuroLLM 1.7B", "EuroLLM 9B", "GPT-4o-mini"],
        "Justice": [50.35, 52.85, 81.6],
        "Deontology": [49.9, 59.3, 78.95],
        "Virtue": [33.65, 70.45, 93.5],
        "Utilitarianism": [45.35, 46.95, 67.85],
        "Commonsense": [52.65, 56.85, 75.25],
        "Average": [46.3, 57.3, 79.45]  # Can be excluded if not needed in plotting
    }
    
    # Model names and ethical theory groups (excluding "Average")
    models = data["Model"]
    groups = ["Commonsense", "Deontology", "Justice", "Virtue", "Utilitarianism"]
    
    # Extracting scores per group
    score_data = {group: data[group] for group in groups}
    
    # Define color map for each model
    color_map = {
        "EuroLLM 1.7B": "#56B4E9",  # Light Blue
        "EuroLLM 9B": "#009E73",    # Green
        "GPT-4o-mini": "#D55E00",   # Orange
    }
    
    # Number of groups and models
    n_groups = len(groups)
    n_models = len(models)
    
    # Set positions for the groups on the x-axis
    x = np.arange(n_groups)
    
    # Width for each bar
    bar_width = 0.2
    total_group_width = n_models * bar_width  # Total width occupied per group
    
    # Create offsets to spread bars within each group
    offsets = np.linspace(-total_group_width / 2 + bar_width / 2, 
                          total_group_width / 2 - bar_width / 2, 
                          n_models)
    
    plt.figure(figsize=(12, 8))
    
    # Plot bars for each model
    for m_idx, model in enumerate(models):
        model_scores = [score_data[group][m_idx] for group in groups]
        plt.bar(x + offsets[m_idx], model_scores, width=bar_width, color=color_map[model], label=model)
    
    # Set x-axis ticks and labels
    plt.xticks(x, groups, fontsize=14)
    plt.yticks(fontsize=15)  # Set y-axis tick labels font size
    
    # Set axis labels and title
    plt.ylabel('Accuracy (%)', fontsize=16)
    
    # Add legend
    plt.legend(fontsize=15, ncol=1, loc='upper left')
    
    # Add grid for clarity
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("big_bar_chart.png", dpi=300)
    plt.show()


def postFT():
    # ----------------------------------------------------------------------
    # 1. Read Data
    # ----------------------------------------------------------------------
    df = pd.read_csv("postFTresults.csv")

    # ----------------------------------------------------------------------
    # 2. Prep: Combine (Split + Phase) for Raw Scores
    # ----------------------------------------------------------------------
    df_melt = df.melt(
        id_vars=['Dataset', 'Split', 'Language'],
        value_vars=['Pre FT', 'Post FT'],
        var_name='Phase',
        value_name='Accuracy'
    )
    df_melt["Split_Phase"] = df_melt["Split"] + "-" + df_melt["Phase"]

    # Optionally enforce a particular order of categories on the x-axis:
    categories = ["Test-Pre FT", "Test-Post FT", "Hard-Pre FT", "Hard-Post FT"]
    df_melt["Split_Phase"] = pd.Categorical(df_melt["Split_Phase"], categories=categories)

    # List all unique datasets
    datasets = sorted(df_melt["Dataset"].unique())
    n_plots = len(datasets)
    print(f"Number of datasets = {n_plots}")

    # ----------------------------------------------------------------------
    # 3. Set up the Okabe–Ito Palette
    # ----------------------------------------------------------------------
    okabe_ito = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]
    sns.set_theme(style="whitegrid", font_scale=1.2)  # Increased font scale
    sns.set_palette(okabe_ito)

    # ----------------------------------------------------------------------
    # 4. Raw Scores in a 3+2 Centered Layout
    # ----------------------------------------------------------------------
    if n_plots != 5:
        print("WARNING: Code assumes exactly 5 datasets for the 3+2 layout.")

    # Create main figure for raw scores with larger figure size
    fig1 = plt.figure(figsize=(18, 10))  # Increased figure size

    # We'll use 2 rows × 6 columns => 12 slots.
    positions_raw = [(1,2), (3,4), (5,6), (8,9), (10,11)]
    axes_raw = []

    for i, ds in enumerate(datasets):
        ax = fig1.add_subplot(2, 6, positions_raw[i])
        axes_raw.append(ax)

        # Filter data for this dataset
        sub = df_melt[df_melt["Dataset"] == ds]

        # Plot
        sns.barplot(
            data=sub,
            x="Split_Phase",
            y="Accuracy",
            hue="Language",
            ax=ax
        )
        ax.set_xlabel(" ", fontsize=16)  # Increased font size
        ax.set_ylabel("Accuracy (%)", fontsize=16)  # Increased font size
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=14)  # Larger tick labels
        ax.set_yticklabels(ax.get_yticks(), fontsize=14)  # Larger tick labels
        
        ax.set_title(ds, fontsize=18)  # Larger title

        # Dashed vertical line between categories index=1 and 2
        ax.axvline(
            x=1.5,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            zorder=3
        )

        # Remove subplot-level legend
        ax.get_legend().remove()

    # Make a single, figure-level legend in bottom-left
    handles_raw, labels_raw = axes_raw[0].get_legend_handles_labels()
    fig1.legend(
        handles_raw,
        labels_raw,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),  # adjust if needed
        fontsize=20  # Larger legend font size
    )

    fig1.tight_layout()  # Increased padding between subplots
    fig1.savefig("raw_scores_combined.png", dpi=300, bbox_inches="tight")

    # ----------------------------------------------------------------------
    # 5. Plot DIFFERENCE (Post FT - Pre FT) in 3+2 Centered Layout
    # ----------------------------------------------------------------------
    df["Difference"] = df["Post FT"] - df["Pre FT"]
    unique_datasets = sorted(df["Dataset"].unique())  # should match 'datasets'

    fig2 = plt.figure(figsize=(18, 10))  # Increased figure size
    axes_diff = []
    positions_diff = [(1,2), (3,4), (5,6), (8,9), (10,11)]

    for i, ds in enumerate(unique_datasets):
        ax = fig2.add_subplot(2, 6, positions_diff[i])
        axes_diff.append(ax)

        # Filter data
        sub = df[df["Dataset"] == ds]

        # Plotting the difference
        sns.barplot(
            data=sub,
            x="Split",
            y="Difference",
            hue="Language",
            ax=ax
        )

        ax.set_xlabel(" ", fontsize=16)  # Increased font size
        ax.set_ylabel("% Difference", fontsize=16)  # Increased font size

        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)  # Larger tick labels
        ax.set_yticklabels(ax.get_yticks(), fontsize=14)  # Larger tick labels    
        ax.set_title(ds, fontsize=18)  # Larger title

        # (Optional) dashed line for 2 categories "Test" vs. "Hard"
        ax.axvline(
            x=0.5,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            zorder=3
        )

        # Remove subplot-level legend
        ax.get_legend().remove()

    # Single, figure-level legend in bottom-left
    handles_diff, labels_diff = axes_diff[0].get_legend_handles_labels()
    fig2.legend(
        handles_diff,
        labels_diff,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        fontsize=20  # Larger legend font size
    )

    plt.tight_layout()  # Increased padding between subplots
    fig2.savefig("difference_scores_centered.png", dpi=300, bbox_inches="tight")

    # ----------------------------------------------------------------------
    # 6. Show Plots
    # ----------------------------------------------------------------------
    plt.show()
    
def failures():
    # Define the data
    data = {
        'Ethical Framework': [
            'Commonsense Test', 'Commonsense Hard',
            'Deontology Test', 'Deontology Hard',
            'Justice Test', 'Justice Hard',
            'Virtue Test', 'Virtue Hard',
            'Utility Test', 'Utility Hard'
        ],
        '1.7B': [4.22, 5.22, 0.95, 0.62, 7.51, 7.16, 13.07, 12.13, 10.55, 9.67],
        '9B Quantised': [1.47, 1.64, 0.64, 0.45, 0.85, 0.73, 4.52, 4.29, 28.94, 30.34],
        'GPT-4o-mini': [0.05, 0.03, 0.00, 0.00, 0.00, 0.00, 0.20, 0.84, 0.00, 0.00]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Extract ethical frameworks and test types for hierarchical labeling
    frameworks = []
    test_types = []
    for item in df['Ethical Framework']:
        parts = item.split(' ')
        frameworks.append(' '.join(parts[:-1]))  # Everything except the last word
        test_types.append(parts[-1])  # Last word (Test or Hard)

    # Add these as new columns
    df['Framework'] = frameworks
    df['Test Type'] = test_types

    # Set color scheme with higher contrast
    colors = {
        '1.7B': '#56B4E9',         # lighter blue
        '9B Quantised': '#009E73', # green
        'GPT-4o-mini': '#D55E00'   # orange
    }
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get unique frameworks and models
    unique_frameworks = df['Framework'].unique()
    models = ['1.7B', '9B Quantised', 'GPT-4o-mini']
    
    # Parameters for bar spacing
    n_frameworks = len(unique_frameworks)
    n_test_types = 2  # Test and Hard
    n_models = len(models)
    group_width = 0.8  # Width allocated for each framework group
    bar_width = group_width / (n_test_types * n_models + n_test_types - 1)  # Width of individual bars with spacing
    spacing = bar_width * 0.2  # Space between bars within a test type
    
    # Calculate positions for x-ticks (framework centers)
    framework_centers = np.arange(n_frameworks)
    
    # Plot the bars
    bar_positions = []
    for f_idx, framework in enumerate(unique_frameworks):
        framework_data = df[df['Framework'] == framework]
        
        # Process each test type (Test, Hard)
        for t_idx, test_type in enumerate(['Test', 'Hard']):
            test_data = framework_data[framework_data['Test Type'] == test_type]
            if len(test_data) == 0:
                continue
                
            # Calculate the starting position for this test group
            group_start = framework_centers[f_idx] - group_width/2 + t_idx * (n_models * (bar_width + spacing) + spacing)
            
            # Plot each model's bar
            for m_idx, model in enumerate(models):
                bar_pos = group_start + m_idx * (bar_width + spacing)
                bar_positions.append(bar_pos)
                value = test_data[model].values[0]
                bar = ax.bar(bar_pos, value, bar_width, color=colors[model], 
                             label=model if (f_idx == 0 and t_idx == 0 and m_idx < len(models)) else "")
                
                # Add value labels above bars
                ax.text(bar_pos, value + 0.5, f'{value:.2f}', 
                        ha='center', va='bottom', fontsize=9, rotation=45)
    
    # Set up the hierarchical x-axis labels
    framework_positions = []
    for f_idx in range(n_frameworks):
        framework_positions.append(framework_centers[f_idx])
    
    # Main x-axis (frameworks)
    ax.set_xticks(framework_positions)
    ax.set_xticklabels(unique_frameworks, fontsize=14)
    
    # Secondary x-axis for test types
    ax2 = ax.twiny()
    ax2.spines['bottom'].set_position(('outward', 40))  # Move the second x-axis down
    
    # Calculate positions for test type labels
    test_type_positions = []
    test_type_labels = []
    
    for f_idx in range(len(unique_frameworks)):
        base_pos = framework_centers[f_idx]
        
        # Position for "Test" label
        test_pos = base_pos - group_width/4
        test_type_positions.append(test_pos)
        test_type_labels.append("Test")
        
        # Position for "Hard" label
        hard_pos = base_pos + group_width/4
        test_type_positions.append(hard_pos)
        test_type_labels.append("Hard")
    
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(test_type_positions)
    ax2.set_xticklabels(test_type_labels, fontsize=12)
    
    # Customize the plot
    ax.set_xlabel('Ethical Framework', fontsize=16)
    ax.set_ylabel('Failure Rate (%)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    
    # Create a legend with larger text
    legend = ax.legend(fontsize=16, loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("failure_rates.png", dpi=300, bbox_inches="tight")
    
    # Show the plot
    plt.show()
    

def create_difference_bar_chart(csv_path="failures.csv", save_path=None):
    """
    Create 5 separate bar charts showing the difference between Pre and Post values,
    split by unique categories (assumed to be 5 distinct categories) from the CSV file.
    The individual plots are arranged in a custom 2×6 grid layout so that they appear nicely centered.
    
    Parameters:
    -----------
    csv_path : str
        Path to the failures.csv file.
    save_path : str, optional
        Path to save the resulting figure.
    """
    # Read the CSV file and fix the data types
    df = pd.read_csv(csv_path)
    
    # Convert percentage strings to float values
    df['Pre'] = df['Pre'].str.rstrip('%').astype(float)
    df['Post'] = df['Post'].str.rstrip('%').astype(float)
    
    # Calculate the difference (Post - Pre)
    df['Difference'] = df['Post'] - df['Pre']
    
    # Create a combined column for better x-axis labeling
    # (e.g. "Category (Difficulty)"; adjust if needed)
    df['Category_Difficulty'] = df['Difficulty']
    
    # Define the Okabe–Ito color palette (as in postFT)
    okabe_ito = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]
    
    # Apply the theme settings
    sns.set_theme(style="whitegrid", font_scale=1.2)
    sns.set_palette(okabe_ito)
    
    # Assume we want to split the data by Category:
    # (Change this grouping if needed)
    unique_categories = sorted(df['Category'].unique())
    if len(unique_categories) != 5:
        raise ValueError("Expected exactly 5 unique categories in the data.")
    
    # Create a figure and define the positions (hack) for a centers layout.
    # We use a grid with 2 rows and 6 columns and fill only 5 specific positions.
    fig = plt.figure(figsize=(18, 10))
    axes_diff = []
    positions = [(1, 2), (3, 4), (5, 6), (8, 9), (10, 11)]
    
    for i, cat in enumerate(unique_categories):
        # Create an axis at the specified grid location
        ax = fig.add_subplot(2, 6, positions[i])
        axes_diff.append(ax)
        
        # Filter data for the current category
        sub = df[df['Category'] == cat]
        
        # Create the grouped bar plot.
        # Using x='Category_Difficulty' allows you to see details of difficulty,
        # while hue='Language' shows comparisons between languages.
        sns.barplot(
            data=sub,
            x='Category_Difficulty',
            y='Difference',
            hue='Language',
            ax=ax
        )
        
        # Customize the axis
        ax.set_xlabel(" ", fontsize=18)
        ax.set_ylabel("% Difference", fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_title(cat, fontsize=20)
        
        # Optional horizontal line at zero for better reference
        ax.axvline(
            x=0.5,
            color='black',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            zorder=3
        )

        
        # Remove subplot-level legend so we can use a consolidated figure-level legend
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    
    # Create a single, figure-level legend using handles from one of the axes
    handles, labels = axes_diff[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower left',
        bbox_to_anchor=(0.01, 0.01),
        fontsize=20,
        title='Language',
        title_fontsize=22
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    


def main():
    benchmark()
    # postFT()
    #failures()
    #create_difference_bar_chart(csv_path="failures.csv", save_path="difference_bar_chart.png")
    
if __name__ == "__main__":
    main()