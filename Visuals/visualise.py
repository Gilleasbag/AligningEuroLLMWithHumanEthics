#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

def benchmark():
            # Data dictionary provided
    data = {
        "Model": ["EuroLLM 1.7B", "EuroLLM 9B", "GPT-4o-mini"],
        "Justice (Test)": [49.8, 53.0, 83.8],
        "Justice (Hard Test)": [50.9, 52.7, 79.4],
        "Deontology (Test)": [50.9, 60.8, 80.0],
        "Deontology (Hard Test)": [48.9, 57.8, 77.9],
        "Virtue (Test)": [37.7, 76.0, 95.3],
        "Virtue (Hard Test)": [29.6, 64.9, 91.7],
        "Utilitarianism (Test)": [45.7, 48.5, 73.2],
        "Utilitarianism (Hard Test)": [45.0, 45.4, 62.5],
        "Commonsense (Test)": [53.3, 57.2, 78.0],
        "Commonsense (Hard Test)": [52.0, 56.5, 72.5],
        "Average (Test)": [47.3, 59.1, 82.1],
        "Average (Hard Test)": [45.3, 55.5, 76.8]
    }

    # Model names and ethical theory groups (Average removed)
    models = data["Model"]
    groups = ["Justice", "Deontology", "Virtue", "Utilitarianism", "Commonsense"]

    # Organize the data, grouping Test and Hard Test scores per ethical theory.
    score_data = {}
    for group in groups:
        test_key = f"{group} (Test)"
        hard_key = f"{group} (Hard Test)"
        score_data[group] = {
            "Test": data[test_key],
            "Hard Test": data[hard_key]
        }

    # Define separate color maps for Test and Hard Test scores for each model.
    color_map_test = {
        "EuroLLM 1.7B": "#56B4E9",  # lighter blue
        "EuroLLM 9B":   "#009E73",  # green
        "GPT-4o-mini":  "#D55E00",  # orange
    }
    color_map_hard = {
        "EuroLLM 1.7B": "#0072B2",  # darker blue
        "EuroLLM 9B":   "#00684A",  # darker green
        "GPT-4o-mini":  "#8B3900",  # darker orange
    }

    # Number of groups and number of bars per group:
    n_groups = len(groups)
    bars_per_group = 6  # 2 bars per model (Test and Hard Test) for 3 models.

    # Set positions for the groups on the x-axis.
    x = np.arange(n_groups)

    # Width for each bar (adjustable)
    bar_width = 0.13

    # Total width for each group is the cumulative width of 6 bars.
    total_group_width = bars_per_group * bar_width

    # Create offsets for each bar so that they are centered on the group tick.
    offsets = np.linspace(-total_group_width / 2 + bar_width / 2, 
                        total_group_width / 2 - bar_width / 2, 
                        bars_per_group)

    plt.figure(figsize=(12, 8))

    # Used to track labels already added to the legend.
    used_labels = set()

    # Loop over each ethical theory (group) and plot bars for every model.
    for i, group in enumerate(groups):
        test_scores = score_data[group]["Test"]
        hard_scores = score_data[group]["Hard Test"]
        
        # For each model, plot two bars (Test and Hard Test).
        for m_idx, model in enumerate(models):
            # Compute bar indices:
            # index 0: first model Test, 1: first model Hard Test,
            # index 2: second model Test, 3: second model Hard Test,
            # index 4: third model Test, 5: third model Hard Test.
            bar_index_test = m_idx * 2
            bar_index_hard = m_idx * 2 + 1
            
            # Compute positions by adding the offset.
            pos_test = x[i] + offsets[bar_index_test]
            pos_hard = x[i] + offsets[bar_index_hard]
            
            # Plot the Test bar for this model.
            label_test = f"{model} Test"
            if label_test not in used_labels:
                plt.bar(pos_test, test_scores[m_idx], width=bar_width, 
                        color=color_map_test[model], label=label_test)
                used_labels.add(label_test)
            else:
                plt.bar(pos_test, test_scores[m_idx], width=bar_width, 
                        color=color_map_test[model])
            
            # Plot the Hard Test bar using a different color.
            label_hard = f"{model} Hard Test"
            if label_hard not in used_labels:
                plt.bar(pos_hard, hard_scores[m_idx], width=bar_width, 
                        color=color_map_hard[model], label=label_hard)
                used_labels.add(label_hard)
            else:
                plt.bar(pos_hard, hard_scores[m_idx], width=bar_width, 
                        color=color_map_hard[model])

    # Set the x-axis ticks and labels.
    plt.xticks(x, groups, fontsize=14)
    #set y ticks labels to be fontsize 15
    plt.yticks(fontsize=15)

    # Set axis labels and title.
    plt.ylabel('Accuracy(%)', fontsize=16)

    # Add legend.
    plt.legend(fontsize=13, ncol=2, loc='upper left')

    # Add grid for clarity.
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

def main():
    benchmark()
    postFT()
    
if __name__ == "__main__":
    main()