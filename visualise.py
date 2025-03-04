#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
    "EuroLLM 1.7B": "royalblue",
    "EuroLLM 9B": "limegreen",
    "GPT-4o-mini": "crimson"
}

color_map_hard = {
    "EuroLLM 1.7B": "navy",
    "EuroLLM 9B": "forestgreen",
    "GPT-4o-mini": "firebrick"
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
plt.xticks(x, groups, fontsize=12)

# Set axis labels and title.
plt.xlabel('Ethical Theory', fontsize=14)
plt.ylabel('Score', fontsize=14)
plt.title('Comparison of Models across Ethical Theories and Test Sets', fontsize=16)

# Add legend.
plt.legend(fontsize=10, ncol=2, loc='upper left')

# Add grid for clarity.
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("big_bar_chart.png", dpi=300)
plt.show()
