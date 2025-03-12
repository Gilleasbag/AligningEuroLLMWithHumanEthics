import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# ========================
# FILE PATH CONFIGURATION (Both translation methods)
# ========================
file_paths = {
    'Commonsense': {
        'de': {
            'OPUS': "Translations/COMET_MT/OPUS/Commonsense/comet_scores_de_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_de_test.csv"
        },
        'fr': {
            'OPUS': "Translations/COMET_MT/OPUS/Commonsense/comet_scores_fr_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_fr_test.csv"
        },
        'es': {
            'OPUS': "Translations/COMET_MT/OPUS/Commonsense/comet_scores_es_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_es_test.csv"
        }
    },
    'Deontology': {
        'de': {
            'OPUS': "Translations/COMET_MT/OPUS/Deontology/comet_scores_de_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Deontology/comet_scores_de_combined.csv"
        },
        'fr': {
            'OPUS': "Translations/COMET_MT/OPUS/Deontology/comet_scores_fr_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Deontology/comet_scores_fr_combined.csv"
        },
        'es': {
            'OPUS': "Translations/COMET_MT/OPUS/Deontology/comet_scores_es_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Deontology/comet_scores_es_combined.csv"
        }
    },
    'Justice': {
        'de': {
            'OPUS': "Translations/COMET_MT/OPUS/Justice/comet_scores_de_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Justice/comet_scores_de_test.csv"
        },
        'fr': {
            'OPUS': "Translations/COMET_MT/OPUS/Justice/comet_scores_fr_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Justice/comet_scores_fr_test.csv"
        },
        'es': {
            'OPUS': "Translations/COMET_MT/OPUS/Justice/comet_scores_es_test.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Justice/comet_scores_es_test.csv"
        }
    },
    'Utilitarianism': {
        'de': {
            'OPUS': "Translations/COMET_MT/OPUS/Utility/comet_scores_de_utility.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Utility/comet_scores_de_utility.csv"
        },
        'fr': {
            'OPUS': "Translations/COMET_MT/OPUS/Utility/comet_scores_fr_utility.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Utility/comet_scores_fr_utility.csv"
        },
        'es': {
            'OPUS': "Translations/COMET_MT/OPUS/Utility/comet_scores_es_utility.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Utility/comet_scores_es_utility.csv"
        }
    },
    'Virtue': {
        'de': {
            'OPUS': "Translations/COMET_MT/OPUS/Virtue/comet_scores_de_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Virtue/comet_scores_de_combined.csv"
        },
        'fr': {
            'OPUS': "Translations/COMET_MT/OPUS/Virtue/comet_scores_fr_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Virtue/comet_scores_fr_combined.csv"
        },
        'es': {
            'OPUS': "Translations/COMET_MT/OPUS/Virtue/comet_scores_es_combined.csv",
            'GPT': "Translations/COMET_MT/OPENAI/Virtue/comet_scores_es_combined.csv"
        }
    }
}

# ========================
# UTILITY FUNCTION: LOAD MEAN SCORE
# ========================
def load_mean_score(csv_path):
    """
    Load the CSV file, convert all columns to numeric, and compute the overall mean.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(how='all')
    df = df.apply(pd.to_numeric, errors='coerce')
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return np.mean(values)

# ========================
# COLLECT AVERAGE SCORES FOR EACH TRANSLATION METHOD
# ========================
results = []
# Fixed orders.
dataset_order = ['Commonsense', 'Deontology', 'Justice', 'Utilitarianism', 'Virtue']
language_order = ['DE', 'FR', 'ES']
method_order = ['OPUS', 'GPT']
# Color mapping for languages.
color_map = {
    'DE': "#56B4E9",  # German: sky blue
    'FR': "#E69F00",  # French: orange
    'ES': "#009E73"   # Spanish: bluish green
}
# Hatch mapping for translation methods.
hatch_map = {
    'OPUS': "",       # no hatch for OPUS
    'GPT': "//"      # hatch pattern for GPT
}

for dataset in dataset_order:
    for lang in language_order:
        for method in method_order:
            try:
                path = file_paths[dataset][lang.lower()][method]
                mean_val = load_mean_score(path)
                results.append({
                    'Dataset': dataset,
                    'Language': lang,
                    'Method': method,
                    'Avg COMET Score': mean_val
                })
            except Exception as e:
                print(f"Error processing {dataset} - {lang} - {method}: {e}")

df = pd.DataFrame(results)
df['Language'] = pd.Categorical(df['Language'], categories=language_order, ordered=True)
df['Method'] = pd.Categorical(df['Method'], categories=method_order, ordered=True)
df = df.sort_values(by=['Dataset', 'Language', 'Method'])

# ========================
# PLOT GROUPED BAR CHART COMPARING TRANSLATION METHODS (GROUPED BY DATASET)
# ========================
# Each dataset group will contain 6 bars (3 languages × 2 methods). We create a gap between groups.
n_bars_per_dataset = len(language_order) * len(method_order)  # 6 bars per dataset

bar_width = 0.3
group_gap = 0.5  # extra gap between groups
# Group width including gap:
group_width = n_bars_per_dataset * bar_width + group_gap

# Compute group centers for each dataset:
group_centers = []
for i in range(len(dataset_order)):
    center = i * group_width + (n_bars_per_dataset * bar_width) / 2
    group_centers.append(center)

plt.figure(figsize=(15, 8))  # Increased vertical size

# Define ordering within each dataset group.
order_dict = {('DE', 'OPUS'): 0,
              ('DE', 'GPT'): 1,
              ('FR', 'OPUS'): 2,
              ('FR', 'GPT'): 3,
              ('ES', 'OPUS'): 4,
              ('ES', 'GPT'): 5}

# Plot each bar within its dataset group.
for i, dataset in enumerate(dataset_order):
    subset = df[df['Dataset'] == dataset].copy()
    for idx, row in subset.iterrows():
        dataset_idx = dataset_order.index(dataset)
        group_center = group_centers[dataset_idx]
        left_edge = dataset_idx * group_width  # left edge for this dataset group
        offset = order_dict[(row['Language'], row['Method'])] * bar_width + bar_width / 2
        x_pos = left_edge + offset
        plt.bar(x_pos, row['Avg COMET Score'],
                width=bar_width,
                color=color_map[row['Language']],
                hatch=hatch_map[row['Method']])
    # Optionally, draw a vertical dashed line to separate dataset groups.
    if i < len(dataset_order) - 1:
        plt.axvline(x=(i + 1) * group_width - group_gap / 2, color='gray', linestyle='--', alpha=0.5)

# Set x-ticks to the center of each dataset group.
plt.xticks(group_centers, dataset_order, fontsize=16)
plt.xlabel("Dataset", fontsize=18)
plt.ylabel("Average COMET Score", fontsize=18)
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11), fontsize=16)
plt.grid(axis='y', linestyle='--', linewidth=0.7)

# Create legends:
# 1. Legend for language (color)
language_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[lang]) for lang in language_order]
language_labels = [{"DE": "German", "FR": "French", "ES": "Spanish"}[lang] for lang in language_order]
# 2. Legend for translation method (hatch)
method_handles = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', hatch=hatch_map[method]) for method in method_order]
method_labels = ["OPUS", "GPT"]

first_legend = plt.legend(language_handles, language_labels, title="Language (Color)", loc='upper left', fontsize=14)
second_legend = plt.legend(method_handles, method_labels, title="Translation Method (Hatch)", loc='upper right', fontsize=14)
plt.gca().add_artist(first_legend)

plt.tight_layout()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_png = f"comet_scores_comparison_grouped_{timestamp}.png"
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()

print(f"Bar chart saved to {output_png}")
