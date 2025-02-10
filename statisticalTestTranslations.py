import pandas as pd
import scipy.stats as stats

# File paths
csv_path_OPUS_commonsense_de = "Translations/COMET_MT/OPUS/Commonsense/comet_scores_de_test.csv"
csv_path_GPT_commonsense_de = "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_de_test.csv"
csv_path_OPUS_commonsense_fr = "Translations/COMET_MT/OPUS/Commonsense/comet_scores_fr_test.csv"
csv_path_GPT_commonsense_fr = "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_fr_test.csv"
csv_path_OPUS_commonsense_es = "Translations/COMET_MT/OPUS/Commonsense/comet_scores_es_test.csv"
csv_path_GPT_commonsense_es = "Translations/COMET_MT/OPENAI/Commonsense/openai_comet_scores_es_test.csv"

def load_scores(csv_path):
    # Load CSV without headers by specifying header=None
    # The scores will then be in the first column, accessible by df[0]
    df = pd.read_csv(csv_path, header=None, skiprows=2)  # Skipping first two rows if they are not needed

    return df[0].values.tolist()  # Updated to use integer index for the first column



def perform_paired_t_test(scores_model_1, scores_model_2):
    t_stat, p_val = stats.ttest_rel(scores_model_1, scores_model_2)
    print("T-Statistic:", t_stat)
    print("P-Value:", p_val)
    print("Results:", "Significant" if p_val < 0.05 else "Not significant")
    print("Model 1 mean score: ", sum(scores_model_1)/len(scores_model_1))
    print("Model 2 mean score: ", sum(scores_model_2)/len(scores_model_2))

# Load scores from CSV files
scores_OPUS_commonsense_de = load_scores(csv_path_OPUS_commonsense_de)
scores_GPT_commonsense_de = load_scores(csv_path_GPT_commonsense_de)
scores_OPUS_commonsense_fr = load_scores(csv_path_OPUS_commonsense_fr)
scores_GPT_commonsense_fr = load_scores(csv_path_GPT_commonsense_fr)
scores_OPUS_commonsense_es = load_scores(csv_path_OPUS_commonsense_es)
scores_GPT_commonsense_es = load_scores(csv_path_GPT_commonsense_es)

# Perform paired t-tests
print("DE Test:")
perform_paired_t_test(scores_OPUS_commonsense_de, scores_GPT_commonsense_de)
print("\nFR Test:")
perform_paired_t_test(scores_OPUS_commonsense_fr, scores_GPT_commonsense_fr)
print("\nES Test:")
perform_paired_t_test(scores_OPUS_commonsense_es, scores_GPT_commonsense_es)
