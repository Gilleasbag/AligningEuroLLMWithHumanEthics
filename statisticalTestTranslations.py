import pandas as pd
import scipy.stats as stats
import numpy as np
from datetime import datetime

# ========================
# FILE PATH CONFIGURATION
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
# ANALYSIS FUNCTIONS
# ========================
def load_scores(csv_path):
    """Load scores from CSV, skipping header and first row (average if present)"""
    # Read the CSV file
    df = pd.read_csv(csv_path, header=0)
    
    # Drop any rows where all columns are NaN
    df = df.dropna(how='all')
    
    # Reset the index after dropping rows
    df = df.reset_index(drop=True)
    
    # Check and remove any rows that are not part of the data
    # For example, if the first row contains 'average' in the first column, drop it
    if len(df) > 0:
        first_value = df.iloc[0, 0]
        if isinstance(first_value, str) and 'average' in first_value.lower():
            df = df.drop(index=0)
            df = df.reset_index(drop=True)
    
    # Convert all score columns to numeric, errors='coerce' will convert invalid parsing to NaN
    score_columns = df.columns
    df[score_columns] = df[score_columns].apply(pd.to_numeric, errors='coerce')
    
    # Drop any rows with NaN values in all score columns
    df = df.dropna(subset=score_columns, how='all')
    
    return df

def bootstrap_mean_difference(sample1, sample2, n_bootstraps=10000, ci=95):
    """Calculate bootstrap confidence interval for mean difference"""
    mean_diff = np.mean(sample1 - sample2)
    bootstrap_diffs = np.zeros(n_bootstraps)
    n = len(sample1)
    
    for i in range(n_bootstraps):
        indices = np.random.choice(n, n, replace=True)
        bootstrap_diffs[i] = np.mean(sample1.iloc[indices] - sample2.iloc[indices])
    
    alpha = (100 - ci) / 2
    ci_lower, ci_upper = np.percentile(bootstrap_diffs, [alpha, 100 - alpha])
    return mean_diff, ci_lower, ci_upper

def analyze_pair(opus_scores, gpt_scores):
    """Perform statistical analysis on score pairs"""
    assert len(opus_scores) == len(gpt_scores), "Mismatched sample sizes"
    
    mean_diff, ci_lower, ci_upper = bootstrap_mean_difference(opus_scores, gpt_scores)
    t_stat, p_val = stats.ttest_rel(opus_scores, gpt_scores)
    
    return {
        'opus_mean': np.mean(opus_scores),
        'gpt_mean': np.mean(gpt_scores),
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': "{:.2e}".format(p_val),  # Format p-value as scientific notation
        't_statistic': t_stat,
        'sample_size': len(opus_scores),
        'significant': p_val < 0.05
    }

# ========================
# MAIN ANALYSIS PIPELINE
# ========================
def main():
    results = []
    
    for framework in file_paths:
        for lang in ['de', 'fr', 'es']:
            print(f"\nAnalyzing {framework} - {lang.upper()}")
            
            try:
                # Load scores as DataFrames
                opus_df = load_scores(file_paths[framework][lang]['OPUS'])
                gpt_df = load_scores(file_paths[framework][lang]['GPT'])
                
                # Ensure both DataFrames have the same columns
                common_columns = opus_df.columns.intersection(gpt_df.columns)
                if len(common_columns) == 0:
                    raise ValueError("No common COMET score columns found between OPUS and GPT data.")
                
                for score_col in common_columns:
                    opus_scores = opus_df[score_col].reset_index(drop=True)
                    gpt_scores = gpt_df[score_col].reset_index(drop=True)
                    
                    if len(opus_scores) != len(gpt_scores):
                        min_length = min(len(opus_scores), len(gpt_scores))
                        opus_scores = opus_scores.iloc[:min_length]
                        gpt_scores = gpt_scores.iloc[:min_length]
                    
                    analysis = analyze_pair(opus_scores, gpt_scores)
                    
                    results.append({
                        'Ethical Framework': framework,
                        'Language': lang.upper(),
                        'Score Type': score_col,
                        'OPUS Mean': analysis['opus_mean'],
                        'GPT Mean': analysis['gpt_mean'],
                        'Mean Difference': analysis['mean_diff'],
                        'CI Lower': analysis['ci_lower'],
                        'CI Upper': analysis['ci_upper'],
                        'P-Value': analysis['p_value'],  # Already formatted as string
                        'Significance': 'Yes' if analysis['significant'] else 'No',
                        'Sample Size': analysis['sample_size']
                    })
                
            except Exception as e:
                print(f"Error processing {framework}-{lang}: {str(e)}")
                continue
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['Ethical Framework', 'Language', 'Score Type'])
    
    # Format numerical columns
    float_cols = ['OPUS Mean', 'GPT Mean', 'Mean Difference', 'CI Lower', 'CI Upper']
    results_df[float_cols] = results_df[float_cols].round(4)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comet_analysis_results_{timestamp}.csv"
    
    # Save the results to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\nAnalysis complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
