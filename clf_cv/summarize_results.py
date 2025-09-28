import pandas as pd
import json
import numpy as np
import ast

# Load the CSV file
df = pd.read_csv('project.csv')

# Parse the config column (it contains Python dict strings, not JSON)
def parse_dict_string(dict_str):
    try:
        # Use ast.literal_eval to safely evaluate the string as a Python literal
        return ast.literal_eval(dict_str)
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing string: {e}")
        print(f"Problematic string: {dict_str[:100]}...")
        return {}

# Parse summary and config columns
df['config_dict'] = df['config'].apply(parse_dict_string)
df['summary_dict'] = df['summary'].apply(parse_dict_string)

# Extract metrics from all runs
results = []
for idx, row in df.iterrows():
    config = row['config_dict']
    summary = row['summary_dict']
    
    # Extract run info
    run_name = config.get('RUN_NAME', 'Unknown')
    fold = config.get('fold', 'Unknown')
    
    # Extract test metrics
    test_metrics = {
        'run_name': run_name,
        'fold': fold,
        'test_acc': summary.get('test/acc_opt', np.nan),
        'test_auc': summary.get('test/auc', np.nan),
        'test_sens': summary.get('test/sens_opt', np.nan),
        'test_spec': summary.get('test/spec_opt', np.nan),
        'test_f1': summary.get('test/f1_opt', np.nan),
        'test_kappa': summary.get('test/kappa_opt', np.nan)
    }
    
    results.append(test_metrics)

# Create a DataFrame with results
results_df = pd.DataFrame(results)

# Group by run_name and calculate mean and std for each metric
grouped = results_df.groupby('run_name').agg(
    mean_acc=('test_acc', 'mean'),
    std_acc=('test_acc', 'std'),
    mean_auc=('test_auc', 'mean'),
    std_auc=('test_auc', 'std'),
    mean_sens=('test_sens', 'mean'),
    std_sens=('test_sens', 'std'),
    mean_spec=('test_spec', 'mean'),
    std_spec=('test_spec', 'std'),
    mean_f1=('test_f1', 'mean'),
    std_f1=('test_f1', 'std'),
    mean_kappa=('test_kappa', 'mean'),
    std_kappa=('test_kappa', 'std'),
    count=('fold', 'count')
)

# Format the results nicely
for metric in ['acc', 'auc', 'sens', 'spec', 'f1', 'kappa']:
    grouped[f'{metric}'] = grouped[f'mean_{metric}'].map('{:.4f}'.format) + " ± " + grouped[f'std_{metric}'].map('{:.4f}'.format)

print("\nSummarized Results by Model:")
print(grouped[['acc', 'auc', 'sens', 'spec', 'f1', 'kappa', 'count']])

# Calculate 95% confidence intervals
alpha = 0.05  # 95% confidence
grouped_with_ci = grouped.copy()
for metric in ['acc', 'auc', 'sens', 'spec', 'f1', 'kappa']:
    for run_name in grouped.index:
        n = grouped.loc[run_name, 'count']
        mean = grouped.loc[run_name, f'mean_{metric}']
        std = grouped.loc[run_name, f'std_{metric}']
        
        # Calculate confidence interval
        from scipy import stats
        ci = stats.t.ppf(1-alpha/2, n-1) * (std / np.sqrt(n))
        grouped_with_ci.loc[run_name, f'{metric}_ci_lower'] = mean - ci
        grouped_with_ci.loc[run_name, f'{metric}_ci_upper'] = mean + ci
        grouped_with_ci.loc[run_name, f'{metric}_with_ci'] = f"{mean:.4f} [{(mean-ci):.4f}, {(mean+ci):.4f}]"

# Print results with confidence intervals
print("\nResults with 95% Confidence Intervals:")
for metric in ['acc', 'auc', 'sens', 'spec', 'f1', 'kappa']:
    print(f"\n{metric.upper()} with 95% CI:")
    for run_name in grouped_with_ci.index:
        print(f"{run_name}: {grouped_with_ci.loc[run_name, f'{metric}_with_ci']}")

# Optionally, save results to CSV
grouped[['acc', 'auc', 'sens', 'spec', 'f1', 'kappa', 'count']].to_csv('summarized_results.csv')
grouped_with_ci[[col for col in grouped_with_ci.columns if 'with_ci' in col]].to_csv('results_with_confidence_intervals.csv')

# Print individual fold results for each model
print("\nDetailed Results by Fold:")
print(results_df)
results_df.to_csv('detailed_results.csv', index=False)