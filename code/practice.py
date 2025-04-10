import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, classification_report)

# Function to calculate metrics and return as a dictionary
def calculate_metrics(confusion):
    y_true = []
    y_pred = []

    num_classes = confusion.shape[0]
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            count = confusion[true_label, pred_label]
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    # Return metrics as a dictionary
    return {
        "Accuracy": accuracy,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1 Score (Macro)": f1_macro,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1 Score (Weighted)": f1_weighted,
        "Cohen's Kappa": kappa
    }

# Function to save metrics to an Excel file
def save_metrics_to_excel(metrics_dict, experiment_name, output_file):
    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics_dict).T
    metrics_df.columns = ["Multi-class", "Binary"]
    metrics_df.insert(0, "Experiment", experiment_name)

    # Save the DataFrame to an Excel file
    metrics_df.to_excel(output_file, sheet_name="Metrics", engine="openpyxl")
    print(f"Metrics have been saved to {output_file}")

# Main script
# Main script
if __name__ == "__main__":
    # Define the experiment name
    experiment_name = "multi-prof-base-20-th_025"

    # Define the confusion matrices
    confusion_4x4 = np.array([
        [4, 17, 2, 1],
        [3, 26, 3, 0],
        [17, 39, 6, 3],
        [0, 5, 1, 2]
    ])

    confusion_2x2 = np.array([
        [6, 18],
        [25, 80]
    ])
# Calculate metrics for both confusion matrices
    metrics_4x4 = calculate_metrics(confusion_4x4)
    metrics_2x2 = calculate_metrics(confusion_2x2)

    # Create a flattened dictionary with all metrics in a single row
    flattened_metrics = {"Experiment": experiment_name}
    
    # Add 4x4 matrix metrics with prefixed column names
    for metric_name, metric_value in metrics_4x4.items():
        flattened_metrics[f"multi-class_{metric_name}"] = metric_value
    
    # Add 2x2 matrix metrics with prefixed column names
    for metric_name, metric_value in metrics_2x2.items():
        flattened_metrics[f"binary{metric_name}"] = metric_value
    
    # Create DataFrame with a single row containing all metrics
    metrics_df = pd.DataFrame([flattened_metrics])
    
    # Define the output file path
    output_file = "/Users/sean/Masters/MSc_2025/code/multi-class/metrics_output.xlsx"
    
    # Check if file exists to determine if we should append
    import os
    if os.path.exists(output_file):
        # Read existing file and append new data
        existing_df = pd.read_excel(output_file)
        updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        updated_df.to_excel(output_file, sheet_name="Metrics", index=False, engine="openpyxl")
    else:
        # Create new file
        metrics_df.to_excel(output_file, sheet_name="Metrics", index=False, engine="openpyxl")
    
    print(f"Metrics have been saved to {output_file}")