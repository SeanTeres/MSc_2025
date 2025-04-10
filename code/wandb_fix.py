import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import io
from PIL import Image

# Initialize the API
api = wandb.Api()

# Make sure these values are correct
username = "seanteres-wits-university"  # Your W&B username
project_name = "MBOD-New"  # Your actual project name
run_id = "73k1r8bh"  # The specific run ID to update

# Construct the full run path
run_path = f"{username}/{project_name}/{run_id}"
print(f"Attempting to access run at: {run_path}")

try:
    # Get the completed run
    run = api.run(run_path)
    print(f"Successfully connected to run: {run.name}")
    
    # Recreate the confusion matrices from your test results
    cm_binary_d2 = np.array([[ 2, 22], [14, 91]])  # Your binary confusion matrix 
    cm_multi_d2 = np.array([[ 0, 20,  4,  0],
                          [ 0, 20, 12,  0],
                          [ 4, 29, 29,  3],
                          [ 1,  1,  4,  2]])  # Your multi-class confusion 

    # Initialize a new wandb run in resume mode
    print(f"Attempting to resume run with ID: {run_id}")
    with wandb.init(project=project_name, id=run_id, resume="allow") as resumed_run:
        print(f"Successfully resumed run: {resumed_run.name}")
        
        # 1. Create binary confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_binary_d2, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Abnormal'],
                   yticklabels=['Normal', 'Abnormal'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Binary Classification Confusion Matrix')
        # Log directly to wandb
        resumed_run.log({"Binary Confusion Matrix": wandb.Image(plt)})
        plt.close()
        
        # 2. Create multi-class confusion matrix
        class_names = ['Normal (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)']
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_multi_d2, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Multi-class Classification Confusion Matrix')
        # Log directly to wandb
        resumed_run.log({"Multi-class Confusion Matrix": wandb.Image(plt)})
        plt.close()
        
        # 3. Create normalized multi-class confusion matrix
        plt.figure(figsize=(10, 8))
        cm_norm = cm_multi_d2.astype('float') / cm_multi_d2.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Multi-class Confusion Matrix')
        # Log directly to wandb
        resumed_run.log({"Normalized Multi-class Confusion Matrix": wandb.Image(plt)})
        plt.close()
        
        print("Successfully logged all confusion matrices")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nTroubleshooting suggestions:")
    print("1. Verify your W&B API key is correctly set up")
    print("2. Check that the username, project name, and run ID are correct")
    print("3. Ensure you have permission to modify this run")
    print("4. Try running 'wandb login' in your terminal before running this script")