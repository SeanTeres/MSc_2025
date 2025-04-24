import wandb

sweep_config = {
    'method': 'grid',  # You can also use 'random' or 'bayes'
    'name': 'cl-random-margin-sweep',
    'metric': {
        'name': 'val_map',  # Optimize validation mAP
        'goal': 'maximize'  # We want to maximize this metric
    },
    'parameters': {
        'initial_margin': {
            'values': [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]  # Different margin values to try
        },
        'batch_size': {'value': 16},
        'n_epochs': {'value': 200},
        'learning_rate': {'value': 1e-4},
        'mining': {'value': 'Random'},
        'augmentations': {'value': False},
        'margin_scheduling': {'value': False}  # Keep margin scheduling disabled
    }
}