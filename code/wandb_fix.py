import wandb


wandb.login(key = '176da722bd80e35dbc4a8cea0567d495b7307688')

api = wandb.Api()

# Replace these with your project details
ENTITY = "seanteres-wits-university"
PROJECT = "MBOD-New"

# Get all runs from the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:
    run.finish()

