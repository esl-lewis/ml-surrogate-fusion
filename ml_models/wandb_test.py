import wandb

wandb.login()

import math
import random

# Launch 5 simulated experiments
for run in range(5):
    # 1️⃣ Start a new run to track this script
    wandb.init(
        # Set entity to specify your username or team name
        # ex: entity="carey",
        # Set the project where this run will be logged
        project="basic-intro",
        # Track hyperparameters and run metadata
        config={"learning_rate": 0.02, "architecture": "CNN", "dataset": "CIFAR-100",},
    )

    offset = random.random()

    # This simple block simulates a training loop logging metrics
    for x in range(50):
        acc = 0.16 * (math.log(1 + x + random.random()) + random.random() + offset)
        loss = 1 - 0.16 * (math.log(1 + x + random.random()) + random.random() + offset)
        # 2️⃣ Log metrics from your script to W&B
        wandb.log({"acc": acc, "loss": loss})

    # Mark the run as finished
    wandb.finish()

