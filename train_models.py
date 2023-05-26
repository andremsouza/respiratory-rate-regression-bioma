"""Setup and run models.

This script is used to setup and run the models. It is used to train the models
and save the trained models.

It is recommended to run this script with nohup and redirecting the output to a
file. For example:
    > nohup python train_models.py > train_models.out &
"""
# %%
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

import data
import models
import config
import utils

SKIP_TRAINED_MODELS = True

# %% [markdown]
# # Data

# %%
# PyTorch device
device = torch.device(config.DEVICE)
# Load data
# Use seed for selecting training and test data
RANDOM_SEED = 42
# Load annotation file
# If train_annotation and test_annotation files do not exist, create them
try:
    train_annotation = pd.read_csv(config.TRAIN_ANNOTATION_FILE)
    test_annotation = pd.read_csv(config.TEST_ANNOTATION_FILE)
except FileNotFoundError:
    # Create train and test annotation files with random sampling
    annotation = pd.read_csv(config.ANNOTATION_FILE)
    train_annotation, test_annotation = train_test_split(
        annotation,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )
    # Save annotation files
    train_annotation.to_csv(config.TRAIN_ANNOTATION_FILE, index=False)
    test_annotation.to_csv(config.TEST_ANNOTATION_FILE, index=False)

# %% [markdown]
# # MViT-V2

# %%
# Create data loaders
transforms = models.MViT_V2_S_Weights.DEFAULT.transforms()
# torch.multiprocessing.set_start_method('spawn')
train_loader = DataLoader(
    data.VideoDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=transforms,
        # transforms(
        # data.expand_video_into_batches(x, batch_size=16, stride=8, device=device)
        # ),
        target_transform=lambda x: data.expand_label(x, 1, device=device),
    ),
    batch_size=1,
    shuffle=True,
    num_workers=24,
)

# %%

# Train models
for learning_rate in models.LEARNING_RATES:
    # Create model and load state dict if it exists
    model = models.MViTV2Regression(weights=models.MViT_V2_S_Weights.DEFAULT).to(device)
    # Load state dict if it exists
    try:
        model.load_state_dict(torch.load(f"models/mvitv2_{learning_rate}.pt"))
        if SKIP_TRAINED_MODELS:
            continue
    except FileNotFoundError:
        # Touch file so it exists
        # This crudely enables training multiple models in parallel
        # However, if there is an interruption during training, the file will not be deleted
        # In this case, the file will need to be deleted manually
        open(f"models/mvitv2_{learning_rate}.pt", "a", encoding="utf-8").close()
    except EOFError:
        continue
    print(
        f"{datetime.datetime.now()}: "
        f"Training mvitv2 model w/ {learning_rate} learning rate"
    )
    utils.train(
        model,
        train_loader,
        criterion=torch.nn.MSELoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        device=device,
        num_epochs=10,
        n_classes=1,
        verbose=True,
    )
    # Save model state dict
    print(
        f"{datetime.datetime.now()}: "
        f"Saving mvitv2 model w/ {learning_rate} learning rate"
    )
    torch.save(
        model.state_dict(),
        f"models/mvitv2_{learning_rate}.pt",
    )

# %%
