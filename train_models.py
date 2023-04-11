"""Setup and run models.

This script is used to setup and run the models. It is used to train the models
and save the trained models.
"""
# %%
import datetime
import warnings

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
# Load data
device = torch.device(config.DEVICE)

# Get length of dataset
dataset_length = len(
    data.AudioDataset(
        config.ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=None,
        target_transform=None,
    )
)
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
    # Sort by Timestamp
    train_annotation = train_annotation.sort_values(by="Timestamp")
    test_annotation = test_annotation.sort_values(by="Timestamp")
    # Save annotation files
    train_annotation.to_csv(config.TRAIN_ANNOTATION_FILE, index=False)
    test_annotation.to_csv(config.TEST_ANNOTATION_FILE, index=False)

    # %% [markdown]
# # Fully Connected Neural Network

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: torch.flatten(x[None, :, :64]),
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)
# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for n_layers in models.fully_connected.N_LAYERS:
        for m_units in models.fully_connected.M_UNITS:
            for learning_rate in models.fully_connected.LEARNING_RATES:
                # Create model and load state dict if it exists
                model = models.fully_connected.FullyConnected(
                    n_layers=n_layers,
                    m_units=m_units,
                    n_features=96 * 64,
                    m_labels=9,
                ).to(device)
                try:
                    torch.load(
                        f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
                    )
                    if SKIP_TRAINED_MODELS:
                        continue
                except FileNotFoundError:
                    # Touch file so it exists
                    open(
                        f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
                        "a",
                        encoding="utf-8",
                    ).close()
                except EOFError:
                    continue
                print(
                    f"{datetime.datetime.now()}: "
                    f"Training FC model w/ {n_layers} layers, {m_units} units, {learning_rate} lr"
                )
                utils.train(
                    model,
                    train_loader,
                    criterion=torch.nn.BCELoss(),
                    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
                    device=device,
                    num_epochs=10,
                    n_classes=9,
                    verbose=True,
                )
                # Save model state dict
                print(
                    f"{datetime.datetime.now()}: "
                    f"Saving model w/ {n_layers} layers, {m_units} units, {learning_rate} lr"
                )
                torch.save(
                    model.state_dict(),
                    f"models/fc_{n_layers}_{m_units}_{learning_rate}.pt",
                )

# %%
# TODO: Test model and select samples with highest entropy

# %% [markdown]
# # AlexNet

# %%
# Create data loaders
train_loader = DataLoader(
    data.AudioDataset(
        config.TRAIN_ANNOTATION_FILE,
        config.DATA_DIRECTORY,
        transform=lambda x: x[None, :, :64],
        target_transform=lambda x: x[0, :],
    ),
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=24,
)

# Train models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for learning_rate in models.alexnet.LEARNING_RATES:
        # Create model and load state dict if it exists
        model = models.alexnet.AlexNet(num_classes=9, dropout=0.5).to(device) # type: ignore
        # Load state dict if it exists
        try:
            model.load_state_dict(torch.load(f"models/alexnet_{learning_rate}.pt"))
            if SKIP_TRAINED_MODELS:
                continue
        except FileNotFoundError:
            # Touch file so it exists
            open(f"models/alexnet_{learning_rate}.pt", "a", encoding="utf-8").close()
        except EOFError:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Training AlexNet model w/ {learning_rate} learning rate"
        )
        utils.train(
            model,
            train_loader,
            criterion=torch.nn.BCELoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
            device=device,
            num_epochs=10,
            n_classes=9,
            verbose=True,
        )
        # Save model state dict
        print(
            f"{datetime.datetime.now()}: "
            f"Saving AlexNet model w/ {learning_rate} learning rate"
        )
        torch.save(
            model.state_dict(),
            f"models/alexnet_{learning_rate}.pt",
        )

# %%
# TODO: Test model and select samples with highest entropy
