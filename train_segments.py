"""Setup and run models.

This script is used to setup and run the models. It is used to train the models
and save the trained models.

It is recommended to run this script with nohup and redirecting the output to a
file. For example:
    > nohup python train_models.py > train_models.out &
"""
# %%
import datetime

import torch
from torch.utils.data import DataLoader

# import torchvision

import data
import models
import config
import utils

# %% [markdown]
# # Data

# %%
# PyTorch device
device = torch.device(config.DEVICE)
# Load data
# Use seed for selecting training and test data
RANDOM_SEED = 42
# Load annotation file
# Load annotation file
train_annotations_file, test_annotations_file = data.split_annotations(
    config.ANNOTATION_FILE_CLASSIFICATION,
    train_percentage=config.TRAIN_PERCENTAGE,
    val_percentage=config.VAL_PERCENTAGE,
    random_seed=RANDOM_SEED,
)
# %% [markdown]
# # MViT-V2

# %%
# Create data loaders
transforms = models.MViT_V2_S_Weights.DEFAULT.transforms()
# torch.multiprocessing.set_start_method('spawn')
train_loader = DataLoader(
    data.VideoDataset(
        train_annotations_file,
        config.DATA_DIRECTORY,
        fps=7.5,
        transform=transforms,
        # transforms(
        # data.expand_video_into_batches(x, batch_size=16, stride=8, device=device)
        # ),
        target_transform=lambda x: data.expand_label(x, 1),
        # filter_ids=[5007, 5008, 5009, 5010, 5011, 5012],
        bbox_transform=False,
        trim_video=False,
        classification=True,
    ),
    batch_size=1,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)
test_loader = DataLoader(
    data.VideoDataset(
        test_annotations_file,
        config.DATA_DIRECTORY,
        fps=7.5,
        transform=transforms,
        # transforms(
        # data.expand_video_into_batches(x, batch_size=16, stride=8, device=device)
        # ),
        target_transform=lambda x: data.expand_label(x, 1),
        # filter_ids=[5007, 5008, 5009, 5010, 5011, 5012],
        bbox_transform=False,
        trim_video=False,
        classification=True,
    ),
    batch_size=1,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)

# %%

# Train models
for learning_rate in models.LEARNING_RATES:
    # Create model and load state dict if it exists
    model = models.MViTV2BinaryClassification(  # pylint: disable=invalid-name
        weights=models.MViT_V2_S_Weights.DEFAULT,
    ).to(device)
    # Load state dict if it exists
    try:
        model.load_state_dict(
            torch.load(
                f"{config.MODELS_CLASSIFICATION_DIRECTORY}{model.__class__.__name__}_{learning_rate}_best.pt"
            )
        )
        if config.SKIP_TRAINED_MODELS:
            continue
        print(
            f"{datetime.datetime.now()}: "
            f"Loaded mvitv2 model w/ {learning_rate} learning rate"
        )
    except FileNotFoundError:
        # Touch file so it exists
        # This crudely enables training multiple models in parallel
        # However, if there is an interruption during training, the file will not be deleted
        # In this case, the file will need to be deleted manually
        with open(
            f"{config.MODELS_CLASSIFICATION_DIRECTORY}{model.__class__.__name__}_{learning_rate}_best.pt",
            "wb",
        ) as f:
            f.close()
    except EOFError:
        # If the file is empty, then it is not a valid state dict
        if config.SKIP_TRAINED_MODELS:
            continue
    print(
        f"{datetime.datetime.now()}: "
        f"Training mvitv2 model w/ {learning_rate} learning rate"
    )
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimizer, mode="min", factor=0.2, patience=10, verbose=True
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=10, T_mult=1, eta_min=1e-6, verbose=True
    )
    best_model_weights, metrics = utils.fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=torch.nn.BCELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=200,
        patience=20,
        device=device,
        save_best="disk",
        verbose=True,
    )
    # Load best model weights
    model.load_state_dict(
        torch.load(
            f"{config.MODELS_CLASSIFICATION_DIRECTORY}{model.__class__.__name__}_{learning_rate}_best.pt"
        )
    )
    # Save model state dict
    print(
        f"{datetime.datetime.now()}: "
        f"Saving mvitv2 model w/ {learning_rate} learning rate"
    )
    torch.save(
        model.state_dict(),
        f"{config.MODELS_CLASSIFICATION_DIRECTORY}{model.__class__.__name__}_{learning_rate}_best.pt",
    )
    break

# %%
