"""Regression of breathing rate in a dataset of video clips."""

# %% [markdown]
# ## Imports

# %%
import argparse
from datetime import datetime
from functools import partial
import os
import random

import dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights
from torchvision.transforms.v2 import Compose, InterpolationMode, Normalize, Resize
from torchvision.io import write_video
from torchvision.transforms._presets import VideoClassification

import config
from datasets import VideoDataset
from models import R2Plus1D18Regression

# %% [markdown]
# # Constants and arguments

# %%
dotenv.load_dotenv(verbose=True, override=True)
torch.set_float32_matmul_precision("high")

# Initialize argument parser
parser = argparse.ArgumentParser()
# Add arguments
# Add label studio url
parser.add_argument(
    "--label-studio-url",
    type=str,
    default=os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"),
    help="URL of Label Studio instance",
)
# Add label studio api key
parser.add_argument(
    "--label-studio-api-key",
    type=str,
    default=os.getenv("LABEL_STUDIO_API_KEY", ""),
    help="API key for Label Studio instance",
)
# Add label studio container id
parser.add_argument(
    "--label-studio-container-id",
    type=str,
    default=os.getenv("LABEL_STUDIO_CONTAINER_ID", None),
    help="Container ID for Label Studio instance",
)
# Add label studio container data dir
parser.add_argument(
    "--label-studio-container-data-dir",
    type=str,
    default=os.getenv("LABEL_STUDIO_CONTAINER_DATA_DIR", None),
    help="Container data directory for Label Studio instance",
)
# Add label studio download dir
parser.add_argument(
    "--label-studio-download-dir",
    type=str,
    default=os.getenv("LABEL_STUDIO_DOWNLOAD_DIR", "data/lsvideos/"),
    help="Download directory for Label Studio instance",
)
# Add label studio project id
parser.add_argument(
    "--label-studio-project-id",
    type=str,
    default=os.getenv("LABEL_STUDIO_PROJECT_ID", None),
    help="Project ID for Label Studio instance",
)
# Add target fps
parser.add_argument(
    "--target-fps",
    type=float,
    default=os.getenv("TARGET_FPS", 5.0),
    help="Target FPS for video clips",
)
# Add sample_size
parser.add_argument(
    "--sample-size",
    type=int,
    default=os.getenv("SAMPLE_SIZE", 16),
    help="Sample size for video clips",
)
# Add hop length
parser.add_argument(
    "--hop-length",
    type=int,
    default=os.getenv("HOP_LENGTH", 8),
    help="Hop length for video clips",
)
# Add filter task ids (list of integers, separated by commas)
parser.add_argument(
    "--filter-task-ids",
    type=str,
    default=os.getenv("FILTER_TASK_IDS", ""),
    help="List of task IDs to filter",
)
# Add bbox_transform bool
parser.add_argument(
    "--bbox-transform",
    type=bool,
    default=os.getenv("BBOX_TRANSFORM", False),
    help="Whether to transform bounding boxes",
)
# Add bbox_transform_corners bool
parser.add_argument(
    "--bbox-transform-corners",
    type=bool,
    default=os.getenv("BBOX_TRANSFORM_CORNERS", False),
    help="Whether to transform bounding box corners",
)
# Add download_videos bool
parser.add_argument(
    "--download-videos",
    type=bool,
    default=os.getenv("DOWNLOAD_VIDEOS", True),
    help="Whether to download videos",
)
# Add download_videos_overwrite bool
parser.add_argument(
    "--download-videos-overwrite",
    type=bool,
    default=os.getenv("DOWNLOAD_VIDEOS_OVERWRITE", False),
    help="Whether to overwrite existing videos",
)
# Add verbose bool
parser.add_argument(
    "--verbose",
    type=bool,
    default=os.getenv("VERBOSE", True),
    help="Whether to print verbose output",
)
# Add model dir
parser.add_argument(
    "--model-dir",
    type=str,
    default=os.getenv("MODEL_DIR", "models/"),
    help="Directory for models",
)
# Add log dir
parser.add_argument(
    "--log-dir",
    type=str,
    default=os.getenv("LOG_DIR", "logs/"),
    help="Directory for logs",
)
# Add num_workers
parser.add_argument(
    "--num-workers",
    type=int,
    default=os.getenv("NUM_WORKERS", os.cpu_count() // 2 if os.cpu_count() else 1),
    help="Number of workers for dataloaders",
)
# Add batch size
parser.add_argument(
    "--batch-size",
    type=int,
    default=os.getenv("BATCH_SIZE", 16),
    help="Batch size for training",
)
# Add optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default=os.getenv("OPTIMIZER", "adamw"),
    help="Optimizer for training",
)
# Add learning rate
parser.add_argument(
    "--learning-rate",
    type=float,
    default=os.getenv("LEARNING_RATE", 0.001),
    help="Learning rate for training",
)
# Add weight decay
parser.add_argument(
    "--weight-decay",
    type=float,
    default=os.getenv("WEIGHT_DECAY", 0.01),
    help="Weight decay for training",
)
# Add max epochs
parser.add_argument(
    "--max-epochs",
    type=int,
    default=os.getenv("MAX_EPOCHS", 1000),
    help="Maximum number of epochs for training",
)
# Add patience
parser.add_argument(
    "--patience",
    type=int,
    default=os.getenv("PATIENCE", 8),
    help="Patience for early stopping",
)
# Add seed
parser.add_argument(
    "--seed",
    type=int,
    default=os.getenv("SEED", 42),
    help="Seed for random number generators",
)
# Add model name
parser.add_argument(
    "--model-name",
    type=str,
    default=os.getenv("MODEL_NAME", "r2plus1d18"),
    help="Name of model",
)
# Add pretrained bool
parser.add_argument(
    "--pretrained",
    type=bool,
    default=os.getenv("PRETRAINED", True),
    help="Whether to use pretrained model",
)
# Parse --f argument for Jupyter Notebook (ignored by argparse)
parser.add_argument(
    "--f",
    type=str,
    default="",
    help="",
)
# Parse arguments
args = parser.parse_args()
# Print all arguments
for arg in vars(args):
    print(arg, getattr(args, arg))
# Set argument constants
LABEL_STUDIO_URL: str = args.label_studio_url
LABEL_STUDIO_API_KEY: str = args.label_studio_api_key
LABEL_STUDIO_CONTAINER_ID: str = args.label_studio_container_id
LABEL_STUDIO_CONTAINER_DATA_DIR: str = args.label_studio_container_data_dir
LABEL_STUDIO_DOWNLOAD_DIR: str = args.label_studio_download_dir
LABEL_STUDIO_PROJECT_ID: str = args.label_studio_project_id
TARGET_FPS: float = args.target_fps
SAMPLE_SIZE: int = args.sample_size
HOP_LENGTH: int = args.hop_length
FILTER_TASK_IDS: list | None = (
    [int(task_id) for task_id in args.filter_task_ids.split(",")]
    if args.filter_task_ids
    else None
)
BBOX_TRANSFORM: bool = args.bbox_transform
BBOX_TRANSFORM_CORNERS: bool = args.bbox_transform_corners
DOWNLOAD_VIDEOS: bool = args.download_videos
DOWNLOAD_VIDEOS_OVERWRITE: bool = args.download_videos_overwrite
VERBOSE: bool = args.verbose
MODEL_DIR: str = args.model_dir
LOG_DIR: str = args.log_dir
NUM_WORKERS: int = args.num_workers
BATCH_SIZE: int = args.batch_size
OPTIMIZER: str = args.optimizer
LEARNING_RATE: float = args.learning_rate
WEIGHT_DECAY: float = args.weight_decay
MAX_EPOCHS: int = args.max_epochs
PATIENCE: int = args.patience
SEED: int = args.seed
MODEL_NAME: str = args.model_name + "_regression"
PRETRAINED: bool = args.pretrained
# Set random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# %%
# Create directories if they don't exist
if not os.path.exists(LABEL_STUDIO_DOWNLOAD_DIR):
    os.makedirs(LABEL_STUDIO_DOWNLOAD_DIR, exist_ok=True)
if not os.path.exists(
    LOG_DIR,
):
    os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

# %% [markdown]
# # Main

# %%
if __name__ == "__main__":
    # Get transforms from weights
    if BBOX_TRANSFORM:
        # Use r2plus1d18 transforms, without center crop
        transform = partial(
            VideoClassification, crop_size=(112, 112), resize_size=(112, 112)
        )()
    else:
        # Use r2plus1d18 default transforms
        transform = R2Plus1D_18_Weights.DEFAULT.transforms()
    # Load dataset
    dataset = VideoDataset(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY,
        project_id=int(LABEL_STUDIO_PROJECT_ID),
        data_dir=LABEL_STUDIO_DOWNLOAD_DIR,
        container_id=LABEL_STUDIO_CONTAINER_ID,
        container_data_dir=LABEL_STUDIO_CONTAINER_DATA_DIR,
        fps=TARGET_FPS,
        sample_size=SAMPLE_SIZE,
        hop_length=HOP_LENGTH,
        filter_task_ids=FILTER_TASK_IDS,
        bbox_transform=BBOX_TRANSFORM,
        bbox_transform_corners=BBOX_TRANSFORM_CORNERS,
        download_videos=DOWNLOAD_VIDEOS,
        download_videos_overwrite=DOWNLOAD_VIDEOS_OVERWRITE,
        classification=False,
        prune_invalid=False,
        transform=transform,
        target_transform=lambda x: torch.tensor(x).unsqueeze(0),
        verbose=VERBOSE,
    )
    # Get task ids
    task_ids = dataset.annotations["id"].unique()
    # Split task ids into train and test
    train_task_ids, test_task_ids = train_test_split(
        task_ids, test_size=0.2, random_state=SEED
    )
    # Split dataset into train and test
    del dataset
    train_dataset = VideoDataset(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY,
        project_id=int(LABEL_STUDIO_PROJECT_ID),
        data_dir=LABEL_STUDIO_DOWNLOAD_DIR,
        container_id=LABEL_STUDIO_CONTAINER_ID,
        container_data_dir=LABEL_STUDIO_CONTAINER_DATA_DIR,
        fps=TARGET_FPS,
        sample_size=SAMPLE_SIZE,
        hop_length=HOP_LENGTH,
        filter_task_ids=train_task_ids,
        bbox_transform=BBOX_TRANSFORM,
        bbox_transform_corners=BBOX_TRANSFORM_CORNERS,
        download_videos=DOWNLOAD_VIDEOS,
        download_videos_overwrite=DOWNLOAD_VIDEOS_OVERWRITE,
        classification=False,
        prune_invalid=True,
        transform=transform,
        target_transform=lambda x: torch.tensor(x).unsqueeze(0),
        verbose=VERBOSE,
    )
    test_dataset = VideoDataset(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY,
        project_id=int(LABEL_STUDIO_PROJECT_ID),
        data_dir=LABEL_STUDIO_DOWNLOAD_DIR,
        container_id=LABEL_STUDIO_CONTAINER_ID,
        container_data_dir=LABEL_STUDIO_CONTAINER_DATA_DIR,
        fps=TARGET_FPS,
        sample_size=SAMPLE_SIZE,
        hop_length=HOP_LENGTH,
        filter_task_ids=test_task_ids,
        bbox_transform=BBOX_TRANSFORM,
        bbox_transform_corners=BBOX_TRANSFORM_CORNERS,
        download_videos=DOWNLOAD_VIDEOS,
        download_videos_overwrite=DOWNLOAD_VIDEOS_OVERWRITE,
        classification=False,
        prune_invalid=True,
        transform=transform,
        target_transform=lambda x: torch.tensor(x).unsqueeze(0),
        verbose=VERBOSE,
    )
    print(f"[{datetime.now()}]: Loaded datasets")
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"[{datetime.now()}]: Created data loaders")
    # Create model
    # Set dataloaders (for generic use)
    train_dataloaders: list[DataLoader] = [train_dataloader]
    test_dataloaders: list[DataLoader] = [test_dataloader]

# %%
# if __name__ == "__main__":
#     # Get a sample from the dataset and save it
#     train_dataset.transform = None
#     sample, _ = train_dataset[0]
#     print(f"[{datetime.now()}]: Got sample")
#     # Save sample as video
#     write_video(
#         os.path.join("sample.mp4"),
#         # sample.permute(1, 2, 3, 0),
#         sample.permute(0, 2, 3, 1),
#         fps=TARGET_FPS,
#     )
#     print(f"[{datetime.now()}]: Saved sample as video {os.path.join('sample.mp4')}")

# %%
if __name__ == "__main__" and 1 == 2:
    for train_dataloader, test_dataloader in zip(train_dataloaders, test_dataloaders):
        for loss_fn_name in ["mseloss"]:
            experiment_name: str = (
                f"{MODEL_NAME}_"
                f"pretrained{PRETRAINED}_"
                f"batch{BATCH_SIZE}_"
                f"bbox{BBOX_TRANSFORM}_"
                f"corner{BBOX_TRANSFORM_CORNERS}_"
                f"{loss_fn_name}_"
            )
            loss_function: nn.Module = {
                "l1loss": nn.L1Loss(),
                "mseloss": nn.MSELoss(),
            }[loss_fn_name]
            model: R2Plus1D18Regression = R2Plus1D18Regression(
                num_classes=1,
                optimizer=OPTIMIZER,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                early_stopping_patience=PATIENCE,
                loss_function=loss_function,
                weights=R2Plus1D_18_Weights.DEFAULT if PRETRAINED else None,
            )
            # Train model
            early_stopping: EarlyStopping = EarlyStopping(
                monitor="val_loss", patience=PATIENCE, mode="min"
            )
            loggers: list = [
                CSVLogger(save_dir=LOG_DIR, name=experiment_name),
                TensorBoardLogger(save_dir=LOG_DIR, name=experiment_name),
            ]
            checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
                dirpath=MODEL_DIR,
                filename=experiment_name + "-{val_loss:.2f}-{epoch:02d}",
                monitor="val_loss",
                verbose=True,
                save_top_k=1,
                save_weights_only=False,
                mode="min",
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
            )
            trainer: pl.Trainer = pl.Trainer(
                callbacks=[early_stopping, checkpoint_callback],
                max_epochs=MAX_EPOCHS,
                logger=loggers,
                log_every_n_steps=min(50, len(train_dataloader)),
            )
            trainer.fit(model, train_dataloader, test_dataloader)
            print(f"[{datetime.now()}]: Finished training {experiment_name}")
    print(f"[{datetime.now()}]: Finished training")

# %%
