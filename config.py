"""Configuration file for the project.

This file contains all the global constants used in the project.
"""
# %%
import torch

# %% [markdown]
# # Constants

# %%
# ANNOTATION_FILE = "/srv/andre/30_09_2020.csv"
ANNOTATION_FILE_REGRESSION = "project-6-at-2023-08-01-16-01-a81a144c.json"
ANNOTATION_FILE_CLASSIFICATION = "project-6-at-2023-08-01-16-01-a81a144c.json"
LABEL_COLUMN = "Taxa / 60s"
# TRAIN_ANNOTATION_FILE = "./train_annotation.csv"
# TRAIN_ANNOTATION_FILE = "project-6-at-2023-07-17-15-08-b48ff462.json"
# TEST_ANNOTATION_FILE = "./test_annotation.csv"
# TEST_ANNOTATION_FILE = "project-6-at-2023-07-17-15-08-b48ff462.json"
TRAIN_PERCENTAGE: float = 0.8
VAL_PERCENTAGE: float = 0.2

ANNOTATION_SECONDS = 30

DATA_DIRECTORY = "./videos/"
# DATA_DIRECTORY = "./videos/label-studio/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIRECTORY = "./models/"
MODELS_CLASSIFICATION_DIRECTORY = "./models/classification/"
# MODELS_DIRECTORY = "./models/bbox/"

NUM_WORKERS = 1
BATCH_SIZE = 16
STRIDE = 8
AUTOAUGMENT = True
AUTOAUGMENT_N = 1
PRED_THRESHOLD = 0.5
WEIGHT_DECAY = 0.05

SKIP_TRAINED_MODELS = False

# %%
