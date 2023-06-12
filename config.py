"""Configuration file for the project.

This file contains all the global constants used in the project.
"""
# %%
import torch

# %% [markdown]
# # Constants

# %%
# ANNOTATION_FILE = "/srv/andre/30_09_2020.csv"
ANNOTATION_FILE = "./Resp_30s.csv"
LABEL_COLUMN = "Taxa / 60s"
TRAIN_ANNOTATION_FILE = "./train_annotation.csv"
TEST_ANNOTATION_FILE = "./test_annotation.csv"

ANNOTATION_SECONDS = 30

DATA_DIRECTORY = "./videos/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIRECTORY = "./models/"

BATCH_SIZE = 16
STRIDE = 8
AUTOAUGMENT = True
AUTOAUGMENT_N = 16
PRED_THRESHOLD = 0.5

SKIP_TRAINED_MODELS = False

# %%
