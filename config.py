"""Configuration file for the project.

This file contains all the global constants used in the project.
"""
# %%
import torch

# %% [markdown]
# # Constants

# %%
# ANNOTATION_FILE = "/srv/andre/30_09_2020.csv"
ANNOTATION_FILE = "./Resp_30s.xlsx"
TRAIN_ANNOTATION_FILE = "./train_annotation.csv"
TEST_ANNOTATION_FILE = "./test_annotation.csv"

ANNOTATION_SECONDS = 30

DATA_DIRECTORY = "./videos/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
STRIDE = 8

PRED_THRESHOLD = 0.5

# %%
