"""Define models for regression of videos with PyTorch."""

# %% [markdown]
# # Imports

# %%
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

from data import VideoDataset

# %% [markdown]
# # Model

# %%


class MViTV2Regression(nn.Module):
    def __init__(self, weights=MViT_V2_S_Weights.DEFAULT, *args, **kwargs) -> None:
        """Initialize the model."""
        super().__init__(*args, **kwargs)
        self.model = mvit_v2_s(weights=weights)
        # Change the number of output features to 1
        self.model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=768, out_features=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


# %%

# %%
