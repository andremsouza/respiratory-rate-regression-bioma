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
# # Constants

# %%
LEARNING_RATES = [1e-3, 1e-4, 1e-5, 1e-6]

# %% [markdown]
# # Model

# %%


class MViTV2Regression(nn.Module):
    def __init__(self, weights=MViT_V2_S_Weights.DEFAULT, *args, **kwargs) -> None:
        """Initialize the model.

        Args:
            weights (MViT_V2_S_Weights, optional): Weights for the model. Defaults to MViT_V2_S_Weights.DEFAULT.
        """
        super().__init__(*args, **kwargs)
        self.model = mvit_v2_s(weights=weights)
        self.weights = weights
        # Change the number of output features to 1
        self.model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=768, out_features=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, num_frames, num_channels, height, width)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, 1)
        """
        return self.model(x)


# %%

# %%
