"""Define models for regression of videos with PyTorch."""

# %% [markdown]
# # Imports

# %%
import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

# %% [markdown]
# # Constants

# %%
LEARNING_RATES = [
    1.6e-3,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
]

# %% [markdown]
# # Model

# %%


class MViTV2Regression(nn.Module):
    """MViT_V2_S model for regression.

    Model for regression of videos using the MViT_V2_S model from torchvision.
    The model is initialized with weights from the torchvision model.
    The number of output features is changed to 1.

    Args:
        weights (MViT_V2_S_Weights, optional): Weights for the model.
            Defaults to MViT_V2_S_Weights.DEFAULT.

    Attributes:
        model (torch.nn.Module): The model.
        weights (MViT_V2_S_Weights): The weights used to initialize the model.
    """

    def __init__(self, *args, weights=MViT_V2_S_Weights.DEFAULT, **kwargs) -> None:
        """Initialize the model.

        Args:
            weights (MViT_V2_S_Weights, optional): Weights for the model.
                Defaults to MViT_V2_S_Weights.DEFAULT.
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
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, num_frames, num_channels, height, width)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, 1)
        """
        return self.model(x)


class MViTV2BinaryClassification(nn.Module):
    """MViT_V2_S model for binary classification.

    Model for binary classification of videos using the MViT_V2_S model from torchvision.
    The model is initialized with weights from the torchvision model.
    The number of output features is changed to 1 and a sigmoid activation function is added.

    Args:
        weights (MViT_V2_S_Weights, optional): Weights for the model.
            Defaults to MViT_V2_S_Weights.DEFAULT.

    Attributes:
        model (torch.nn.Module): The model.
        weights (MViT_V2_S_Weights): The weights used to initialize the model.
    """

    def __init__(self, *args, weights=MViT_V2_S_Weights.DEFAULT, **kwargs) -> None:
        """Initialize the model.

        Args:
            weights (MViT_V2_S_Weights, optional): Weights for the model.
                Defaults to MViT_V2_S_Weights.DEFAULT.
        """
        super().__init__(*args, **kwargs)
        self.model = mvit_v2_s(weights=weights)
        self.weights = weights
        # Change the number of output features to 1
        self.model.head = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5, inplace=True),
            torch.nn.Linear(in_features=768, out_features=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.
                Shape: (batch_size, num_frames, num_channels, height, width)

        Returns:
            torch.Tensor: Output tensor. Shape: (batch_size, 1)
        """
        return self.model(x)


# %%

# %%
