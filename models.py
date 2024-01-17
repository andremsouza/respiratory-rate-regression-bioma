"""Define models for regression of videos with PyTorch."""

# %% [markdown]
# # Imports

# %%
import lightning.pytorch as pl
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
import torchvision
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

import config
import data

# %% [markdown]
# # Constants

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class MViTV2Regression(pl.LightningModule):
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
        # self.weights = weights
        # Change the number of output features to 1
        self._update_heads()
        # Initiate auto_augment
        self.auto_augment = torchvision.transforms.AutoAugment()
        # Initialize training metrics
        self.train_mae = MeanAbsoluteError()
        self.train_mape = MeanAbsolutePercentageError()
        self.train_mse = MeanSquaredError(squared=True)
        self.train_rmse = MeanSquaredError(squared=False)
        # Initialize validation metrics
        self.val_mae = MeanAbsoluteError()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_mse = MeanSquaredError(squared=True)
        self.val_rmse = MeanSquaredError(squared=False)
        # Initialize test metrics
        self.test_mae = MeanAbsoluteError()
        self.test_mape = MeanAbsolutePercentageError()
        self.test_mse = MeanSquaredError(squared=True)
        self.test_rmse = MeanSquaredError(squared=False)

    def _update_heads(self) -> None:
        """Update the heads of the model."""
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

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        inputs, labels = batch
        idx: int = 0
        # expand into batches of N frames with stride S
        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
        while inputs.shape[1] < config.BATCH_SIZE:
            # get first batch_size frames from inputs
            inputs_first = data.expand_video_into_batches(
                inputs,
                batch_size=config.BATCH_SIZE,
                stride=config.STRIDE,
                first_only=True,
            ).float()
            # remove stride frames from inputs
            inputs = inputs[:, config.STRIDE :, :, :].float()
            # auto augment inputs
            inputs_augmented = [inputs_first.to(torch.uint8)]
            if self.auto_augment is not None:
                # squeeze batch dimension
                inputs_first = inputs_first.squeeze(0)
                # invert dimensions to use torchvision transforms
                inputs_first = inputs_first.permute(1, 0, 2, 3)
                # create N augmented sequences and stack them
                for _ in range(config.AUTOAUGMENT_N):
                    inputs_augmented.append(
                        self.auto_augment(inputs_first.to(torch.uint8))
                        .permute(1, 0, 2, 3)
                        .unsqueeze(0)
                    )
                inputs_augmented = torch.cat(inputs_augmented, dim=0)
                # send to device
                inputs_augmented = inputs_augmented.float().to(DEVICE)
            else:
                # send to device
                inputs_augmented = inputs_first.float().to(DEVICE)
            idx += 1
            # expand labels to comply with inputs_augmented shape
            labels = labels.expand(inputs_augmented.shape[0], -1)
            # send labels to device
            labels = labels.float().to(DEVICE)
            # forward pass
            outputs = self(inputs_augmented)
            # calculate loss
            loss = F.mse_loss(outputs, labels)
            # log loss
            self.log("train_loss", loss)
            # calculate metrics
            self.train_mae(outputs, labels)
            self.train_mape(outputs, labels)
            self.train_mse(outputs, labels)
            self.train_rmse(outputs, labels)
            # log metrics
            self.log("train_mae", self.train_mae)
            self.log("train_mape", self.train_mape)
            self.log("train_mse", self.train_mse)
            self.log("train_rmse", self.train_rmse)
            # return loss
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        pass

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        pass


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
        # self.weights = weights
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
