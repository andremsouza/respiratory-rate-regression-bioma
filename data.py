"""Preprocess data for training and testing."""

# %% [markdown]
# # Imports

# %%
import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import GaussianBlur, Resize

# %% [markdown]
# # Constants

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"

# %% [markdown]
# # Dataset

# %%


class VideoDataset(Dataset):
    """Dataset for video data."""

    def __init__(self, data_dir, transform=None, target_transform=None):
        """Initialize the dataset."""
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data = pd.read_csv(os.path.join(data_dir, "data.csv"))
        self.labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):  #  -> tuple[torch.Tensor, torch.float]:
        """Return the idx-th element of the dataset."""
        video = np.load(os.path.join(self.data_dir, self.data.iloc[idx, 0]))
        video = torch.tensor(video, dtype=torch.float)
        if self.transform:
            video = self.transform(video)
        label = torch.tensor(self.labels.iloc[idx, 1], dtype=torch.float)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label


# %% [markdown]
# # Transform functions

# %%


def instance_segment(
    video: torch.Tensor,
    model,
    weights,
    threshold=0.0,
    batch_size=1,
    stride=1,
    device=DEVICE,
    verbose=False,
) -> torch.Tensor:
    """Segment the video using an instance segmentation model."""
    # send model to device
    model.to(device)
    # set model to evaluation mode
    model.eval()
    if verbose:
        print("Performing instance segmentation on video with shape", video.shape)
    # get the preprocessing function for the model
    transforms = weights.transforms()
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    masks = []
    for i in range(0, len(video), stride):
        # check if there are enough frames left to fill a batch
        if i + batch_size > len(video):
            # if not, use the remaining frames
            batch_size = len(video) - i
        # preprocess next batch of video frames
        if verbose:
            print(f"Processing frames {i} to {i + batch_size}")
        frames = torch.stack(
            [transforms(frame.permute(2, 0, 1)) for frame in video[i : i + batch_size]]
        )
        outputs = model(frames.to(device))
        # output is a list of dicts with keys "boxes", "labels", "scores", and "masks"
        # get the masks for the cow class
        for idx, output in enumerate(outputs):
            # get indexes of instances labeled as cows
            cow_idxs = output["labels"] == class_to_idx["cow"]
            # get the masks for the cow class
            cow_masks = output["masks"][cow_idxs]
            # get the scores for the cow class
            cow_scores = output["scores"][cow_idxs]
            # get the mask for the cow class with the highest bbox area
            # NOTE: this is not the best way to do this, but it works for now
            # get bbox areas
            areas = []
            for box in output["boxes"][cow_idxs]:
                areas.append((box[2] - box[0]) * (box[3] - box[1]))
            areas = torch.tensor(areas)
            # get the mask with the highest bbox area
            mask = cow_masks[areas.argmax()]
            # get the score for the mask
            score = cow_scores[areas.argmax()]
            # add the mask to the list of masks if the score is above the threshold
            if score >= threshold:
                masks.append(mask.numpy(force=True))
        # free up memory
        del frames
        del outputs

        gc.collect()
        torch.cuda.empty_cache()
    return torch.tensor(np.stack(masks))


def semantic_segment(
    video: torch.Tensor,
    model,
    weights,
    batch_size=1,
    stride=1,
    device=DEVICE,
    verbose=False,
) -> torch.Tensor:
    """Segment the video using a semantic segmentation model.

    Args:
        video (torch.Tensor): The video to segment.
        model (torch.nn.Module): The model to use for segmentation.
        weights (torchvision.models.segmentation._utils._SegmentationModel): The weights
            to use for the model.
        batch_size (int, optional): The number of frames to process at a time. Defaults to
            1.
        stride (int, optional): The number of frames to skip between batches. Defaults to
            1.
        device (torch.device, optional): The device to use for processing. Defaults to
            DEVICE.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        np.ndarray: The segmentation masks for the video.
    """
    # send model to device
    model.to(device)
    # set model to evaluation mode
    model.eval()
    if verbose:
        print("Performing semantic segmentation on video with shape", video.shape)
    # get the preprocessing function for the model
    transforms = weights.transforms()
    # preprocess each frame and group them into batches for the model
    class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
    masks = []
    if verbose:
        print("Processing video in batches of size", batch_size, "with stride", stride)
    for i in range(0, len(video), stride):
        # check if there are enough frames left to fill a batch
        if i + batch_size > len(video):
            # if not, use the remaining frames
            batch_size = len(video) - i
        # preprocess next batch of video frames
        if verbose:
            print(f"Processing frames {i} to {i + batch_size}")
        frames = torch.stack(
            [transforms(frame.permute(2, 0, 1)) for frame in video[i : i + batch_size]]
        )
        predictions = model(frames.to(device))["out"]
        # predictions is a tensor of shape (batch_size, num_classes, H, W)
        # normalize the predictions
        normalized_masks = predictions.softmax(dim=1)
        # add the mask for the cow class to the list of masks
        for normalized_mask in normalized_masks:
            masks.append(normalized_mask[class_to_idx["cow"]].numpy(force=True))
        # masks.append(normalized_masks[0, class_to_idx["cow"]].cpu())
        # Force garbage collection to free up memory
        del frames
        del predictions
        del normalized_masks
        gc.collect()
        torch.cuda.empty_cache()
    if verbose:
        print("Done processing video.")
    return torch.tensor(np.stack(masks))


def segment(
    video: torch.Tensor,
    threshold=0.0,
    gaussian_blur=False,
    kernel_size=3,
    sigma=(0.1, 2.0),
    batch_size=16,
    stride=8,
    device=DEVICE,
    verbose=False,
) -> torch.Tensor:
    """Segment the video using a segmentation model."""
    # perform semantic segmentation on video
    semantic_masks = semantic_segment(
        video=video,
        model=torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        ).to(device),
        weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT,
        batch_size=batch_size,
        stride=stride,
        device=device,
        verbose=verbose,
    )
    # free up memory
    gc.collect()
    torch.cuda.empty_cache()
    # perform instance segmentation on video
    instance_masks = instance_segment(
        video=video,
        model=torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        ).to(device),
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        threshold=0.0,
        batch_size=batch_size,
        stride=stride,
        device=device,
        verbose=verbose,
    )
    # free up memory
    gc.collect()
    torch.cuda.empty_cache()
    # binarize the semantic masks
    semantic_masks = semantic_masks > threshold
    # binarize the instance masks
    instance_masks = instance_masks > threshold
    # if gaussian_blur: apply gaussian blur to the masks and binarize them
    if gaussian_blur:
        gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        semantic_masks = gaussian_blur(semantic_masks) > 0.0
        instance_masks = gaussian_blur(instance_masks) > 0.0
    # verify shapes of masks
    # if shape different than video, resize masks
    if (
        semantic_masks.shape[1] != video.shape[1]
        or semantic_masks.shape[2] != video.shape[2]
    ):
        resize = Resize(size=(video.shape[1], video.shape[2]))
        semantic_masks = resize(semantic_masks)
    if (
        instance_masks.shape[1] != video.shape[1]
        or instance_masks.shape[2] != video.shape[2]
    ):
        resize = Resize(size=(video.shape[1], video.shape[2]))
        instance_masks = resize(instance_masks)
    # combine the semantic and instance masks with logical AND
    masks = semantic_masks & instance_masks
    # apply mask to video
    masked_video = video * masks.unsqueeze(1).float()
    return masked_video


def sliding_window(
    video: torch.Tensor, length: int = 16, stride: int = 8
) -> torch.Tensor:
    """Apply sliding window to the video."""
    raise NotImplementedError("Sliding window not implemented yet.")


# %%
# Load video with PyTorch
video, _, info = torchvision.io.read_video("videos/video1_01.mp4", pts_unit="sec")

# %% [markdown]
# # Test semantic segmentation

# %%
# Load model and weights
weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
model.to(DEVICE)
model.eval()

# %%
semantic_masks = semantic_segment(
    video,
    model,
    weights,
    batch_size=1,
    stride=1,
    device=DEVICE,
    verbose=True,
)

# delete model and weights to free up memory
del model
del weights
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# # Test instance segmentation

# %%
# Load model and weights
weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
model.to(DEVICE)
model.eval()

# %%
instance_masks = instance_segment(
    video,
    model,
    weights,
    threshold=0.5,
    batch_size=1,
    stride=1,
    device=DEVICE,
    verbose=True,
)

# delete model and weights to free up memory
del model
del weights
gc.collect()
torch.cuda.empty_cache()

# %%
# Test segmentation
masked_video = segment(
    video,
    threshold=0.5,
    gaussian_blur=True,
    kernel_size=3,
    sigma=(0.1, 2.0),
    batch_size=1,
    stride=1,
    device=DEVICE,
    verbose=True,
)

# %%
# Save video
torchvision.io.write_video(
    "videos/video1_01_masked.mp4",
    masked_video,
    fps=info["video_fps"],
    video_codec="libx264",
    audio_codec="aac",
)

# %%
