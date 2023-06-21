"""Preprocess data for training and testing."""

# %% [markdown]
# # Imports

# %%
import gc
import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import GaussianBlur, Grayscale, Resize

import config

# %% [markdown]
# # Constants

# %%
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# %% [markdown]
# # Dataset

# %%


class VideoDataset(Dataset):
    """Dataset for video data."""

    def __init__(
        self, annotations_file, data_dir, fps=7.5, transform=None, target_transform=None
    ):
        """Initialize the dataset.

        Args:
            annotations_file (str): Path to the annotation file.
            data_dir (str): Path to the data directory.
            fps (float, optional): Frames per second. Defaults to 7.5.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the
                target of a sample.
        """
        if isinstance(annotations_file, str):
            # if csv file, load it
            if annotations_file.endswith(".csv"):
                self.annotations = pd.read_csv(annotations_file, index_col=0)
            # if xlsx file, load the first sheet
            elif annotations_file.endswith(".xlsx"):
                self.annotations = pd.read_excel(
                    annotations_file, header=0, index_col=0
                )
            else:
                raise ValueError("Annotations file must be a csv or xlsx file.")
        elif isinstance(annotations_file, pd.DataFrame):
            # if dataframe, use it
            self.annotations = annotations_file
        else:
            raise ValueError(
                "Annotations file must be a csv or xlsx file path or a dataframe."
            )
        # convert label column to float
        self.annotations.loc[:, config.LABEL_COLUMN] = self.annotations.loc[
            :, config.LABEL_COLUMN
        ].astype(float)
        # Remove rows with missing values
        self.annotations = self.annotations.loc[
            (
                self.annotations.loc[:, config.LABEL_COLUMN].notna()
                & self.annotations.loc[:, config.LABEL_COLUMN]
                > 0
            ),
            :,
        ]
        self.data_dir = data_dir
        # List videos
        self.file_list = [
            file
            for file in os.listdir(data_dir)
            if file.endswith(".mp4") and file.split(".")[0] in self.annotations.index
        ]
        self.fps = fps
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.annotations)

    def __getitem__(self, idx):  # -> tuple[torch.Tensor, torch.float]:
        """Return the idx-th element of the dataset.

        Args:
            idx (int): Index of the element to return.

        Returns:
            tuple[torch.Tensor, torch.float]: Video and label.
        """
        # file name in index
        video_name = self.annotations.index[idx] + ".mp4"
        video, _, info = torchvision.io.read_video(
            os.path.join(self.data_dir, video_name),
            pts_unit="sec",
            output_format="TCHW",
        )
        # Resample video fps
        video = resample_video(video=video, fps=info["video_fps"], target_fps=self.fps)
        if self.transform:
            video = self.transform(video)
        label = self.annotations.loc[:, config.LABEL_COLUMN].iloc[idx]
        if self.target_transform:
            label = self.target_transform(label)
        return video, label


# %% [markdown]
# # Transforms

# %%


def expand_video_into_batches(
    video: torch.Tensor,
    batch_size: int = 16,
    stride: int = 8,
    first_only: bool = False,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    """Expand a video with a single label into batches of frames.

    Args:
        video: A tensor of shape (channels, frames, height, width).
        batch_size: The number of frames in each batch.
        stride: The number of frames to skip between batches.
        device: The device to send the batches to.

    Returns:
        A tensor of shape (batches, batch_size, channels, height, width).
    """
    # create a list of batches
    batches = []
    # iterate over the video in batches of size `batch_size`
    for i in range(0, video.shape[1], stride):
        # check if there are enough frames left to fill a batch
        if i + batch_size > video.shape[1]:
            # if not, skip
            break
        # add the next batch of frames to the list
        batches.append(video[:, i : i + batch_size, :, :])
        # if only the first batch is needed, stop
        if first_only:
            break
    # stack the batches into a tensor
    batches = torch.stack(batches)
    # send the tensor to the device
    # batches = batches.to(device)
    return batches


def expand_label(
    label: float, number_of_batches: int, device: torch.device = DEVICE
) -> torch.Tensor:
    """Expand a label into a tensor of shape (batches, 1).

    Args:
        label: A tensor of shape (1).
        number_of_batches: The number of batches to expand the label into.
        device: The device to send the batches to.

    Returns:
        A tensor of shape (batches, 1).
    """
    # create a tensor of shape (batches, 1) filled with the label
    label_tensor = torch.full((number_of_batches, 1), label)
    # send the tensor to the device
    # label_tensor = label_tensor.to(device)
    return label_tensor


def resample_video(
    video: torch.Tensor,
    fps: float,
    target_fps: float,
    device: str | torch.device = DEVICE,
) -> torch.Tensor:
    """Resample a video to a target fps.

    Args:
        video: A tensor of shape (T, C, H, W).
        fps: The fps of the video.
        target_fps: The target fps.
        device: The device to send the batches to.

    Returns:
        A tensor of shape (T', C, H, W).
    """
    # calculate the number of frames in the video
    number_of_frames = video.shape[0]
    # calculate the duration of the video
    duration = number_of_frames / fps
    # calculate the number of frames in the resampled video
    number_of_frames_resampled = int(duration * target_fps)
    # manually resample the video
    # create a list of frames to sample
    frames_to_sample = torch.linspace(
        0, number_of_frames - 1, number_of_frames_resampled
    )
    # round the frames to sample
    frames_to_sample = torch.round(frames_to_sample).long()
    # sample the frames
    video_resampled = video[frames_to_sample, :, :, :]
    # send the tensor to the device
    # video_resampled = video_resampled.to(device)
    return video_resampled


# %% [markdown]
# # Segmentation functions

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
    """Segment the video using an instance segmentation model.

    Args:
        video: A tensor of shape (frames, height, width, channels).
        model: The model to use for segmentation.
        weights: The weights to use for segmentation.
        threshold: The threshold to use for segmentation.
        batch_size: The number of frames to process in each batch.
        stride: The number of frames to skip between batches.
        device: The device to send the batches to.
        verbose: Whether to print progress.

    Returns:
        A mask of the same shape as the video.
    """
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
            areas: torch.Tensor = []  # type: ignore
            for box in output["boxes"][cow_idxs]:
                areas.append((box[2] - box[0]) * (box[3] - box[1]))  # type: ignore
            areas = torch.tensor(areas)
            # get the mask with the highest bbox area
            # if there are no masks, append a max with ones
            if len(areas) == 0:
                mask = torch.ones(1, video.shape[1], video.shape[2])
                score = 0.0
            else:
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
    return torch.tensor(np.stack(masks)).detach().to(device)


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
    return torch.tensor(np.stack(masks)).unsqueeze(1).detach().to(device)


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
    """Segment the video using a segmentation model.

    Args:
        video (torch.Tensor): The video to segment.
        threshold (float, optional): The minimum score for a mask to be included in the
            output. Defaults to 0.0.
        gaussian_blur (bool, optional): Whether to apply a gaussian blur to the masks.
            Defaults to False.
        kernel_size (int, optional): The size of the kernel to use for the gaussian
            blur. Defaults to 3.
        sigma (tuple, optional): The range of standard deviations to use for the gaussian
            blur. Defaults to (0.1, 2.0).
        batch_size (int, optional): The number of frames to process at a time. Defaults to
            16.
        stride (int, optional): The number of frames to skip between batches. Defaults to
            8.
        device (torch.device, optional): The device to use for processing. Defaults to
            DEVICE.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        torch.Tensor: The segmentation masks for the video.
    """
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
    masked_video = (video.permute(0, 3, 1, 2) * masks).permute(0, 2, 3, 1)
    return masked_video.detach().to(device)


# %%
if __name__ == "__main__":
    # For each video in data directory, load, segment, and save
    for video_path in glob.glob(f"{config.DATA_DIRECTORY}*.mp4"):
        # get video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        # Load video with PyTorch
        video, _, info = torchvision.io.read_video(
            video_path, pts_unit="sec", output_format="THWC"
        )
        # Resize video to 480p
        video = Resize(size=(480, 640))(video.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # if masked video already exists, skip
        if os.path.exists(f"{config.DATA_DIRECTORY}/masked/{video_name}.mp4"):
            continue
        # Segment video
        masked_video = (
            segment(
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
            .detach()
            .cpu()
        )
        # Save video
        torchvision.io.write_video(
            f"{config.DATA_DIRECTORY}/masked/{video_name}.mp4",
            masked_video,
            fps=info["video_fps"],
            video_codec="libx264",
            audio_codec="aac",
        )
        # Force garbage collection to free up memory
        del video
        del masked_video
        gc.collect()
        torch.cuda.empty_cache()

# %%
# # # Test semantic segmentation
# # %%
# # Load model and weights
# weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
# model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
# model.to(DEVICE)
# model.eval()

# # %%
# semantic_masks = semantic_segment(
#     video,
#     model,
#     weights,
#     batch_size=1,
#     stride=1,
#     device=DEVICE,
#     verbose=True,
# )
# # delete model and weights to free up memory
# del model
# del weights
# gc.collect()
# torch.cuda.empty_cache()
# # # Test instance segmentation
# # %%
# # Load model and weights
# weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
# model.to(DEVICE)
# model.eval()
# # %%
# instance_masks = instance_segment(
#     video,
#     model,
#     weights,
#     threshold=0.5,
#     batch_size=1,
#     stride=1,
#     device=DEVICE,
#     verbose=True,
# )
# # delete model and weights to free up memory
# del model
# del weights
# gc.collect()
# torch.cuda.empty_cache()

# %%
