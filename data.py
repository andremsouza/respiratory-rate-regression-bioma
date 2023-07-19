"""Preprocess data for training and testing."""

# %% [markdown]
# # Imports

# %%
import gc
import glob
import os
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import GaussianBlur, Resize

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
        self,
        annotations_file: str,
        data_dir: str,
        fps: float = 7.5,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        filter_ids: list[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            annotations_file (str): Path to the annotation file.
            data_dir (str): Path to the data directory.
            fps (float, optional): Frames per second. Defaults to 7.5.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the
                target of a sample.
            filter_ids (list, optional): List of ids to filter. Defaults to None.
        """
        self.annotations_file = annotations_file
        self.label_column = config.LABEL_COLUMN
        if isinstance(annotations_file, str):
            # if csv file, load it
            if annotations_file.endswith(".csv"):
                self.annotations = pd.read_csv(annotations_file, index_col=0)
            # if xlsx file, load the first sheet
            elif annotations_file.endswith(".xlsx"):
                self.annotations = pd.read_excel(
                    annotations_file, header=0, index_col=0
                )
            # if json file, load it and preprocess (Label Studio)
            elif annotations_file.endswith(".json"):
                self.label_column = "breathing_rate"
                self.annotations = load_label_studio_annotations(
                    annotations_file=annotations_file,
                    label_column="value",
                    filter_ids=filter_ids,
                )
            else:
                raise ValueError("Annotations file must be a csv, xlsx or json file.")
        elif isinstance(annotations_file, pd.DataFrame):
            # if dataframe, use it
            self.annotations = annotations_file
        else:
            raise ValueError(
                "Annotations file must be a csv, xlsx or json file path or a dataframe."
            )
        self.data_dir = data_dir
        if annotations_file.endswith(".xlsx") or annotations_file.endswith(".csv"):
            # convert label column to float
            self.annotations.loc[:, self.label_column] = self.annotations.loc[
                :, self.label_column
            ].astype(float)
            # Remove rows with missing values
            self.annotations = self.annotations.loc[
                (
                    self.annotations.loc[:, self.label_column].notna()
                    & self.annotations.loc[:, self.label_column]
                    > 0
                ),
                :,
            ]
            # List videos
            self.file_list = [
                file
                for file in os.listdir(data_dir)
                if file.endswith(".mp4")
                and file.split(".")[0] in self.annotations.index
            ]
            assert len(self.file_list) == len(self.annotations)
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
        if self.annotations_file.endswith(".xlsx") or self.annotations_file.endswith(
            ".csv"
        ):
            # Get video name
            video_name = self.annotations.index[idx] + ".mp4"
            # Get label
            label = self.annotations.loc[:, self.label_column].iloc[idx]
            # Read video
            video, _, info = torchvision.io.read_video(
                os.path.join(self.data_dir, video_name),
                pts_unit="sec",
                output_format="TCHW",
            )
        elif self.annotations_file.endswith(".json"):
            # Get video name
            video_name = self.annotations.loc[idx, "file_upload"]
            # Get label
            label = self.annotations.loc[idx, self.label_column]
            # Search for video file suffix in data_dir
            video_name = [
                file for file in os.listdir(self.data_dir) if video_name.endswith(file)
            ][0]
            # Read video
            video, _, info = torchvision.io.read_video(
                os.path.join(self.data_dir, video_name),
                pts_unit="sec",
                output_format="TCHW",
            )
            # Bounding box transform (if available)
            try:
                bboxes_sequence = [
                    x
                    for x in self.annotations.loc[idx, "value"]
                    if x["type"] == "videorectangle"
                ][0]["value"]["sequence"]

            except (IndexError, KeyError, ValueError):
                pass
            else:
                video = bounding_box_transform(
                    video, bboxes_sequence, fps=info["video_fps"], percent=True
                )
            # Trim video time (if available)
            try:
                start_time = self.annotations.loc[idx, "segment.value.start"]
                end_time = self.annotations.loc[idx, "segment.value.end"]
            except KeyError:
                pass
            else:
                start_frame = int(start_time * info["video_fps"])
                end_frame = int(end_time * info["video_fps"])
                video = video[start_frame:end_frame, :, :, :]

        else:
            raise ValueError("Annotations file must be a csv, xlsx or json file.")
        # Resample video fps
        if self.fps is not None and self.fps < info["video_fps"]:
            video = resample_video(
                video=video, fps=info["video_fps"], target_fps=self.fps
            )
        if self.transform:
            video = self.transform(video)
        if self.target_transform:
            label = self.target_transform(label)
        return video, label


# %% [markdown]
# # Data loaders/readers

# %%


def load_label_studio_annotations(
    annotations_file: str | dict,
    label_column: str = "value",
    filter_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Load annotations from a Label Studio JSON file.

    Args:
        annotations_file (str | dict): Path to the annotations file or dictionary of
            annotations.
        data_dir (str): Path to the directory containing the videos.
        label_column (str, optional): Name of the column containing the labels.
            Defaults to "value".
        filter_ids (list[str] | None, optional): List of ids to keep. Defaults to None.

    Returns:
        pd.DataFrame: Annotations.
    """
    # Load annotations
    if isinstance(annotations_file, str):
        annotations = pd.read_json(annotations_file)
    elif isinstance(annotations_file, dict):
        annotations = pd.DataFrame(annotations_file)
    else:
        raise ValueError(
            "Annotations file must be a json file path or a dictionary of annotations."
        )
    # Filter annotations
    if filter_ids is not None:
        annotations = annotations.loc[annotations["id"].isin(filter_ids), :]
    # Get annotation values of each annotation and append as columns
    # (rows when more than one annotation)
    annotations = pd.concat(
        [
            annotations,
            pd.json_normalize(annotations.set_index("id")["annotations"])
            .set_index(annotations.index)
            .add_prefix("annotations."),
        ],
        axis=1,
    )
    # Drop annotations column
    annotations = annotations.drop(columns=["annotations"])
    # Melt annotations columns into rows
    annotations = annotations.melt(
        id_vars=[
            col for col in annotations.columns if not col.startswith("annotations.")
        ],
        value_vars=[
            col for col in annotations.columns if col.startswith("annotations.")
        ],
        var_name="annotation",
        value_name="value",
    )
    # Remove temporary prefix
    annotations["annotation"] = annotations["annotation"].str.replace(
        "annotations.", ""
    )
    # Get annotation values of each annotation and append as columns
    # (rows when more than one annotation)
    annotations = pd.concat(
        [
            annotations,
            pd.json_normalize(annotations.set_index("id")["value"])
            .set_index(annotations.index)
            .add_prefix("annotation."),
        ],
        axis=1,
    )
    # Drop value column
    annotations = annotations.drop(columns=["value"])
    # Melt annotation columns into rows
    annotations = annotations.melt(
        id_vars=[col for col in annotations.columns if not col.endswith("result")],
        value_vars=[col for col in annotations.columns if col.endswith("result")],
        var_name="annotation_value",
        value_name=label_column,
    )
    # Remove temporary prefix
    annotations["annotation_value"] = annotations["annotation_value"].str.replace(
        "annotation.", ""
    )
    # For each row, get video segments and append as columns (rows when more than one segment)
    for i, row in annotations.iterrows():
        new_rows = []
        segments = pd.json_normalize(
            [
                video_segment
                for video_segment in row["value"]
                if video_segment["type"] == "labels"
            ]
        ).add_prefix("segment.")
        # concatenate each segment as a row to the dataframe
        # keep row's columns and add segment's columns
        for j, video_segment in segments.iterrows():
            new_rows.append(pd.concat([row, video_segment], axis=0))
        # remove row
        annotations = annotations.drop(i)
        # concatenate new rows
        annotations = pd.concat([annotations, pd.DataFrame(new_rows)], axis=0)
    # Reset index
    annotations = annotations.reset_index(drop=True)
    # For each row, get breathing rate
    for i, row in annotations.iterrows():
        breathing_rate: float = [x for x in row["value"] if x["type"] == "number"][0][
            "value"
        ]["number"]
        annotations.loc[i, "breathing_rate"] = breathing_rate
    # Filter rows with no breathing rate (0)
    annotations = annotations.loc[annotations["breathing_rate"] > 0, :]
    # Reset index
    annotations = annotations.reset_index(drop=True)

    return annotations


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
    batches: list[torch.Tensor] | torch.Tensor = []
    # iterate over the video in batches of size `batch_size`
    for i in range(0, video.shape[1], stride):
        # check if there are enough frames left to fill a batch
        if i + batch_size > video.shape[1]:
            # if not, skip
            break
        # add the next batch of frames to the list
        batches.append(video[:, i : i + batch_size, :, :])  # type: ignore
        # if only the first batch is needed, stop
        if first_only:
            break
    # stack the batches into a tensor
    batches = torch.stack(batches)  # type: ignore
    # send the tensor to the device
    # batches = batches.to(device)
    return batches


def expand_label(
    label: float, number_of_batches: int, device: str | torch.device = DEVICE
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


def bounding_box_transform(
    video: torch.Tensor,
    bboxes: list[dict],
    interpolation: str = "linear",
    fps: float | None = None,
    percent: bool = False,
) -> torch.Tensor:
    """Transform the video to a bounding box.

    Given a list of bounding boxes, transform the video to the bounding box.
    Each bounding box is a dictionary with the following keys:
        - "x": The x coordinate of the top left corner of the bounding box.
        - "y": The y coordinate of the top left corner of the bounding box.
        - "width": The width of the bounding box.
        - "height": The height of the bounding box.
        - "time": The time of the bounding box in seconds.
        - "frame": The frame of the bounding box in the video.
        - "rotation": The rotation of the bounding box in degrees.

    When multiple bounding boxes are provided, we interpolate between them. For example,
    if we have bounding boxes at 0.0s and 1.0s, we will transform the video to the first
    bounding box at 0.0s, the second bounding box at 1.0s, and the intermediate bounding
    for each frame in between.

    Args:
        video (torch.Tensor): The video to transform.
        bboxes (list[dict]): The list of bounding boxes.
        interpolation (int, optional): The interpolation method to use. Defaults to
            "linear".
        fps (float, optional): The frames per second of the video. Defaults to None.
        percent (bool, optional): Whether the bounding box coordinates are in percent.

    Returns:
        torch.Tensor: The transformed video.
    """
    # interpolate bounding boxes for each frame
    bboxes = interpolate_bboxes(
        bboxes, video.shape[0], interpolation=interpolation, fps=fps
    )
    # transform video to bounding box
    # each frame has a different bounding box
    assert video.shape[0] == len(bboxes)
    # create empty video
    transformed_video = torch.zeros_like(video)
    # for each frame
    for frame, bbox in enumerate(bboxes):
        # transform frame to bounding box
        transformed_video[frame] = transform_to_bbox(
            frame=video[frame],
            x=bbox["x"],
            y=bbox["y"],
            width=bbox["width"],
            height=bbox["height"],
            rotation=bbox["rotation"],
            percent=percent,
        )
    return transformed_video


def interpolate_bboxes(
    bboxes: list[dict],
    num_frames: int,
    interpolation: str = "linear",
    fps: float | None = None,
) -> list[dict]:
    """Interpolate bounding boxes.

    Given a list of bounding boxes, interpolate between them. For example, if we have
    bounding boxes at 0.0s and 1.0s, we will interpolate the bounding boxes for each
    frame in between.

    Given a list of bounding boxes, transform the video to the bounding box.
    Each bounding box is a dictionary with the following keys:
        - "x": The x coordinate of the top left corner of the bounding box.
        - "y": The y coordinate of the top left corner of the bounding box.
        - "width": The width of the bounding box.
        - "height": The height of the bounding box.
        - "time": The time of the bounding box in seconds.
        - "frame": The frame of the bounding box in the video.
        - "rotation": The rotation of the bounding box in degrees.

    Args:
        bboxes (list[dict]): The list of bounding boxes.
        num_frames (int): The number of frames in the video.
        interpolation (int, optional): The interpolation method to use. Defaults to
            "linear".
        fps (float, optional): The frames per second of the video. Defaults to None.

    Returns:
        list[dict]: The interpolated bounding boxes.
    """
    # if no bounding boxes, return empty list
    if len(bboxes) == 0:
        return []
    # if only one bounding box, return the same bounding box for each frame
    if len(bboxes) == 1:
        return [bboxes[0]] * num_frames
    # if interpolation is linear, interpolate linearly
    if interpolation == "linear":
        # ensure bounding boxes are sorted by time
        bboxes = sorted(bboxes, key=lambda x: x["time"])
        # for each bbox pair, interpolate between them
        # thus generating a bbox for each frame
        interpolated_bboxes: list[dict] = []
        for i in range(len(bboxes) - 1):
            start_bbox: dict = bboxes[i]
            end_bbox: dict = bboxes[i + 1]
            # get the start and end times
            start_time: float = start_bbox["time"]
            end_time: float = end_bbox["time"]
            # get the start and end frames
            if fps is not None:
                # if fps is provided, use it to calculate the start and end frames
                start_frame = int(start_time * fps) + 1
                end_frame = int(end_time * fps) + 1
            else:
                # otherwise, use the start and end frames provided
                start_frame: int = start_bbox["frame"]
                end_frame: int = end_bbox["frame"]
            # get the start and end x coordinates
            start_x: float = start_bbox["x"]
            end_x: float = end_bbox["x"]
            # get the start and end y coordinates
            start_y: float = start_bbox["y"]
            end_y: float = end_bbox["y"]
            # get the start and end width
            start_width: float = start_bbox["width"]
            end_width: float = end_bbox["width"]
            # get the start and end height
            start_height: float = start_bbox["height"]
            end_height: float = end_bbox["height"]
            # get the start and end rotation
            start_rotation: float = start_bbox["rotation"]
            end_rotation: float = end_bbox["rotation"]

            # interpolate between the start and end bounding boxes
            # with numpy
            # get the times for each frame
            times: np.ndarray = np.linspace(
                start=start_time, stop=end_time, num=end_frame - start_frame
            )
            # get the x coordinates for each frame
            xs: np.ndarray = np.linspace(
                start=start_x, stop=end_x, num=end_frame - start_frame
            )
            # get the y coordinates for each frame
            ys: np.ndarray = np.linspace(
                start=start_y, stop=end_y, num=end_frame - start_frame
            )
            # get the widths for each frame
            widths: np.ndarray = np.linspace(
                start=start_width, stop=end_width, num=end_frame - start_frame
            )
            # get the heights for each frame
            heights: np.ndarray = np.linspace(
                start=start_height, stop=end_height, num=end_frame - start_frame
            )
            # get the rotations for each frame
            rotations: np.ndarray = np.linspace(
                start=start_rotation, stop=end_rotation, num=end_frame - start_frame
            )

            # append the interpolated bounding boxes
            interpolated_bboxes += [
                {
                    "x": x,
                    "y": y,
                    "time": time,
                    "frame": frame,
                    "width": width,
                    "height": height,
                    "rotation": rotation,
                }
                for x, y, width, height, time, frame, rotation in zip(
                    xs,
                    ys,
                    widths,
                    heights,
                    times,
                    range(start_frame, end_frame),
                    rotations,
                )
            ]
        # append the last bounding box to remaining frames, adjusting the frame number
        last_bbox_frame: int = interpolated_bboxes[-1]["frame"]
        while len(interpolated_bboxes) < num_frames:
            interpolated_bboxes.append(
                {
                    "x": bboxes[-1]["x"],
                    "y": bboxes[-1]["y"],
                    "time": bboxes[-1]["time"],  # TODO: calculate time
                    "frame": last_bbox_frame + 1,
                    "width": bboxes[-1]["width"],
                    "height": bboxes[-1]["height"],
                    "rotation": bboxes[-1]["rotation"],
                }
            )
            last_bbox_frame += 1
        # Raise exception if the number of interpolated bounding boxes is not equal to
        # the number of frames
        assert len(interpolated_bboxes) == num_frames
        # Raise exception if the interpolated bounding boxes are not sorted by frame
        assert all(
            interpolated_bboxes[i]["frame"] <= interpolated_bboxes[i + 1]["frame"]
            for i in range(len(interpolated_bboxes) - 1)
        )
        # return the interpolated bounding boxes
        return interpolated_bboxes
    raise NotImplementedError


def transform_to_bbox(
    frame: torch.Tensor,
    x: float,
    y: float,
    width: float,
    height: float,
    rotation: float,
    interpolation: str = "bilinear",
    percent: bool = False,
) -> torch.Tensor:
    """Transforms a frame to a bounding box.

    Args:
        frame (torch.Tensor): The frame to transform.
        x (float): The x coordinate of the bounding box.
        y (float): The y coordinate of the bounding box.
        width (float): The width of the bounding box.
        height (float): The height of the bounding box.
        rotation (float): The rotation of the bounding box in degrees.
        interpolation (str, optional): The interpolation method to use. Defaults to
            "bilinear".
        percent (bool, optional): Whether the x, y, width, and height are in percent.

    Returns:
        torch.Tensor: The transformed frame.
    """
    # get original frame size
    original_height, original_width = frame.shape[-2:]
    # convert percent to pixels
    if percent:
        x *= original_width / 100
        y *= original_height / 100
        width *= original_width / 100
        height *= original_height / 100
    # rotate
    frame = torchvision.transforms.functional.rotate(
        frame,
        angle=rotation,
        interpolation={
            "bilinear": torchvision.transforms.functional.InterpolationMode.BILINEAR,
            "nearest": torchvision.transforms.functional.InterpolationMode.NEAREST,
        }[interpolation],
    )
    # crop
    frame = torchvision.transforms.functional.crop(
        frame,
        top=int(y),
        left=int(x),
        height=int(height),
        width=int(width),
    )
    # resize
    frame = torchvision.transforms.functional.resize(
        frame,
        size=(original_height, original_width),
        interpolation={
            "bilinear": torchvision.transforms.functional.InterpolationMode.BILINEAR,
            "nearest": torchvision.transforms.functional.InterpolationMode.NEAREST,
        }[interpolation],
        antialias=True,
    )
    # return the transformed frame
    return frame


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
