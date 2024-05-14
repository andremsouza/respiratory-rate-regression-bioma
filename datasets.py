"""Datasets for video models."""

# %% [markdown]
# # Imports

# %%
import math
import os
from typing import Any, Callable
import warnings

import cv2
import dotenv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_video

from label_studio.api import LabelStudioAPI

# %% [markdown]
# # Constants

# %%
# Load environment variables
dotenv.load_dotenv(verbose=True, override=True)
LABEL_STUDIO_URL: str = str(os.getenv("LABEL_STUDIO_URL", "http://localhost:8080"))
LABEL_STUDIO_API_KEY: str = str(os.getenv("LABEL_STUDIO_API_KEY", ""))
LABEL_STUDIO_CONTAINER_ID: str = str(os.getenv("LABEL_STUDIO_CONTAINER_ID", None))
LABEL_STUDIO_CONTAINER_DATA_DIR: str = str(
    os.getenv("LABEL_STUDIO_CONTAINER_DATA_DIR", None)
)
LABEL_STUDIO_DOWNLOAD_DIR: str = str(os.getenv("LABEL_STUDIO_DOWNLOAD_DIR", None))
LABEL_STUDIO_PROJECT_ID: int = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

# %% [markdown]
# # Classes

# %%


class VideoDataset(Dataset):
    """Video dataset.

    Args:
        url (str): Label Studio URL.
        api_key (str): Label Studio API key.
        project_id (int): Project ID.
        data_dir (str, optional): Data directory. Defaults to None.
        container_id (str, optional): Container ID. Defaults to None.
        container_data_dir (str, optional): Container data directory. Defaults to None.
        fps (float, optional): Frames per second. Defaults to 5.
        sample_size (int, optional): Sample size. Defaults to 16.
        hop_length (int, optional): Hop length. Defaults to 8.
        filter_task_ids (list[int], optional): Task IDs to filter. Defaults to None.
        bbox_transform (bool, optional): Whether to transform the video to the bounding
            box. Defaults to False.
        download_videos (bool, optional): Whether to download videos. Defaults to False.
        download_videos_overwrite (bool, optional): Whether to overwrite existing videos.
            Defaults to False.
        classification (bool, optional): Whether to use classification. Defaults to
            False.
        transform (Callable, optional): Transform to apply to samples. Defaults to None.
        target_transform (Callable, optional): Transform to apply to targets. Defaults
            to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Raises:
        ValueError: If project_id is invalid.
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        project_id: int,
        data_dir: str | None = None,
        container_id: str | None = None,
        container_data_dir: str | None = None,
        fps: float = 5,
        sample_size: int = 16,
        hop_length: int = 8,
        filter_task_ids: list[int] | None = None,
        bbox_transform: bool = False,
        bbox_transform_corners: bool = False,
        download_videos: bool = False,
        download_videos_overwrite: bool = False,
        classification: bool = False,
        prune_invalid: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            url (str): Label Studio URL.
            api_key (str): Label Studio API key.
            project_id (int): Project ID.
            data_dir (str, optional): Data directory. Defaults to None.
            container_id (str, optional): Container ID. Defaults to None.
            container_data_dir (str, optional): Container data directory. Defaults to
                None.
            fps (float, optional): Frames per second. Defaults to 5.
            sample_size (int, optional): Sample size. Defaults to 16.
            hop_length (int, optional): Hop length. Defaults to 8.
            filter_task_ids (list[int], optional): Task IDs to filter. Defaults to None.
            bbox_transform (bool, optional): Whether to transform the video to the
                bounding box. Defaults to False.
            download_videos (bool, optional): Whether to download videos. Defaults to
                False.
            download_videos_overwrite (bool, optional): Whether to overwrite existing
                videos. Defaults to False.
            classification (bool, optional): Whether to use classification. Defaults to
                False.
            transform (Callable, optional): Transform to apply to samples. Defaults to
                None.
            target_transform (Callable, optional): Transform to apply to targets.
                Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to
                False.

        Raises:
            ValueError: If project_id is invalid.
        """
        super().__init__()
        self.verbose = verbose
        # Initialize Label Studio API
        self.api = LabelStudioAPI(
            url=url,
            api_key=api_key,
            data_dir=data_dir,
            container_id=container_id,
            container_data_dir=container_data_dir,
        )
        # Check project_id
        self._check_project_id(project_id=project_id, verbose=self.verbose)
        self.project_id: int = project_id
        # If data_dir is None, use current working directory
        # If data_dir does not exist, create it
        self.data_dir: str = (
            data_dir if data_dir is not None else os.path.join(os.getcwd(), "data/")
        )
        self.fps: float = fps
        self.sample_size: int = sample_size
        self.hop_length: int = hop_length
        self.filter_ids: list[int] | None = filter_task_ids
        self.bbox_transform: bool = bbox_transform
        self.bbox_transform_corners: bool = bbox_transform_corners
        self.download_videos: bool = download_videos
        self.download_videos_overwrite: bool = download_videos_overwrite
        self.classification: bool = classification
        self.prune_invalid: bool = prune_invalid
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform
        # Load tasks
        self._load_annotations()
        # Process annotations
        self._process_annotations()
        # Prune invalid samples
        if self.prune_invalid:
            self._prune_invalid()

    def _check_project_id(self, project_id: int, verbose: bool = False) -> None:
        """Check if project_id is valid.

        Args:
            project_id (int): Project ID to check.

        Raises:
            ValueError: If project_id is invalid.
        """
        if int(project_id) not in self.api.list_projects()[1]:
            raise ValueError(
                f"Invalid project_id: {project_id}"
                + (
                    f"\n Available project_ids: {self.api.list_projects()[1]}"
                    if verbose
                    else "."
                )
            )

    def _load_annotations(self) -> None:
        """Load tasks from LabelStudio."""
        # Get tasks and annotations from LabelStudio
        self.annotations: pd.DataFrame = self.api.list_project_tasks(
            project_id=self.project_id,
            page_size=-1,
            only_labeled=True,
            to_dataframe=True,
        )[0]
        # Filter tasks with valid (non-empty) annotations
        self.annotations = self.annotations[
            self.annotations["annotation_value.result"].notna()
        ]
        # Filter ids if specified
        if self.filter_ids is not None:
            self.annotations = self.annotations[
                self.annotations["id"].isin(self.filter_ids)
            ]
        # Transform annotations data column
        self.annotations["data"] = self.annotations.apply(
            lambda x: str(x["data"]["video"]), axis=1
        )
        # Get task annotations
        # Create columns with object type
        self.annotations["breathing_rate"] = np.nan
        self.annotations["bbox_sequence"] = np.nan
        self.annotations["lado"] = np.nan
        self.annotations["segments"] = np.nan
        # Convert new columns to object type
        self.annotations["breathing_rate"] = self.annotations["breathing_rate"].astype(
            "object"
        )
        self.annotations["bbox_sequence"] = self.annotations["bbox_sequence"].astype(
            "object"
        )
        self.annotations["lado"] = self.annotations["lado"].astype("object")
        self.annotations["segments"] = self.annotations["segments"].astype("object")
        for idx, row in self.annotations.iterrows():
            row_result = pd.json_normalize(row["annotation_value.result"])
            # Get original_length from result
            try:
                original_length: float | None = row_result["original_length"].max()
            except KeyError:
                original_length = None
            breathing_rate: float = 0.0
            try:
                if len(row_result[row_result["from_name"] == "breathingrate30s"]) > 0:
                    breathing_rate = row_result[
                        row_result["from_name"] == "breathingrate30s"
                    ].iloc[0]["value.number"]
            except KeyError:
                breathing_rate = 0.0
            # Get bbox sequence from result, if available
            bbox_sequence: list[dict] = []
            try:
                if len(row_result[row_result["from_name"] == "box"]) > 0:
                    bbox_sequence = row_result[row_result["from_name"] == "box"].iloc[
                        0
                    ]["value.sequence"]
            except KeyError:
                bbox_sequence = []
            # Get lado from result, if available
            lado: str = ""
            try:
                if len(row_result[row_result["from_name"] == "lado"]) > 0:
                    lado = row_result[row_result["from_name"] == "lado"].iloc[0][
                        "value.choices"
                    ][0]
            except KeyError:
                lado = ""
            # Get segments from result, if available
            segments: list[dict] = []
            try:
                for _, segment in row_result[
                    row_result["from_name"] == "tricks"
                ].iterrows():
                    segments.append(
                        {
                            "start": segment["value.start"],
                            "end": segment["value.end"],
                            "labels": segment["value.labels"],
                        }
                    )
            except KeyError:
                segments = []
            # Set original length in dataframe
            self.annotations.at[idx, "original_length"] = original_length
            # Set breathing rate, bbox sequence and segments in dataframe
            self.annotations.at[idx, "breathing_rate"] = breathing_rate
            self.annotations.at[idx, "bbox_sequence"] = (
                bbox_sequence if bbox_sequence else None
            )
            self.annotations.at[idx, "lado"] = lado
            # Interpolate bounding boxes
            if self.bbox_transform and bbox_sequence:
                # Get fps and number of frames with cv2
                filename: str = os.path.join(
                    self.data_dir, self.annotations.at[idx, "data"].split("/")[-1]
                )
                cap = cv2.VideoCapture(filename)
                video_fps: float = cap.get(cv2.CAP_PROP_FPS)
                frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                self.annotations.at[idx, "bbox_sequence"] = self._interpolate_bboxes(
                    bboxes=bbox_sequence,
                    num_frames=frame_count,
                    interpolation="linear",
                    fps=video_fps,
                )
            self.annotations.at[idx, "segments"] = segments if segments else None
        # Drop original result column
        self.annotations.drop(columns=["annotation_value.result"], inplace=True)
        # Sort by task id and annotation id
        self.annotations = self.annotations.sort_values(
            by=["id", "annotation_value.id"]
        )

    def _process_annotations(self) -> None:
        # TODO: Implement processing
        # Download videos
        if self.download_videos:
            # Create data directory if it does not exist
            os.makedirs(self.data_dir, exist_ok=True)
            self.annotations.apply(
                self._download_task_video,
                axis=1,
                overwrite=self.download_videos_overwrite,
            )
        # Convert annotations to segments
        segment_lists = self.annotations.apply(
            self._annotation_to_segments,
            axis=1,
        )
        # Merge segment lists into single list
        segments = []
        for segment_list in segment_lists:
            segments += segment_list
        # Convert segments to dataframe
        self.segments = pd.DataFrame(segments)
        # Convert segments to samples
        sample_lists = self.segments.apply(
            self._segment_to_samples,
            axis=1,
        )
        # Merge sample lists into single list
        samples = []
        for sample_list in sample_lists:
            samples += sample_list
        # Convert samples to dataframe
        self.samples = pd.DataFrame(samples)
        # raise NotImplementedError

    def _annotation_to_segments(self, annotation: pd.Series) -> list[dict]:
        """Convert annotation to segments.

        Args:
            annotation_id (int): Annotation ID.

        Returns:
            list[dict]: List of segments.
        """
        segment_samples: list[dict] = []  # Store segments in list of dicts
        # Get segments
        if annotation["segments"] is not None:
            segments = sorted(annotation["segments"], key=lambda x: x["start"])
        else:
            segments = []
        # For each segment, generate sample
        if len(segments) > 0:
            complement_ranges = []  # Calculate complement ranges
            for seg_idx, segment in enumerate(segments):
                if seg_idx == 0:
                    complement_ranges.append(
                        {"start": 0.0, "end": segment["start"], "labels": ["NOT OK"]}
                    )
                else:
                    complement_ranges.append(
                        {
                            "start": segments[seg_idx - 1]["end"],
                            "end": segment["start"],
                            "labels": ["NOT OK"],
                        }
                    )
            # Add last complement range
            complement_ranges.append(
                {
                    "start": segments[-1]["end"],
                    "end": annotation["original_length"],
                    "labels": ["NOT OK"],
                }
            )
            segments += complement_ranges  # Add complement ranges to segments
            # Remove segments with same start and end time
            segments = [
                segment for segment in segments if segment["start"] != segment["end"]
            ]
            # Sort segments by start time ascending
            segments = sorted(segments, key=lambda x: x["start"])
            # Generate samples
            for seg_idx, segment in enumerate(segments):
                # Only add segment if start and end times are different
                segment_samples.append(annotation.drop("segments").to_dict())
                segment_samples[-1].update(
                    {
                        "segment_start": segment["start"],
                        "segment_end": segment["end"],
                        "segment_id": seg_idx,
                        "segment_labels": segment["labels"],
                    }
                )
        else:
            # Use full video
            segment_samples.append(annotation.drop("segments").to_dict())
            segment_samples[-1].update(
                {
                    "segment_start": 0.0,
                    "segment_end": annotation["original_length"],
                    "segment_id": 0,
                    "segment_labels": ["NOT OK"],
                }
            )
        return segment_samples

    def _segment_to_samples(self, segment: pd.Series) -> list[dict]:
        """Convert segment to samples.

        Args:
            segment (pd.Series): Segment.

        Returns:
            list[dict]: List of samples.
        """
        samples: list[dict] = []
        # Get segment start and end times
        segment_start_time: float = segment["segment_start"]
        segment_end_time: float = segment["segment_end"]

        # Read video timestamps
        filename: str = os.path.join(self.data_dir, segment["data"].split("/")[-1])
        cap = cv2.VideoCapture(filename)
        video_fps: float = cap.get(cv2.CAP_PROP_FPS)
        frame_count: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        resample_rate: float = self.fps / video_fps
        sample_size_original: int = int(self.sample_size / resample_rate)
        hop_length_original: int = int(self.hop_length / resample_rate)
        # Check if segment start and end times are valid
        if np.isnan(segment_start_time) or np.isnan(segment_end_time):
            segment_start_time = 0.0
            segment_end_time = frame_count / video_fps
        # Calculate segment start and end frames
        segment_start_frame: int = int(segment_start_time * video_fps)
        segment_end_frame: int = int(segment_end_time * video_fps)
        # Calculate original start and end frames for each sample
        sample_start_frames = np.arange(
            segment_start_frame,
            segment_end_frame - sample_size_original,
            hop_length_original,
        )
        sample_end_frames = sample_start_frames + sample_size_original
        # Calculate original start and end times for each sample
        sample_start_times = sample_start_frames / video_fps
        sample_end_times = sample_end_frames / video_fps
        # Calculate segment start and end frames after resampling
        segment_start_frame_resampled: int = int(segment_start_frame * resample_rate)
        segment_end_frame_resampled: int = int(segment_end_frame * resample_rate)
        # Calculate resampled start and end frames for each sample
        sample_start_frames_resampled = np.arange(
            segment_start_frame_resampled,
            segment_end_frame_resampled - self.sample_size,
            self.hop_length,
        )
        sample_end_frames_resampled = sample_start_frames_resampled + self.sample_size
        # Get sample start and end times
        sample_start_times_resampled = sample_start_frames_resampled / self.fps
        sample_end_times_resampled = sample_end_frames_resampled / self.fps
        # Generate samples
        for sample_idx, (
            sample_start_frame,
            sample_end_frame,
            sample_start_frame_resampled,
            sample_end_frame_resampled,
            sample_start_time,
            sample_end_time,
            sample_start_time_resampled,
            sample_end_time_resampled,
        ) in enumerate(
            zip(
                sample_start_frames,
                sample_end_frames,
                sample_start_frames_resampled,
                sample_end_frames_resampled,
                sample_start_times,
                sample_end_times,
                sample_start_times_resampled,
                sample_end_times_resampled,
            )
        ):
            samples.append(
                {
                    "task_id": segment["id"],
                    "annotation_id": segment["annotation_value.id"],
                    "segment_id": segment["segment_id"],
                    "sample_id": sample_idx,
                    "data": filename,
                    "fps_target": self.fps,
                    "fps_original": video_fps,
                    "sample_size_target": self.sample_size,
                    "sample_size_original": sample_size_original,
                    "hop_length_target": self.hop_length,
                    "hop_length_original": hop_length_original,
                    "sample_start_frame": sample_start_frame,
                    "sample_end_frame": sample_end_frame,
                    "sample_start_frame_resampled": sample_start_frame_resampled,
                    "sample_end_frame_resampled": sample_end_frame_resampled,
                    "sample_start_time": sample_start_time,
                    "sample_end_time": sample_end_time,
                    "sample_start_time_resampled": sample_start_time_resampled,
                    "sample_end_time_resampled": sample_end_time_resampled,
                    "breathing_rate": segment["breathing_rate"],
                    "bbox_sequence": segment["bbox_sequence"],
                    "lado": segment["lado"],
                    "segment_labels": segment["segment_labels"],
                }
            )
        return samples

    def _prune_invalid(self) -> None:
        """Prune invalid samples."""
        if self.bbox_transform:
            # Remove samples with no bounding box sequence
            self.samples = self.samples[
                self.samples["bbox_sequence"].apply(lambda x: x is not None)
            ]
            # Remove samples with empty bounding box sequence
            self.samples = self.samples[
                self.samples["bbox_sequence"].apply(lambda x: len(x) > 0)
            ]
            # Remove samples with bounding box sequence outside range
            self.samples = self.samples[
                self.samples.apply(
                    lambda x: len(
                        x["bbox_sequence"][
                            x["sample_start_frame"] : x["sample_end_frame"]
                        ]
                    )
                    > 0,
                    axis=1,
                )
            ]
            if self.bbox_transform_corners:
                # Remove samples without corner annotation
                self.samples = self.samples[
                    self.samples["lado"].apply(lambda x: x is not None)
                ]
        if not self.classification:
            # Remove samples with no breathing rate
            self.samples = self.samples[
                self.samples["breathing_rate"].apply(lambda x: x is not None)
            ]
            # Remove samples with NaN breathing rate
            self.samples = self.samples[
                self.samples["breathing_rate"].apply(lambda x: not np.isnan(x))
            ]
            # Remove samples with invalid breathing rate
            self.samples = self.samples[
                self.samples["breathing_rate"].apply(lambda x: x > 0.0)
            ]
            # Remove "Not OK" samples
            self.samples = self.samples[
                self.samples["segment_labels"].apply(lambda x: "NOT OK" not in x)
            ]
        else:
            # Remove samples with no labels
            self.samples = self.samples[
                self.samples["segment_labels"].apply(lambda x: x is not None)
            ]
            # Remove samples with empty labels
            self.samples = self.samples[
                self.samples["segment_labels"].apply(lambda x: len(x) > 0)
            ]
        # drop_indices: list[int] = []
        # # Try to iterate over all samples
        # # If an exception is raised, print the exception and the sample
        # for idx in range(len(self)):  # pylint: disable=consider-using-enumerate
        #     try:
        #         self[idx]
        #     except Exception:
        #         drop_indices.append(idx)
        #     # Print progress every 1% of samples
        #     if idx % (len(self) // 100) == 0:
        #         print(
        #             f"Pruning invalid samples: {idx}/{len(self)} ({len(drop_indices)} invalid)"
        #         )
        # # Drop invalid samples
        # self.samples = self.samples.drop(index=drop_indices)
        # # Reset index
        # self.samples = self.samples.reset_index(drop=True)
        # # Raise warning
        # warnings.warn(f"Pruned {len(drop_indices)} invalid samples.")

    def _download_task_video(
        self, annotation: pd.Series, overwrite: bool = False
    ) -> None:
        """Download task video."""
        # Get task
        filename = annotation["data"].split("/")[-1]
        # Check if video is already downloaded
        if os.path.exists(os.path.join(self.data_dir, filename)) and not overwrite:
            return
        # Download video
        self.api.download_task_video(task_id=annotation["id"])

    def _interpolate_bboxes(
        self,
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
                    start_frame = start_bbox["frame"]
                    end_frame = end_bbox["frame"]
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
            last_bbox_count: int = 0
            while len(interpolated_bboxes) < num_frames:
                interpolated_bboxes.append(
                    {
                        "x": bboxes[-1]["x"],
                        "y": bboxes[-1]["y"],
                        "time": bboxes[-1]["time"] + last_bbox_count / fps,  # type: ignore
                        "frame": last_bbox_frame + 1,
                        "width": bboxes[-1]["width"],
                        "height": bboxes[-1]["height"],
                        "rotation": bboxes[-1]["rotation"],
                    }
                )
                last_bbox_frame += 1
                last_bbox_count += 1
            if len(interpolated_bboxes) > num_frames:
                interpolated_bboxes = interpolated_bboxes[:num_frames]
            # Raise exception if the number of interpolated bounding boxes is not equal to
            # the number of frames
            # assert len(interpolated_bboxes) == num_frames
            # Raise exception if the interpolated bounding boxes are not sorted by frame
            # assert all(
            #     interpolated_bboxes[i]["frame"] <= interpolated_bboxes[i + 1]["frame"]
            #     for i in range(len(interpolated_bboxes) - 1)
            # )
            # return the interpolated bounding boxes
            return interpolated_bboxes
        raise NotImplementedError

    def _rotate_point_around_origin(self, x, y, x0, y0, theta):
        """Rotate a point around the origin.

        Args:
            x (float): The x coordinate of the point.
            y (float): The y coordinate of the point.
            x0 (float): The x coordinate of the origin.
            y0 (float): The y coordinate of the origin.
            theta (float): The rotation angle in degrees.

        Returns:
            tuple[float, float]: The rotated point.
        """
        # Convert theta from degrees to radians
        theta_rad = math.radians(theta)
        # Translate point to origin
        x_translated = x - x0
        y_translated = y - y0

        # Perform the rotation
        x_rotated = x_translated * math.cos(theta_rad) - y_translated * math.sin(
            theta_rad
        )
        y_rotated = x_translated * math.sin(theta_rad) + y_translated * math.cos(
            theta_rad
        )

        # Translate the point back
        x2 = x_rotated + x0
        y2 = y_rotated + y0

        return x2, y2

    def _frame_to_bbox(
        self,
        frame: torch.Tensor,
        x: float,
        y: float,
        width: float,
        height: float,
        rotation: float,
        interpolation: str = "bilinear",
        percent: bool = False,
        corner: str | None = None,
        corner_ratio_w: float = 0.5,
        corner_ratio_h: float = 0.5,
    ) -> torch.Tensor:
        """Transform a frame to a bounding box.

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
            corner (str, optional): The corner of the bbox to use. Can be "top-left",
                "top-right", "bottom-left", "bottom-right". Defaults to None.
            corner_ratio_w (float, optional): The ratio of the width to use for the corner.
                Defaults to 0.5.
            corner_ratio_h (float, optional): The ratio of the height to use for the corner.
                Defaults to 0.5.
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
        # Ajustar os valores de x e y após a rotação
        x, y = self._rotate_point_around_origin(
            x, y, original_width // 2, original_height // 2, -rotation
        )
        # crop
        if corner is None:
            frame = torchvision.transforms.functional.crop(
                frame,
                top=int(y),
                left=int(x),
                height=int(height),
                width=int(width),
            )
        else:
            if corner == "top-left":
                frame = torchvision.transforms.functional.crop(
                    frame,
                    top=int(y),
                    left=int(x),
                    height=int(height * corner_ratio_h),
                    width=int(width * corner_ratio_w),
                )
            elif corner == "top-right":
                frame = torchvision.transforms.functional.crop(
                    frame,
                    top=int(y),
                    left=int(x + width * (1 - corner_ratio_w)),
                    height=int(height * corner_ratio_h),
                    width=int(width * corner_ratio_w),
                )
            elif corner == "bottom-left":
                frame = torchvision.transforms.functional.crop(
                    frame,
                    top=int(y + height * (1 - corner_ratio_h)),
                    left=int(x),
                    height=int(height * corner_ratio_h),
                    width=int(width * corner_ratio_w),
                )
            elif corner == "bottom-right":
                frame = torchvision.transforms.functional.crop(
                    frame,
                    top=int(y + height * (1 - corner_ratio_h)),
                    left=int(x + width * (1 - corner_ratio_w)),
                    height=int(height * corner_ratio_h),
                    width=int(width * corner_ratio_w),
                )
            else:
                raise ValueError(
                    f"Invalid valid for corner: {corner}."
                    "Must be one of None, 'top-left', 'top-right', 'bottom-left', 'bottom-right'."
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

    def __len__(self) -> int:
        """Get the number of samples.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        """Get a sample.

        Args:
            idx (int): The sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The sample and target.
        """
        # Get sample
        sample = self.samples.iloc[idx]
        # Read video
        video, _, info = read_video(
            filename=sample["data"],
            start_pts=sample["sample_start_time"],
            end_pts=sample["sample_end_time"] + 1,
            pts_unit="sec",
            output_format="TCHW",
        )
        # If video length is larger than sample size, crop video
        if video.shape[0] > sample["sample_size_original"]:
            video = video[: sample["sample_size_original"], :, :, :]
        # If video_length is smaller than sample size, pad video with last frames
        if video.shape[0] < sample["sample_size_original"]:
            video = torch.cat(
                (
                    video,
                    video[-1, :, :, :].repeat(
                        sample["sample_size_original"] - video.shape[0], 1, 1, 1
                    ),
                ),
                dim=0,
            )
        # assert video.shape[0] == sample["sample_size_original"]
        # Transform video to bounding box
        if self.bbox_transform:
            # Get bounding boxes
            bboxes = sample["bbox_sequence"]
            # Get bbox sequence between start and end bbox indexes
            bboxes = bboxes[sample["sample_start_frame"] : sample["sample_end_frame"]]
            # If bboxes length is smaller than sample size, pad bboxes with last bbox
            if len(bboxes) < sample["sample_size_original"]:
                bboxes += [bboxes[-1]] * (sample["sample_size_original"] - len(bboxes))
            # assert len(bboxes) == video.shape[0]
            # Transform video to bounding box
            for frame_idx, bbox in enumerate(bboxes):
                try:
                    corner = (
                        {
                            "Superior": "top-left",
                            "Inferior": "bottom-left",
                        }[sample["lado"]]
                        if self.bbox_transform_corners
                        else None
                    )
                except KeyError:
                    # If the corner is not available, set it to None
                    # Print a warning, with the sample index and id
                    warnings.warn(
                        f"Corner not available for sample {idx} with id {sample['task_id']}."
                    )
                    corner = None
                video[frame_idx, :, :, :] = self._frame_to_bbox(
                    frame=video[frame_idx, :, :, :],
                    x=bbox["x"],
                    y=bbox["y"],
                    width=bbox["width"],
                    height=bbox["height"],
                    rotation=bbox["rotation"],
                    percent=True,
                    corner=corner,
                    corner_ratio_w=(1 / 2),
                    corner_ratio_h=(2 / 3),
                )

        # Resample video
        if self.fps < info["video_fps"]:
            video = resample_video(
                video=video,
                fps=info["video_fps"],
                target_fps=self.fps,
            )
        # If video length is larger than sample size, crop video
        if video.shape[0] > sample["sample_size_target"]:
            video = video[: sample["sample_size_target"], :, :, :]
        # If video_length is smaller than sample size, pad video with last frames
        if video.shape[0] < sample["sample_size_target"]:
            video = torch.cat(
                (
                    video,
                    video[-1, :, :, :].repeat(
                        sample["sample_size_target"] - video.shape[0], 1, 1, 1
                    ),
                ),
                dim=0,
            )
        # assert video.shape[0] == sample["sample_size_target"]
        # Transform video
        if self.transform is not None:
            video = self.transform(video)
        # Get target
        if self.classification:
            target = sample["segment_labels"]
        else:
            target = sample["breathing_rate"]
        # Transform target
        if self.target_transform is not None:
            target = self.target_transform(target)
        return video, target


# %% [markdown]
# # Functions

# %%


def resample_video(
    video: torch.Tensor,
    fps: float,
    target_fps: float,
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
# # Test

# %%
if __name__ == "__main__":
    print("Running tests...")
    dataset = VideoDataset(
        url=LABEL_STUDIO_URL,
        api_key=LABEL_STUDIO_API_KEY,
        project_id=LABEL_STUDIO_PROJECT_ID,
        data_dir=LABEL_STUDIO_DOWNLOAD_DIR,
        container_id=LABEL_STUDIO_CONTAINER_ID,
        container_data_dir=LABEL_STUDIO_DOWNLOAD_DIR,
        verbose=True,
    )
    # # Iterate over dataset
    # for idx in range(len(dataset)):
    #     video, target = dataset[idx]
    #     print(f"Idx: {idx}, Video: {video.shape}, Target: {target}")
    print("Done!")

# %%
