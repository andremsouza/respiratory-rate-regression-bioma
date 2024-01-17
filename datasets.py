"""Datasets for video models."""

# %% [markdown]
# # Imports

# %%
import os
from typing import Callable

import dotenv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision

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
    def __init__(
        self,
        url: str,
        api_key: str,
        project_id: int,
        data_dir: str | None = None,
        container_id: str | None = None,
        container_data_dir: str | None = None,
        fps: float = 7.5,
        sample_size: int = 16,
        filter_ids: list[int] | None = None,
        bbox_transform: bool = False,
        trim_video: bool = False,
        classification: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        verbose: bool = False,
    ) -> None:
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
        self.data_dir: str = os.path.join(
            data_dir if data_dir is not None else os.getcwd(), "data/"
        )
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.fps: float = fps
        self.sample_size: int = sample_size
        self.filter_ids: list[int] | None = filter_ids
        self.bbox_transform: bool = bbox_transform
        self.trim_video: bool = trim_video
        self.classification: bool = classification
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform
        # Load tasks
        self._load_tasks()
        # Process tasks into samples
        self._process_tasks()
        # TODO: Implement for classification
        # TODO: Implement for regression
        # TODO: Implement bbox transform

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

    def _load_tasks(self) -> None:
        """Load tasks from LabelStudio."""
        # Get tasks and annotations from LabelStudio
        self.tasks: pd.DataFrame = self.api.list_project_tasks(
            project_id=self.project_id,
            page_size=-1,
            only_labeled=True,
            to_dataframe=True,
        )[0]
        # Filter tasks with valid (non-empty) annotations
        self.tasks = self.tasks[self.tasks["annotation_value.result"].notna()]
        # Filter ids if specified
        if self.filter_ids is not None:
            self.tasks = self.tasks[self.tasks["id"].isin(self.filter_ids)]
        # Transform annotations data column
        self.tasks["data"] = self.tasks.apply(lambda x: str(x["data"]["video"]), axis=1)
        # Get task annotations
        # Create columns with object type
        self.tasks["breathing_rate"] = np.nan
        self.tasks["bbox_sequence"] = np.nan
        self.tasks["segments"] = np.nan
        # Convert new columns to object type
        self.tasks["breathing_rate"] = self.tasks["breathing_rate"].astype("object")
        self.tasks["bbox_sequence"] = self.tasks["bbox_sequence"].astype("object")
        self.tasks["segments"] = self.tasks["segments"].astype("object")
        for idx, row in self.tasks.iterrows():
            row_result = pd.json_normalize(row["annotation_value.result"])
            # Get breathing rate from result, if available
            breathing_rate: float = 0.0
            if len(row_result[row_result["from_name"] == "breathingrate"]) > 0:
                breathing_rate = row_result[
                    row_result["from_name"] == "breathingrate"
                ].iloc[0]["value.number"]
            # Get bbox sequence from result, if available
            bbox_sequence: list[dict] = []
            if len(row_result[row_result["from_name"] == "box"]) > 0:
                bbox_sequence = row_result[row_result["from_name"] == "box"].iloc[0][
                    "value.sequence"
                ]
            # Get segments from result, if available
            segments: list[dict] = []
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
            # Set breathing rate, bbox sequence and segments in dataframe
            self.tasks.at[idx, "breathing_rate"] = breathing_rate
            self.tasks.at[idx, "bbox_sequence"] = (
                bbox_sequence if bbox_sequence else None
            )
            self.tasks.at[idx, "segments"] = segments if segments else None
        # Drop original result column
        self.tasks.drop(columns=["annotation_value.result"], inplace=True)

    def _process_tasks(self) -> None:
        # TODO: Implement processing
        raise NotImplementedError


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
    print("Done!")

# %%
