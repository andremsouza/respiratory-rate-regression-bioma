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
        verbose: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
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
        self._check_project_id(project_id, self.verbose)
        self.project_id: int = project_id
        # If data_dir is None, use current working directory
        # If data_dir does not exist, create it
        self.data_dir: str = (
            data_dir if data_dir is not None else os.getcwd() + "/data/"
        ).replace("//", "/")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform
        # Get tasks and annotations from LabelStudio
        self.annotations: pd.DataFrame = self.api.list_project_tasks(
            self.project_id, page_size=-1, only_labeled=False, to_dataframe=False
        )[0]
        # Transform annotations
        self._transform_annotations()

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

    def _transform_annotations(self) -> None:
        pass


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
